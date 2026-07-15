"""Per-scale, usage-driven plasticity — the DEFAULT learning policy for the fractal ladder.

Instead of one global learning-rate curve for every weight, couple *how much each timescale adapts*
to *how much the gate actually routes through it*. Validated prove-in-small (docs/EXPERIMENTS.md,
2026-07-06/07): robustly improves recall of unseen facts and — the standout — COLLAPSES seed-to-seed
variance, at no cost to language-modeling perplexity. Two coupled knobs, both driven by an EMA of the
per-scale gate usage `share` (mean over blocks of the gate softmax mass on each scale):

  * slow weights: each scale's birth weight W0 gets a per-scale LR = base_lr · clamp(share/max_share,
    floor, 1). The scale the model relies on keeps full plasticity; rarely-used scales cool. Applied via
    per-scale optimizer PARAM GROUPS (AdamW is invariant to a constant gradient scale, so only a
    per-group LR actually changes the step).
  * fast weights: each scale's write gain beta_gain = 1 + share (scales the model uses absorb more
    greedily during the forward pass).

This is a TRAINING-TIME schedule (like the LR schedule): its effect is baked into the learned weights.
The final per-scale beta_gain is persisted with the checkpoint (see persist.save_model) so that
inference matches training. Mutually exclusive with neurogenesis, which owns beta_gain and grows the
number of scales (which would invalidate the fixed per-scale param groups) — train.py disables one.

Mirrors the NeurogenesisController lifecycle: build the optimizer via `param_groups()`, then each step
call `apply()` before the forward (sets beta_gain + per-scale LRs from the running usage) and `update()`
after the forward (refreshes the usage EMA from the just-computed gate reads — no extra pass, VIBE #2).
"""

from __future__ import annotations

import torch


def _units(model):
    """The distinct compute units (one if weight-tied, one per depth if untied)."""
    n = model.cfg.depth if model.cfg.untie else 1
    return [model.blocks[d].unit for d in range(n)]


class PlasticityController:
    def __init__(self, model, ema_decay: float = 0.9, w0_floor: float = 0.2):
        self.L = model.cfg.n_scales
        self.decay = ema_decay
        self.floor = w0_floor
        self.ema = None                       # running per-scale gate usage (len L)
        self.grp_index = {}
        for u in _units(model):
            u._log_share = True               # cheap telemetry: unit._last_share filled each forward
        # unique W0 tensor(s) per scale (weight-tied → one per scale; untied → one per block)
        self.w0_by_scale = []
        for l in range(self.L):
            seen, ps = set(), []
            for u in _units(model):
                p = u.cells[l].W0
                if id(p) not in seen:
                    seen.add(id(p)); ps.append(p)
            self.w0_by_scale.append(ps)

    def param_groups(self, model, lr: float):
        """Optimizer param groups: group 0 = everything shared; groups 1..L = each scale's W0."""
        w0_ids = {id(p) for grp in self.w0_by_scale for p in grp}
        shared = [p for p in model.parameters() if id(p) not in w0_ids]
        groups = [{"params": shared, "lr": lr}]
        for l in range(self.L):
            self.grp_index[l] = len(groups)
            groups.append({"params": self.w0_by_scale[l], "lr": lr})
        return groups

    def _share(self, model):
        vals = [u._last_share for u in _units(model) if u._last_share is not None]
        if not vals:
            return None
        return torch.stack(vals).mean(0)      # (L,) usage averaged over blocks

    def apply(self, model, opt, base_lr: float):
        """Before the forward: set per-scale W0 LRs and per-scale beta_gain from the current usage EMA."""
        share = self.ema
        opt.param_groups[0]["lr"] = base_lr
        mx = max(share) if share else 1.0
        for l in range(self.L):
            mult = 1.0 if not share else max(self.floor, min(1.0, share[l] / (mx or 1.0)))
            if l in self.grp_index:
                opt.param_groups[self.grp_index[l]]["lr"] = base_lr * mult
        for u in _units(model):
            for l in range(self.L):
                u.cells[l].set_beta_gain(1.0 + (share[l] if share else 0.0))

    def update(self, model):
        """After the forward: refresh the usage EMA from the just-computed gate reads."""
        s = self._share(model)
        if s is not None:
            cur = s.tolist()
            self.ema = cur if self.ema is None else \
                [self.decay * e + (1 - self.decay) * c for e, c in zip(self.ema, cur)]

    def state_dict(self):
        return {"ema": self.ema}

    def load_state_dict(self, st):
        self.ema = st.get("ema")
