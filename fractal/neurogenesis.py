"""Scale neurogenesis — POLICY (when and how to grow); the mechanism is in unit.grow_scale.

Gradient cannot flow through shape (adding a scale is discrete), so growth is a rule OUTSIDE
the gradient, read from telemetry we already measure (see viz_serve.activity):

  trigger = the read gate persistently concentrates >conc_thresh of the mass into ONE scale
            AND the ‖W‖ of that scale SATURATES (plateau — stopped growing) → overloaded memory
            with nowhere to offload → a new (empty, greedy) scale sprouts.

Maturation: the newborn's β_gain decays toward 1.0 with time constant mature_steps (from a greedy
juvenile to a mature scale). Pruning is not addressed yet (growth only — falsifying the hypothesis
"growth helps multi-fact recall" does not need pruning).

The mechanism/policy separation is intentional: unit.grow_scale IS the mechanism, here is ONLY
the decision-making → the same controller drives training (train.py) and the live dashboard (viz_serve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def units(model):
    """Unique FractalUnits (tied → one; untied → per depth). Growth is consistent
    across all of them (the same γ ladder) so that per-depth states have the same number of scales."""
    seen = {}
    for d in range(model.cfg.depth):
        u = model.block_at(d).unit
        seen[id(u)] = u
    return list(seen.values())


@torch.no_grad()
def telemetry(model, states):
    """(share, wnorm) from the last pass. share (L,) = mean gate share per
    scale (from unit._last_share, averaged over untied units). wnorm (L,) = mean ‖W‖
    per scale across depths, or None when there are no states (plain-accum/grad-ckpt)."""
    us = units(model)
    shares = [u._last_share for u in us if u._last_share is not None]
    share = torch.stack(shares).mean(0).float().cpu() if shares else None
    wnorm = None
    if states is not None and len(states) > 0 and states[0] is not None:
        L = len(states[0].W)
        wnorm = torch.tensor([
            sum(float(st.W[l].norm()) for st in states) / len(states) for l in range(L)
        ])
    return share, wnorm


@dataclass
class NeurogenesisController:
    max_scales: int = 6
    conc_thresh: float = 0.6      # the gate must pack >this much mass into a single scale
    plateau_eps: float = 0.02     # ‖W‖ of the dominant scale "saturates" (rel. change < this much)
    cooldown: int = 150           # min iterations between births (growth must be rare)
    warmup: int = 300             # first let the ladder settle, only then consider growth
    birth_beta_gain: float = 2.0  # newborn plasticity (>1 = greedy juvenile)
    mature_steps: int = 400       # time constant of maturation β_gain → 1.0
    demote: bool = True           # demote the permanent scale on growth? (False = gentle v3)
    max_grow_step: int = 10 ** 9  # no births after this step (v3: grow only early → time to recover)
    fast_alpha: float = 0.1       # telemetry EMA (fast)
    slow_alpha: float = 0.02      # telemetry EMA (slow) — plateau = |fast−slow| small

    share_fast: torch.Tensor | None = field(default=None, init=False)
    wn_fast: torch.Tensor | None = field(default=None, init=False)
    wn_slow: torch.Tensor | None = field(default=None, init=False)
    last_grow: int = field(default=-10 ** 9, init=False)
    births: list = field(default_factory=list, init=False)

    def _ema(self, prev, cur, a):
        cur = cur.float()
        if prev is None or prev.shape != cur.shape:
            return cur.clone()
        return (1 - a) * prev + a * cur

    def observe(self, share, wnorm):
        if share is not None:
            self.share_fast = self._ema(self.share_fast, share, self.fast_alpha)
        if wnorm is not None:
            self.wn_fast = self._ema(self.wn_fast, wnorm, self.fast_alpha)
            self.wn_slow = self._ema(self.wn_slow, wnorm, self.slow_alpha)

    def _decision(self, step, n_scales):
        """Returns (grow?, info) — info describes WHY (for a readable log/falsification)."""
        if (n_scales >= self.max_scales or step < self.warmup or step > self.max_grow_step
                or step - self.last_grow < self.cooldown):
            return False, None
        if self.share_fast is None or self.wn_slow is None:
            return False, None
        if self.share_fast.shape[0] != n_scales or self.wn_slow.shape[0] != n_scales:
            return False, None                        # telemetry still from a different number of scales
        d = int(self.share_fast.argmax())
        conc = float(self.share_fast[d])
        slope = abs(float(self.wn_fast[d]) - float(self.wn_slow[d])) / (abs(float(self.wn_slow[d])) + 1e-6)
        info = {"step": step, "dominant": d, "conc": round(conc, 3),
                "wn_slope": round(slope, 4), "wn": round(float(self.wn_slow[d]), 3)}
        return (conc > self.conc_thresh and slope < self.plateau_eps), info

    def mature(self, model):
        """Maturation: each cell's β_gain approaches 1.0 (exp. relaxation with constant mature_steps)."""
        k = math.exp(-1.0 / max(self.mature_steps, 1))
        for u in units(model):
            for c in u.cells:
                g = float(c.beta_gain)
                if g != 1.0:
                    c.set_beta_gain(1.0 + (g - 1.0) * k)

    def maybe_grow(self, model, step):
        """Consider and possibly perform a birth. Returns an info dict (incl. "new_params" for add_param_group), or None."""
        grow, info = self._decision(step, model.cfg.n_scales)
        if not grow:
            return None
        gamma, new_params = None, []
        for u in units(model):
            gamma, ps = u.grow_scale(birth_beta_gain=self.birth_beta_gain, demote=self.demote)
            new_params += ps
        model.cfg.n_scales += 1
        self.last_grow = step
        self.share_fast = self.wn_fast = self.wn_slow = None   # dimensions changed → re-warm up
        info.update({"new_n_scales": model.cfg.n_scales, "birth_gamma": round(float(gamma), 4)})
        self.births.append(dict(info))                          # log without tensors
        info["new_params"] = new_params                         # params only in the return (not in the log)
        return info
