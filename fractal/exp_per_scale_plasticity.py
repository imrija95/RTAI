"""Prove-in-small A/B(/C/D): PER-SCALE PLASTICITY policies vs a global plasticity curve.

Question (from the plasticity-curve discussion): does coupling *how much each part adapts* to the
fractal timescale help recall? A normal LM has one global learning-rate curve; the fractal ladder
gives a per-scale knob (per-scale W0 LR + per-scale fast-weight beta_gain).

Controlled comparison — identical init, identical batches every step, identical optimizer STRUCTURE
(W0 always in per-scale param groups; the baseline just sets every multiplier to 1). Only the
per-scale plasticity POLICY differs:
  * none        : global cosine LR everywhere; beta_gain = 1 on every scale (the baseline).
  * cool_perm   : fast scales stay plastic (beta_gain 1->2), the PERMANENT scale's W0 freezes early
                  and its beta_gain stays 1 (consolidate the permanent store). [first attempt]
  * cool_fast   : the OPPOSITE — the permanent scale stays plastic (beta_gain 1->1.5, W0 full LR),
                  the FAST scales' W0 freeze early and their beta_gain stays 1. [variant A]
  * gate_driven : plasticity FOLLOWS measured gate usage (EMA) — scales the model actually routes
                  through get more plasticity, unused scales cool. Self-tuning, no fixed prior. [variant B]

Evals (VIBE #9/#10) — recall count alone is a weak benchmark (single-fact saturates), so we add
SOFTER and ORTHOGONAL probes, all on HELD-OUT (never-trained) values:
  * decay        : 1-fact accuracy across growing filler distance (16..400) — retention.
  * capacity     : accuracy across growing #facts (1..16) at fixed distance — the real stress.
  * soft quality : mean log-prob and P(correct answer) at a hard setting — sensitive at ceiling.
  * overwrite    : teach key->v1 then key->v2, query -> expect v2 — memory UPDATE, a different skill.

Honest caveats: beta_gain is a non-persistent buffer (resets on reload) → treatment states are
evaluated in-process (same as the repo's evals). The per-scale slow-weight surface is only W0
(projections/MLP are shared by design, VIBE #5), so the fast-weight (beta_gain) arm carries most of
the effect.

Run: uv run python -m fractal.exp_per_scale_plasticity --iters 3000
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch

from fractal.model import Config, FractalLM
from fractal.recall import RecallGen, PREFIXES
from fractal import tokenizer as tk

POLICIES = ["none", "cool_perm", "cool_fast", "gate_driven"]


def _cos_decay(t, T):   # 1 -> 0
    return 0.5 * (1 + math.cos(math.pi * min(t, T) / T))


def _ramp(t, T):        # 0 -> 1
    return 0.5 * (1 - math.cos(math.pi * min(t, T) / T))


def _freeze_early(t, T):  # 1 -> 0 by the HALFWAY point, then 0 (early consolidation)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, 2.0 * t / T)))


def plasticity(policy, t, T, L, gate_share):
    """Return (w0_lr_mult[L], beta_gain[L]) for this step under `policy`.
    gate_share: list length L of the EMA gate usage per scale (or None early)."""
    perm = L - 1
    if policy == "none":
        return [1.0] * L, [1.0] * L
    if policy == "cool_perm":
        mult = [(_freeze_early(t, T) if l == perm else 1.0) for l in range(L)]
        gain = [(1.0 if l == perm else 1.0 + _ramp(t, T)) for l in range(L)]
        return mult, gain
    if policy == "cool_fast":
        mult = [(1.0 if l == perm else _freeze_early(t, T)) for l in range(L)]
        gain = [(1.0 + 0.5 * _ramp(t, T) if l == perm else 1.0) for l in range(L)]
        return mult, gain
    if policy == "gate_driven":
        if gate_share is None:
            return [1.0] * L, [1.0] * L
        mx = max(gate_share) or 1.0
        mult = [max(0.2, min(1.0, s / mx)) for s in gate_share]     # most-used scale = full LR, others cool
        gain = [1.0 + s for s in gate_share]                        # used scales more plastic (~1..1.5)
        return mult, gain
    raise ValueError(policy)


def _make_opt(model, lr):
    """Per-scale W0 param groups (group 0 = shared, groups 1..L = each scale's W0), so a per-scale LR
    multiplier actually changes the step (AdamW is invariant to a constant GRAD scale)."""
    L, depth = model.cfg.n_scales, model.cfg.depth

    def _uniq(ps):                       # weight-tied recursion shares one block → dedupe W0 by id
        seen, out = set(), []
        for p in ps:
            if id(p) not in seen:
                seen.add(id(p)); out.append(p)
        return out

    w0 = [_uniq(model.block_at(d).unit.cells[l].W0 for d in range(depth)) for l in range(L)]
    w0_ids = {id(p) for g in w0 for p in g}
    shared = [p for p in model.parameters() if id(p) not in w0_ids]
    groups = [{"params": shared, "lr": lr}] + [{"params": w0[l], "lr": lr} for l in range(L)]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95)), {l: 1 + l for l in range(L)}


def train_one(tag, model, opt, w0_grp, batches, lr0, iters, block, gate_lambda, policy, dev):
    L = model.cfg.n_scales
    model.train()
    model.block.unit._log_share = True                 # cheap gate-usage telemetry (no extra pass)
    ema = None
    for t, (x, y, w) in enumerate(batches):
        x, y = x.to(dev), y.to(dev)
        w = w.to(dev) if w is not None else None        # LM task: no per-token mask
        lr = lr0 * _cos_decay(t, iters)
        mult, gain = plasticity(policy, t, iters, L, ema)
        for d in range(model.cfg.depth):
            for l in range(L):
                model.block_at(d).unit.cells[l].set_beta_gain(gain[l])
        opt.param_groups[0]["lr"] = lr
        for l in range(L):
            opt.param_groups[w0_grp[l]]["lr"] = lr * mult[l]
        opt.zero_grad(set_to_none=True)
        _, loss, _, _ = model(x, targets=y, loss_weight=w)
        aux = model.block.unit.pop_gate_balance_loss()
        if aux is not None:
            loss = loss + gate_lambda * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sh = model.block.unit._last_share                       # (L,) usage after this forward
        if sh is not None:
            cur = sh.tolist()
            ema = cur if ema is None else [0.9 * e + 0.1 * c for e, c in zip(ema, cur)]
        if t % max(iters // 4, 1) == 0 or t == iters - 1:
            print(f"  [{tag:<11}] iter {t:4d}  loss {loss.item():.4f}", flush=True)


# ---------------- richer, cheap eval suite (all paired: identical episodes across arms/seeds) -------

# (label, kind, n_facts, distance, overwrite) — kept small and legible on purpose
EVAL_SPECS = [
    ("recall D=16   (short)", "acc", 1, 16, False),
    ("recall D=48   (mid)",   "acc", 1, 48, False),
    ("recall D=200  (long)",  "acc", 1, 200, False),
    ("recall D=400  (v.long)", "acc", 1, 400, False),
    ("recall 4-facts",        "acc", 4, 48, False),
    ("logp 4-facts",          "logp", 4, 48, False),
    ("logp 8-facts",          "logp", 8, 96, False),
    ("overwrite (update)",    "acc", 1, 48, True),
]


def _draw(rg, n_facts, distance, overwrite):
    tpl = random.choice(PREFIXES)
    names = random.sample(rg.names, min(n_facts, len(rg.names)))
    pool = rg.test_vals
    vals = [random.choice(pool) for _ in names]
    facts = []
    for nm, vt in zip(names, vals):
        facts += rg._e(tpl.format(n=nm)) + [vt]
    j = random.randrange(len(names))
    ans = vals[j]
    if overwrite:                          # re-teach the queried name with a NEW value
        ans = random.choice(pool)
        facts += rg._e(tpl.format(n=names[j])) + [ans]
    prompt = facts + rg._filler(distance) + rg._e(tpl.format(n=names[j]))
    return prompt, ans


@torch.no_grad()
def _score(model, prompt, ans, dev):
    idx = torch.tensor([prompt], dtype=torch.long, device=dev)
    lg = model(idx)[0][0, -1]
    logp = torch.log_softmax(lg, -1)
    return int(lg.argmax().item() == ans), float(logp[ans])


@torch.no_grad()
def evaluate(models, rg, dev, n=128):
    """Return [(label, kind, [value per model])] on FIXED held-out episodes (same across seeds, so
    the cross-seed variance reflects TRAINING, not eval sampling)."""
    for m in models:
        m.eval()
    out = []
    for i, (label, kind, nf, dist, ov) in enumerate(EVAL_SPECS):
        nf_ = min(nf, len(rg.names))
        random.seed(5000 + i)
        draws = [_draw(rg, nf_, dist, ov) for _ in range(n)]
        vals = []
        for m in models:
            acc, lp = 0, 0.0
            for prompt, ans in draws:
                c, l = _score(m, prompt, ans, dev)
                acc += c; lp += l
            vals.append(lp / n if kind == "logp" else acc / n * 100.0)
        out.append((label, kind, vals))
    return out


@torch.no_grad()
def eval_lm(models, args, dev, nb=20):
    """LM task metric: held-out val perplexity (lower = better). Fixed val batches across arms."""
    from fractal.data import get_batch
    for m in models:
        m.eval()
    random.seed(9999); np.random.seed(9999); torch.manual_seed(9999)
    vb = [get_batch("val", args.batch, args.block, dev, data_dir=args.data_dir) for _ in range(nb)]
    out = []
    for m in models:
        tot = 0.0
        for x, y in vb:
            tot += m(x, targets=y)[1].item()
        out.append(math.exp(tot / nb))
    return [("val perplexity", "ppl", out)]


def _agg(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd


def print_table(all_runs, policies):
    """all_runs: list over seeds of evaluate() outputs. Print mean±std, mark the best per row with *."""
    w = 14
    print(f"\n{'metric':<24}" + "".join(f"{p:>{w}}" for p in policies))
    print("-" * (24 + w * len(policies)))
    kinds = {m[1] for m in all_runs[0]}
    for i, (label, kind, _) in enumerate(all_runs[0]):
        stats = [_agg([run[i][2][pi] for run in all_runs]) for pi in range(len(policies))]
        higher_better = kind in ("acc", "logp")            # ppl: lower = better
        best = (max if higher_better else min)(range(len(policies)), key=lambda pi: stats[pi][0])
        cells = ""
        for pi, (m, sd) in enumerate(stats):
            if kind == "ppl":
                txt = f"{m:6.1f}±{sd:.1f}"
            elif kind == "logp":
                txt = f"{m:5.2f}±{sd:4.2f}"
            else:
                txt = f"{m:3.0f}±{sd:2.0f}%"
            cells += f"{('*' + txt if pi == best else txt):>{w}}"
        print(f"{label:<24}{cells}")
    print(f"\nLEGEND  (mean ± std over {len(all_runs)} seed(s); * = best in its row)")
    if "ppl" in kinds:
        print("  val perplexity: held-out language-modeling perplexity — LOWER = better (general fluency).")
    if kinds & {"acc", "logp"}:
        print("  recall D=N  : % correct recall of an UNSEEN value, N filler tokens between fact & query")
        print("               (short/mid/long/v.long = growing memory distance). Higher = better.")
        print("  recall 4-facts: % correct when 4 facts compete in one episode (capacity pressure).")
        print("  logp        : mean log-prob of the correct answer (higher/less-negative = more confident);")
        print("                sensitive even when accuracy saturates.")
        print("  overwrite   : teach a value, then RE-teach a new one, then query — % returning the NEW value")
        print("                (tests memory UPDATE, not just first-write).")


def run_seed(seed, cfg, args, policies, rg, dev):
    """Train every arm for one seed (identical init + identical batch stream) and return evaluate()."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    base = FractalLM(cfg).to(dev)                       # one init; every arm starts from it
    init = base.state_dict()
    models = []
    for p in policies:
        m = base if p == policies[0] else FractalLM(cfg).to(dev)
        if p != policies[0]:
            m.load_state_dict(init)
        m.block.unit._log_gate = True
        if args.gate_prior_perm > 0:
            m.block.unit.gate_prior_perm = args.gate_prior_perm
        models.append(m)

    # ONE shared batch stream (on CPU to save VRAM); all arms see identical data
    torch.manual_seed(seed + 7); random.seed(seed + 7); np.random.seed(seed + 7)
    if args.task == "lm":                                    # plain language modeling on the corpus
        from fractal.data import get_batch
        batches = [(*get_batch("train", args.batch, args.block, "cpu", data_dir=args.data_dir), None)
                   for _ in range(args.iters)]
    else:                                                    # associative recall episodes (masked answer)
        batches = [rg.batch(args.batch, args.block, "cpu", w_ans=args.w_ans, max_facts=args.max_facts)
                   for _ in range(args.iters)]
    for p, m in zip(policies, models):
        print(f"  -- {p} --", flush=True)
        opt, w0_grp = _make_opt(m, args.lr)
        train_one(p, m, opt, w0_grp, batches, args.lr, args.iters, args.block, args.gate_lambda, p, dev)
    return eval_lm(models, args, dev) if args.task == "lm" else evaluate(models, rg, dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_scales", type=int, default=3)
    ap.add_argument("--gate_lambda", type=float, default=0.03)
    ap.add_argument("--gate_prior_perm", type=float, default=0.5)
    ap.add_argument("--w_ans", type=float, default=5.0)
    ap.add_argument("--max_facts", type=int, default=8)
    ap.add_argument("--n_names", type=int, default=30, help="key pool (bigger = real capacity pressure)")
    ap.add_argument("--tokenizer", type=str, default="fractal_tokenizer.json")
    ap.add_argument("--policies", type=str, default=",".join(POLICIES))
    ap.add_argument("--task", type=str, default="recall", choices=["recall", "lm"],
                    help="recall = associative recall episodes; lm = plain language modeling on the corpus")
    ap.add_argument("--data_dir", type=str, default="fractal_data", help="corpus dir for the lm task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds to average over (base seed + 0..N-1)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = tk.load(args.tokenizer)
    cfg = Config(vocab_size=tok.get_vocab_size(), n_embd=args.n_embd, n_head=8, depth=args.depth,
                 n_scales=args.n_scales, tau0=16.0, rho=4.0)
    policies = [p for p in args.policies.split(",") if p]
    rg = RecallGen(tok, n_names=args.n_names) if args.task == "recall" else None
    print(f"[exp] device={dev} task={args.task} vocab={tok.get_vocab_size()} | n_embd={args.n_embd} "
          f"depth={args.depth} n_scales={args.n_scales} | {args.iters} iters × {len(policies)} arms "
          f"× {args.seeds} seed(s): {policies}", flush=True)

    all_runs = []
    for s in range(args.seeds):
        print(f"\n########## SEED {args.seed + s} ##########", flush=True)
        all_runs.append(run_seed(args.seed + s, cfg, args, policies, rg, dev))

    print("\n================ HELD-OUT EVAL — unseen values ================")
    print_table(all_runs, policies)
    print("\n(honest: single synthetic recall task, tiny model — VALIDATES the mechanism per VIBE #11, "
          "not the magnitude at scale. beta_gain non-persistent → in-process eval; per-scale slow surface = W0.)")


if __name__ == "__main__":
    main()
