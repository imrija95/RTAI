"""Eval kit — CHEAP diagnostics on a TRAINED checkpoint (no training, seconds).

Says WHY the model can(not) recall — better than one noisy accuracy number from a 30-min training run.
Everything runs on a finished `.pt` via inference; feel free to run it on CPU alongside other work.

  uv run python -m fractal.probe <ckpt> [--n_names 256] [--dist 48] [--facts 1,3,6] [--n 64]

What it measures (and what it tells you in plain terms):
  1) ABLATION — recall when the model reads from ONLY one scale (of memory). Does any one work alone,
                or is recall a team effort of several scales? (force_scale, no training.)
  2) GATE@ANSWER — where the model looks EXACTLY at the moment the answer is produced (not averaged over text).
  3) KEYS     — how distinguishable the queries are for different names (pairwise cos; ~1.0 = "key collapse").

Interpretation: when the best SINGLE scale ≪ normal → recall is an emergent binding of several scales
(store + fresh context), not pure store → adding/redirecting scales will not help.
"""

from __future__ import annotations

import argparse

import torch

from fractal import persist, tokenizer as tk
from fractal.recall import RecallGen, PREFIXES


def _units(m):
    seen = {}
    for d in range(m.cfg.depth):
        u = m.block_at(d).unit
        seen[id(u)] = u
    return list(seen.values())


def _set_force(m, s):
    for u in _units(m):
        u.force_scale = s


@torch.no_grad()
def ablation(m, rg, dev, dist, facts, n):
    """Recall (%) when reading from only one scale. Key test: does any one work ALONE?"""
    L = m.cfg.n_scales
    g = m.block.unit.gammas
    print("=== 1) ABLATION: recall (%) when reading from ONLY one scale (held-out values) ===")
    print(f"{'mode':26s} " + " ".join(f"{nf}f" for nf in facts))
    rows = [("normal (gate mixes)", None)] + [(f"only L{l} (γ={g[l]:.3f})", l) for l in range(L)]
    best_single = 0
    for label, fs in rows:
        _set_force(m, fs)
        accs = [round(rg.accuracy(m, dist, dev, n=n, held_out=True, n_facts=nf) * 100) for nf in facts]
        if fs is not None:
            best_single = max(best_single, max(accs))
        print(f"{label:26s} " + " ".join(f"{a:2d}" for a in accs))
    _set_force(m, None)
    normal = [round(rg.accuracy(m, dist, dev, n=n, held_out=True, n_facts=nf) * 100) for nf in facts]
    verdict = ("→ no single scale is enough on its own → recall is a TEAM EFFORT of several scales (store+context)"
               if best_single < max(normal) - 10 else
               "→ one scale almost suffices → recall is (almost) pure store")
    print(verdict)


@torch.no_grad()
def gate_at_answer(m, rg, dev, dist, n):
    """Where the gate reads EXACTLY at the answer position (last token of the prompt), not averaged over text."""
    L, H = m.cfg.n_scales, m.cfg.n_head
    cap = []
    hs = [u.gate.register_forward_hook(lambda mod, i, o: cap.append(o.detach())) for u in _units(m)]
    _set_force(m, None)
    _ = rg.accuracy(m, dist, dev, n=n, held_out=True, n_facts=min(6, len(rg.names)))
    for h in hs:
        h.remove()
    gates = []
    for o in cap:                                        # o: (B,T,H*L)
        B, T, _ = o.shape
        gates.append(o.view(B, T, H, L).softmax(-1)[:, -1].mean(dim=(0, 1)))   # (L,) at the answer position
    ga = torch.stack(gates).mean(0)
    print("\n=== 2) GATE AT THE ANSWER POSITION (where the model reads while producing the answer) ===")
    print("  " + " · ".join(f"L{l}(γ={m.block.unit.gammas[l]:.3f}) {float(ga[l])*100:.0f}%" for l in range(L)))


@torch.no_grad()
def key_separability(m, rg, dev, n_names=48):
    """How distinguishable the QUERIES are for different names. High pairwise cos (~1) = key collapse."""
    names = rg.names[:min(n_names, len(rg.names))]
    tpl = PREFIXES[0]                                    # same template, only the name changes
    ids = [rg.tok.encode(tpl.format(n=nm)).ids for nm in names]
    Ln = min(len(x) for x in ids)
    batch = torch.tensor([x[:Ln] for x in ids], dtype=torch.long, device=dev)   # (N, Ln)
    E, H, hd = m.cfg.n_embd, m.cfg.n_head, m.cfg.head_dim
    cap = []
    h = m.block.unit.to_qk.register_forward_hook(lambda mod, i, o: cap.append(o.detach()))
    _ = m(batch)
    h.remove()
    q = cap[0][:, -1, :E].view(-1, H, hd)                # query at the "is" position, (N,H,hd)
    q = torch.nn.functional.normalize(q, dim=-1)
    cos = torch.einsum("nhd,mhd->nmh", q, q).mean(-1)    # (N,N) averaged over heads
    N = cos.shape[0]
    off = cos[~torch.eye(N, dtype=torch.bool, device=dev)]
    print("\n=== 3) KEY (query) DISTINGUISHABILITY across names — pairwise cos ===")
    print(f"  names: {N} | mean {off.mean():.3f} | max {off.max():.3f}  "
          f"({'COLLAPSE (~1 → names indistinguishable)' if off.mean() > 0.9 else 'separable'})")


def main():
    ap = argparse.ArgumentParser(description="Cheap recall diagnostics on a finished checkpoint.")
    ap.add_argument("ckpt")
    ap.add_argument("--n_names", type=int, default=256)
    ap.add_argument("--dist", type=int, default=48)
    ap.add_argument("--facts", type=str, default="1,3,6")
    ap.add_argument("--n", type=int, default=64, help="episodes per measurement (more = less noise)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = persist.load_model(args.ckpt, dev)
    m.eval()
    rg = RecallGen(tk.load("fractal_tokenizer.json"), n_names=args.n_names)
    facts = [int(x) for x in args.facts.split(",")]
    print(f"\nckpt {args.ckpt.split('/')[-1]} | n_scales={m.cfg.n_scales} "
          f"| γ={[round(g, 3) for g in m.block.unit.gammas]} | dev={dev}\n")
    ablation(m, rg, dev, args.dist, facts, args.n)
    gate_at_answer(m, rg, dev, args.dist, args.n)
    key_separability(m, rg, dev)


if __name__ == "__main__":
    main()
