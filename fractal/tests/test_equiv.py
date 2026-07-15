"""Numerical-agreement test of the atom: recurrent ≡ chunk-parallel delta-rule.

Load-bearing guardrail. delta_recurrent is the readable reference; the chunk path is fast but
tangled. It must produce the same output and final W (within 1e-4 fp32), for γ=1 and γ<1.

T=200 (= 3 full blocks C=64 + a remainder block of 8) is intentional: the boundary decay γ^c in the
decay kernel only shows up ACROSS multiple blocks — on a single block an error would slip through.

Run:  uv run python -m fractal.tests.test_equiv
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fractal.cell import delta_recurrent, delta_chunk, delta_chunk_decay


def _inputs(B=2, H=4, T=200, hd=48, seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, hd), dim=-1)
    k = F.normalize(torch.randn(B, H, T, hd), dim=-1)
    v = torch.randn(B, H, T, hd)
    beta = torch.sigmoid(torch.randn(B, H, T))
    W0 = torch.randn(B, H, hd, hd) * 0.1
    return q, k, v, beta, W0


def _check(gamma, C=64):
    q, k, v, beta, W0 = _inputs()
    o_r, W_r, _ = delta_recurrent(q, k, v, beta, W0.clone(), gamma=gamma)
    o_c, W_c = delta_chunk_decay(q, k, v, beta, W0.clone(), C, gamma=gamma)
    return (o_r - o_c).abs().max().item(), (W_r - W_c).abs().max().item()


def main() -> None:
    ok = True

    print("recurrent ≡ decay-chunk (out / W_final):")
    for gamma in (1.0, 0.99, 0.95, 0.9):
        d_out, d_W = _check(gamma)
        good = d_out < 1e-4 and d_W < 1e-4
        ok &= good
        print(f"  γ={gamma:<5} max|Δout|={d_out:.2e}  max|ΔW|={d_W:.2e}  {'OK' if good else 'FAIL'}")

    # bonus: unification — the decay kernel at γ=1 ≡ the original delta_chunk
    q, k, v, beta, W0 = _inputs()
    o1, W1 = delta_chunk(q, k, v, beta, W0.clone(), 64, 1.0)
    o2, W2 = delta_chunk_decay(q, k, v, beta, W0.clone(), 64, 1.0)
    d = max((o1 - o2).abs().max().item(), (W1 - W2).abs().max().item())
    good = d < 1e-6
    ok &= good
    print(f"unification: delta_chunk(γ=1) ≡ delta_chunk_decay(γ=1)  max|Δ|={d:.2e}  {'OK' if good else 'FAIL'}")

    print("\nOK — all paths agree" if ok else "\nFAIL — paths diverged!")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
