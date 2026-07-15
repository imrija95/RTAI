"""Integration tests of the fractal model.

1) stream ≡ full: the streaming run (token by token, the way chat.py chats) must produce
   the same logits as a full pass. Correctness of the persistent memory rests on this.
2) unrolling: the model can run with a different depth than it was trained at (generative rule).
3) learns: a short overfit of a fixed batch → loss falls → forward+backward+optim are wired correctly.

Run:  uv run python -m fractal.tests.test_model
"""

from __future__ import annotations

import torch

from fractal.model import Config, FractalLM


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stream_equals_full() -> float:
    torch.manual_seed(0)
    dev = _device()
    cfg = Config(vocab_size=64, n_embd=32, n_head=4, depth=2, n_scales=3,
                 tau0=8.0, rho=4.0, chunk_size=16)
    m = FractalLM(cfg).to(dev).eval()
    B, T = 2, 40
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    with torch.no_grad():
        st0 = m.init_states(B, dev)
        logits_full, _, _, _ = m(idx, states=[s.clone() for s in st0])
        st = [s.clone() for s in st0]
        outs = []
        for t in range(T):
            lg, st = m.forward_stream(idx[:, t:t + 1], st)
            outs.append(lg)
        logits_stream = torch.cat(outs, dim=1)
    d = (logits_full - logits_stream).abs().max().item()
    print(f"[stream≡full]   max|Δlogits| = {d:.2e}")
    return d


def test_variable_depth() -> None:
    """A model trained/built with depth=D can be unrolled to a different depth without crashing."""
    dev = _device()
    cfg = Config(vocab_size=64, n_embd=32, n_head=4, depth=3, n_scales=2, chunk_size=16)
    m = FractalLM(cfg).to(dev).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12), device=dev)
    with torch.no_grad():
        for d in (1, 3, 6):
            st = m.init_states(1, dev, depth=d)
            logits, _, _, _ = m(idx, states=st, depth=d)
            assert logits.shape == (1, 12, cfg.vocab_size)
    print("[variable-depth] unrolling depth∈{1,3,6} OK")


def _learns() -> float:
    """Overfit a fixed batch (next-token) — proves the model can learn."""
    torch.manual_seed(0)
    dev = _device()
    cfg = Config(vocab_size=64, n_embd=64, n_head=4, depth=3, n_scales=3,
                 tau0=8.0, rho=4.0, chunk_size=16)
    m = FractalLM(cfg).to(dev).train()
    B, T = 4, 48
    data = torch.randint(0, cfg.vocab_size, (B, T + 1), device=dev)
    idx, tgt = data[:, :-1], data[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for step in range(200):
        _, loss, _, _ = m(idx, targets=tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        if step % 50 == 0:
            print(f"[learns] step {step:3d}  loss {loss.item():.4f}")
        last = loss.item()
    print(f"[learns] loss {first:.3f} → {last:.3f}")
    return last


def test_stream_equals_full() -> None:
    assert _stream_equals_full() < 1e-3


def test_learns() -> None:
    assert _learns() < 0.5


def main() -> None:
    d_stream = _stream_equals_full()
    test_variable_depth()
    final_loss = _learns()

    ok = d_stream < 1e-3 and final_loss < 0.5
    print("\nOK — model is consistent, unrollable, and learns"
          if ok else f"\nFAIL — stream Δ={d_stream:.1e}, final_loss={final_loss:.3f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
