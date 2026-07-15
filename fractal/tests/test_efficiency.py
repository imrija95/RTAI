"""CPU checks for every mechanism used by the efficiency tournament."""

from __future__ import annotations

import os
import tempfile

import torch

from fractal import persist
from fractal.efficiency import VerifiedToolGen, local_credit_loss
from fractal.model import Config, FractalLM


def _cfg(**kw):
    base = dict(vocab_size=64, n_embd=32, n_head=4, depth=2, n_scales=2, chunk_size=16)
    base.update(kw)
    return Config(**base)


def test_top1_mitosis_and_sparse_gradient():
    torch.manual_seed(2)
    model = FractalLM(_cfg(n_experts=4, moe_mode="top1"))
    moe = model.block.mlp
    x = torch.randn(2, 7, 32)
    assert torch.allclose(moe(x), moe.experts[0](x), atol=1e-7, rtol=0), \
        "cloned top-1 MoE changed the initial function"
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))
    model.zero_grad(set_to_none=True)
    moe(x).square().mean().backward()
    grad = [sum(float(p.grad.abs().sum()) for p in e.parameters() if p.grad is not None)
            for e in moe.experts]
    assert grad[2] > 0 and sum(grad[:2] + grad[3:]) == 0
    assert float(moe.router.weight.grad.abs().sum()) > 0, "straight-through router got no task gradient"
    stored, active = model.parameter_counts()
    assert active < stored


def test_event_budget_is_causal_and_streaming_exact():
    torch.manual_seed(3)
    model = FractalLM(_cfg(event_budget=0.25)).eval()
    x = torch.randint(0, 64, (1, 20))
    x_changed = x.clone()
    x_changed[:, 12:] = torch.randint(0, 64, x_changed[:, 12:].shape)
    with torch.no_grad():
        initial = model.init_states(1, "cpu")
        full = model(x, states=[s.clone() for s in initial])[0]
        changed = model(x_changed, states=[s.clone() for s in initial])[0]
        assert torch.equal(full[:, :12], changed[:, :12]), "event path leaked suffix information"
        state, pieces = [s.clone() for s in initial], []
        for t in range(x.shape[1]):
            out, state = model.forward_stream(x[:, t:t + 1], state)
            pieces.append(out)
        stream = torch.cat(pieces, dim=1)
    assert (full - stream).abs().max().item() < 1e-5
    model.train()
    model(x)
    assert abs(model.event_share() - 0.25) < 1e-9


def test_local_credit_cuts_the_earlier_graph():
    torch.manual_seed(4)
    model = FractalLM(_cfg(depth=4)).train()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    w = torch.ones_like(x, dtype=torch.float32)
    flags = []
    hook = model.block.register_forward_hook(lambda _m, _i, out: flags.append(out[0].requires_grad))
    loss = local_credit_loss(model, x, y, w, selected_depth=2)
    hook.remove()
    loss.backward()
    assert flags == [False, False, True], flags
    assert any(p.grad is not None for p in model.block.parameters())


def test_event_state_persists():
    model = FractalLM(_cfg(event_budget=0.25)).eval()
    states = model.init_states(1, "cpu")
    with torch.no_grad():
        _, states = model.forward_stream(torch.randint(0, 64, (1, 3)), states)
    assert states[0].event_count == 3 and states[0].event_sum is not None
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.pt")
        persist.save_states(path, states)
        loaded = persist.load_states(path, "cpu")
    assert loaded[0].event_count == 3
    assert torch.equal(loaded[0].event_sum, states[0].event_sum)


def test_event_budget_accepts_mixed_precision_outputs():
    """Event scatter must return to the residual dtype under autocast."""
    torch.manual_seed(6)
    model = FractalLM(_cfg(event_budget=0.25)).train()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, loss, _, _ = model(x, targets=y)
    loss.backward()
    assert torch.isfinite(loss)


def test_verified_tools_use_weighted_production_format(tmp_path):
    from fractal import tokenizer as tokenizer_mod
    tokenizer_path = tmp_path / "tokenizer.json"
    tok = tokenizer_mod.train_bpe(
        ["calculate 12 + 7", "echo amber river stone", "what time is it"],
        vocab_size=300,
        save_path=str(tokenizer_path),
    )
    gen = VerifiedToolGen(tok, seed=5)
    _, name, args, pieces = gen.episode()
    assert name in ("calc", "echo", "time") and isinstance(args, dict)
    assert any(weight == 4.0 for _, weight in pieces)
    x, y, w = gen.batch(2, 96, "cpu")
    assert x.shape == y.shape == w.shape == (2, 96)
    assert float(w.max()) == 4.0


def main():
    test_top1_mitosis_and_sparse_gradient()
    test_event_budget_is_causal_and_streaming_exact()
    test_local_credit_cuts_the_earlier_graph()
    test_event_state_persists()
    test_event_budget_accepts_mixed_precision_outputs()
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        test_verified_tools_use_weighted_production_format(Path(td))
    print("OK - efficiency mechanisms are sparse, causal, persistent, and trainable")


if __name__ == "__main__":
    main()
