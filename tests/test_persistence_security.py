"""Security and durability checks for model and runtime-state persistence."""

from pathlib import Path

import torch

from fractal import persist as fractal_persist
from fractal.model import Config, FractalLM
from rtai import state as rtai_state
from rtai.model import GPTConfig, RTAIModel


def test_rtai_model_and_state_round_trip(tmp_path: Path) -> None:
    model = RTAIModel(GPTConfig(vocab_size=66, block_size=16, n_layer=1,
                                n_head=2, n_embd=16))
    checkpoint = tmp_path / "model.pt"
    state_path = tmp_path / "state.pt"
    rtai_state.save_model(str(checkpoint), model)
    states = model.init_states(1, "cpu")
    rtai_state.save_state(str(state_path), states)
    loaded_model = rtai_state.load_model(str(checkpoint), "cpu")
    loaded_states = rtai_state.load_state(str(state_path), "cpu")
    assert sum(p.numel() for p in loaded_model.parameters()) == sum(p.numel() for p in model.parameters())
    assert all(torch.equal(left, right) for left, right in zip(states, loaded_states))
    assert checkpoint.stat().st_mode & 0o077 == 0
    assert state_path.stat().st_mode & 0o077 == 0


def test_fractal_model_and_state_round_trip(tmp_path: Path) -> None:
    model = FractalLM(Config(vocab_size=64, n_embd=16, n_head=2, depth=2,
                            n_scales=2, chunk_size=8))
    checkpoint = tmp_path / "fractal.pt"
    state_path = tmp_path / "fractal-state.pt"
    fractal_persist.save_model(str(checkpoint), model)
    states = model.init_states(1, "cpu")
    fractal_persist.save_states(str(state_path), states)
    loaded_model = fractal_persist.load_model(str(checkpoint), "cpu")
    loaded_states = fractal_persist.load_states(str(state_path), "cpu")
    assert sum(p.numel() for p in loaded_model.parameters()) == sum(p.numel() for p in model.parameters())
    assert len(loaded_states) == len(states)


def test_runtime_state_is_not_a_model_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "memory.pt"
    rtai_state.save_state(str(state_path), [torch.zeros(1, 2, 3, 3)])
    try:
        rtai_state.load_model(str(state_path), "cpu")
    except ValueError as exc:
        assert "checkpoint schema" in str(exc)
    else:
        raise AssertionError("runtime state was accepted as a model checkpoint")
