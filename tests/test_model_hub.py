"""Offline model-bundle tests; no Hugging Face account or network is required."""

from pathlib import Path

import pytest
import torch

from fractal import persist as fractal_persist
from fractal.model import Config, FractalLM
from rtai import model_hub, state
from rtai.model import GPTConfig, RTAIModel


def test_safetensors_bundle_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    model = RTAIModel(GPTConfig(vocab_size=66, block_size=16, n_layer=1,
                                n_head=2, n_embd=16)).eval()
    checkpoint = tmp_path / "model.pt"
    bundle = tmp_path / "bundle"
    state.save_model(str(checkpoint), model)
    model_hub.export_bundle(str(checkpoint), str(bundle), "test-model", "apache-2.0")
    loaded = state.load_model(str(bundle), "cpu")
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
    assert not torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)


def test_export_rejects_runtime_memory(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    memory = tmp_path / "memory.pt"
    state.save_state(str(memory), [torch.zeros(1, 2, 3, 3)])
    with pytest.raises(ValueError, match="clean model checkpoints"):
        model_hub.export_bundle(str(memory), str(tmp_path / "bundle"), "bad", "apache-2.0")


def test_fractal_safetensors_bundle_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    model = FractalLM(Config(vocab_size=64, n_embd=16, n_head=2, depth=2,
                            n_scales=2, chunk_size=8)).eval()
    checkpoint = tmp_path / "fractal.pt"
    bundle = tmp_path / "fractal-bundle"
    fractal_persist.save_model(str(checkpoint), model)
    model_hub.export_bundle(str(checkpoint), str(bundle), "fractal-test", "apache-2.0")
    loaded = fractal_persist.load_model(str(bundle), "cpu")
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
