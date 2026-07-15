"""Persistence: saving/loading the model and, above all, the PERSISTENT W state.

The W state = fast weights that self-modify and are meant to survive across sessions/processes.
Snapshots enable a manual rollback when a collapse is visible in the outputs (no auto-rollback).
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

import torch

from .model import GPTConfig, RTAIModel


def atomic_torch_save(obj, path: str | os.PathLike[str]) -> None:
    """Atomically save project data with owner-only permissions."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.",
                                     suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        os.chmod(temporary, 0o600)
        torch.save(obj, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_load(path: str | os.PathLike[str], map_location):
    return torch.load(path, map_location=map_location, weights_only=True)


def save_model(path: str, model: RTAIModel):
    atomic_torch_save({"cfg": model.cfg.__dict__, "model": model.state_dict()}, path)


def load_model(path: str, device) -> RTAIModel:
    if Path(path).is_dir():
        from rtai.model_hub import load_bundle
        return load_bundle(path, device)
    ckpt = _safe_load(path, map_location=device)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("cfg"), dict) \
            or not isinstance(ckpt.get("model"), dict):
        raise ValueError(f"invalid RTAI checkpoint schema: {path}")
    cfg = GPTConfig(**ckpt["cfg"])
    model = RTAIModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def save_state(path: str, states: list[torch.Tensor], meta: dict | None = None):
    """Save the persistent W (one matrix per layer, batch=1)."""
    atomic_torch_save({"states": [s.detach().cpu() for s in states],
                       "meta": meta or {}, "t": time.time()}, path)


def load_state(path: str, device) -> list[torch.Tensor]:
    obj = _safe_load(path, map_location=device)
    if not isinstance(obj, dict) or not isinstance(obj.get("states"), list):
        raise ValueError(f"invalid RTAI runtime-state schema: {path}")
    return [s.to(device) for s in obj["states"]]


def snapshot_state(states: list[torch.Tensor], snap_dir: str, tag: str):
    """Save a snapshot of the state for a possible manual rollback."""
    os.makedirs(snap_dir, exist_ok=True)
    path = os.path.join(snap_dir, f"snap_{tag}.pt")
    save_state(path, states, meta={"tag": tag})
    return path
