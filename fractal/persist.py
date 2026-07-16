"""Persistence of mutable state to disk.

Separate from the model checkpoint: the learned (slow) weights live in the ckpt, whereas the
fast weights `W` (memory that modifies itself at runtime) live here and accumulate across sessions.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch

from fractal.model import Config, FractalLM
from fractal.unit import FractalState


def atomic_torch_save(obj, path: str | os.PathLike[str]) -> None:
    """Write a PyTorch object atomically with owner-only permissions.

    Runtime state can encode information observed during operation. Keeping the
    temporary file beside the destination makes the final ``os.replace`` atomic
    on ordinary local filesystems.
    """
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
    """Load tensor-only project data without permitting arbitrary pickle code."""
    return torch.load(path, map_location=map_location, weights_only=True)


def save_model(path: str, model: FractalLM) -> None:
    # Persist the per-scale fast-weight gain too: it is a non-persistent buffer (so old checkpoints load
    # unchanged), but usage-driven plasticity (fractal/plasticity.py) leaves it != 1 and inference must
    # match training. Stored separately as [block][scale] floats — absent in pre-plasticity checkpoints.
    beta_gain = [[float(c._beta_gain_f) for c in b.unit.cells] for b in model.blocks]
    growing_cortex = (
        None if model.skill_cortex is None
        else {"schema_version": 1, "experts": model.skill_cortex.manifest()}
    )
    atomic_torch_save({"cfg": model.cfg.__dict__, "model": model.state_dict(),
                       "beta_gain": beta_gain, "growing_cortex": growing_cortex}, path)


def load_model(path: str, device) -> FractalLM:
    if Path(path).is_dir():
        from rtai.model_hub import load_bundle
        return load_bundle(path, device)
    ckpt = _safe_load(path, map_location=device)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("cfg"), dict) \
            or not isinstance(ckpt.get("model"), dict):
        raise ValueError(f"invalid FractalLM checkpoint schema: {path}")
    cfg_data = dict(ckpt["cfg"])
    # Growing Cortex checkpoints created before local teaching was introduced contain the archived
    # full hypernetwork compiler. Preserve their schema while new production configs default to none.
    if cfg_data.get("growing_cortex") and "skill_compiler" not in cfg_data:
        cfg_data["skill_compiler"] = "full"
        cfg_data["skill_address_dim"] = cfg_data.get("n_embd", 64)
        cfg_data.setdefault("skill_auto_route", True)
    model = FractalLM(Config(**cfg_data)).to(device)
    growing_cortex = ckpt.get("growing_cortex")
    if growing_cortex is not None:
        if model.skill_cortex is None or growing_cortex.get("schema_version") != 1:
            raise ValueError("invalid growing cortex checkpoint schema")
        model.skill_cortex.restore_structure(growing_cortex.get("experts") or [])
        model.skill_cortex.to(device)
    state = dict(ckpt["model"])
    # Early weight-tied checkpoints predate the ModuleList used by the untied option and contain
    # only the compatibility alias `block.*`. Mirror it into `blocks.0.*` without altering data.
    if not any(k.startswith("blocks.") for k in state):
        state.update({"blocks.0." + k[len("block."):]: v
                      for k, v in state.items() if k.startswith("block.")})
    model.load_state_dict(state)
    bg = ckpt.get("beta_gain")                    # None for pre-plasticity checkpoints → stays 1.0
    if bg is not None:
        for b, gains in zip(model.blocks, bg):
            for c, g in zip(b.unit.cells, gains):
                c.set_beta_gain(g)
    return model


def save_states(path: str, states: list[FractalState]) -> None:
    """Save a list of per-layer states as plain tensor dicts (robust to torch.load)."""
    blob = [{"W": [w.cpu() for w in s.W],
             "eligibility": (None if s.eligibility is None
                             else [e.cpu() for e in s.eligibility]),
             "conv": None if s.conv is None else s.conv.cpu(),
             "hp_sum": None if s.hp_sum is None else s.hp_sum.cpu(),
             "hp_n": s.hp_n,
             "event_prev": None if s.event_prev is None else s.event_prev.cpu(),
             "event_n": s.event_n,
             "event_sum": None if s.event_sum is None else s.event_sum.cpu(),
             "event_count": s.event_count} for s in states]
    atomic_torch_save(blob, path)


def load_states(path: str, device) -> list[FractalState]:
    blob = _safe_load(path, map_location="cpu")
    if not isinstance(blob, list) or not all(isinstance(item, dict) for item in blob):
        raise ValueError(f"invalid FractalLM runtime-state schema: {path}")
    return [FractalState(W=[w.to(device) for w in d["W"]],
                         eligibility=(None if d.get("eligibility") is None else
                                      [e.to(device) for e in d["eligibility"]]),
                         conv=None if d["conv"] is None else d["conv"].to(device),
                         hp_sum=None if d.get("hp_sum") is None else d["hp_sum"].to(device),
                         hp_n=d.get("hp_n", 0),
                         event_prev=None if d.get("event_prev") is None else d["event_prev"].to(device),
                         event_n=d.get("event_n", 0),
                         event_sum=None if d.get("event_sum") is None else d["event_sum"].to(device),
                         event_count=d.get("event_count", 0))
            for d in blob]
