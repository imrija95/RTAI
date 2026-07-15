"""Lightweight monitoring of the integrity of the persistent W weights (NO gating).

Observation only — no auto eval-gate, no auto-rollback. The goal is to make it possible
to tell from the metrics when the model starts to break, so one can intervene manually
(snapshot rollback).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def state_integrity(states: list[torch.Tensor]) -> dict:
    """Health metrics of the persistent weights: norm, max singular value, NaN/Inf."""
    out = {}
    for i, W in enumerate(states):
        Wf = W.float()
        finite = torch.isfinite(Wf).all().item()
        # spectral norm (largest singular value) — a blow-up indicator.
        # averaged over heads, only as a rough guide.
        try:
            sv = torch.linalg.matrix_norm(Wf, ord=2)  # (..., ) per matrix
            spec = float(sv.mean().item())
        except Exception:
            spec = float("nan")
        out[f"L{i}"] = {
            "fro": float(Wf.norm().item()),
            "spec": spec,
            "finite": bool(finite),
        }
    return out


def format_integrity(info: dict) -> str:
    parts = []
    for lname, m in info.items():
        flag = "" if m["finite"] else " ⚠NAN"
        parts.append(f"{lname}: fro={m['fro']:.2f} spec={m['spec']:.2f}{flag}")
    return " | ".join(parts)


@torch.no_grad()
def canary_recall_acc(model, vocab, device, n_pairs=8, n_queries=8, trials=64,
                      generator=None) -> float:
    """Canary probe: in-context recall ability (start from W0, no persistence).

    Only LOGS the health of the core ability — does not intervene. If it declines
    over time, persistent self-modification is probably degrading the base
    (a signal for manual intervention).
    """
    from .data_recall import make_recall_batch

    model.eval()
    x, y = make_recall_batch(trials, n_pairs, n_queries, vocab, device, generator)
    logits, _, _, _ = model(x, states=None)
    mask = y != -1
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == y[mask]).float().mean().item()
    return correct
