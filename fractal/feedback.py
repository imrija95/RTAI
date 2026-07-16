"""Delayed feedback and bounded consolidation for Predictive Event Algebra.

Every token still performs the ordinary delta-rule write. Event Algebra additionally carries an
O(1) eligibility matrix per fast-weight matrix. Later evidence can reinforce or oppose that exact
kind of write without retaining a growing tensor history. Explicit ratings use the centered scale
1..5 -> -1..1; rating 3 is deliberately neutral.

Runtime feedback first changes the session fast state. Strong feedback can also be consolidated
into ``W0``, the learned initial fast weight, so the association is present in a fresh session.
All changes are norm bounded. Checkpointing and higher-level acceptance policy remain the caller's
responsibility because only the caller knows the durable checkpoint and anchor evaluation paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


RATING_CREDIT = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}


@dataclass(frozen=True)
class FeedbackResult:
    rating: int
    credit: float
    fast_update_norm: float
    w0_update_norm: float
    consolidated: bool


@dataclass(frozen=True)
class W0TeachingResult:
    """Measurements from a bounded supervised update of persistent associative weights."""

    update_norm: float
    initial_loss: float
    final_loss: float
    anchor_kl: float
    preserve_kl: float


def credit_for_rating(rating: int) -> float:
    """Map the shared UI scale onto signed delayed credit."""
    try:
        return RATING_CREDIT[int(rating)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rating must be an integer from 1 to 5") from exc


def enable(model) -> None:
    """Enable eligibility traces on a loaded compatible model without changing its slow weights."""
    model.cfg.event_algebra = True
    for block in model.blocks:
        block.unit.event_algebra = True


def _bounded(delta: torch.Tensor, reference: torch.Tensor, fraction: float) -> torch.Tensor:
    if not torch.isfinite(delta).all():
        raise ValueError("feedback produced a non-finite update")
    norm = delta.norm()
    limit = max(float(reference.norm()), 1.0) * float(fraction)
    if float(norm) > limit:
        delta = delta * (limit / float(norm))
    return delta


@torch.no_grad()
def message_eligibility(model, token_ids: list[int], device) -> list:
    """Encode one rated message into isolated, content-targeted eligibility traces.

    The live conversation state is not used or changed. Re-encoding happens only when feedback is
    submitted, so it stays off the generation and training hot paths.
    """
    if not token_ids:
        raise ValueError("cannot rate an empty message")
    enable(model)
    was_training = model.training
    model.eval()
    try:
        states = model.init_states(1, device)
        idx = torch.tensor([token_ids], device=device, dtype=torch.long)
        _, states = model.forward_stream(idx, states)
    finally:
        model.train(was_training)
    return states


@torch.no_grad()
def apply_to_state(states, evidence_states, credit: float, lr: float = 0.25,
                   max_fraction: float = 0.10) -> float:
    """Apply signed evidence to the current session state and return the total update norm."""
    if len(states) != len(evidence_states):
        raise ValueError("state/evidence depth mismatch")
    total_sq = 0.0
    for state, evidence in zip(states, evidence_states):
        if evidence.eligibility is None:
            raise ValueError("evidence state has no eligibility trace")
        if len(state.W) != len(evidence.eligibility):
            raise ValueError("state/evidence scale mismatch")
        for scale, trace in enumerate(evidence.eligibility):
            delta = _bounded(float(lr) * float(credit) * trace, state.W[scale], max_fraction)
            state.W[scale].add_(delta)
            total_sq += float(delta.float().square().sum())
    return total_sq ** 0.5


def observed_surprise(logits: torch.Tensor, token_ids: list[int]) -> float:
    """Mean next-token surprise credit available inside one observed multi-token span."""
    if len(token_ids) < 2:
        return 0.0
    targets = torch.tensor(token_ids[1:], device=logits.device, dtype=torch.long)
    probabilities = logits[0, :-1].softmax(dim=-1).gather(1, targets[:, None]).mean()
    return float((1.0 - probabilities).clamp(0.0, 1.0).detach())


def recent_evidence(before_states, after_states, tokens: int, decay: float):
    """Isolate eligibility written by the latest span from the carried delayed trace."""
    evidence = [state.clone() for state in after_states]
    factor = float(decay) ** int(tokens)
    for before, after, target in zip(before_states, after_states, evidence):
        if after.eligibility is None:
            target.eligibility = None
            continue
        previous = before.eligibility
        target.eligibility = [
            current.clone() if previous is None else current - factor * old
            for current, old in zip(after.eligibility, previous or [])
        ]
    return evidence


def _depths_by_unit(model):
    grouped = {}
    for depth in range(model.cfg.depth):
        unit = model.block_at(depth).unit
        grouped.setdefault(id(unit), (unit, []))[1].append(depth)
    return list(grouped.values())


def _w0_parameters(model, scope: str = "all") -> list[torch.nn.Parameter]:
    if scope not in ("all", "permanent"):
        raise ValueError("W0 teaching scope must be 'all' or 'permanent'")
    parameters = []
    for unit, _depths in _depths_by_unit(model):
        for cell in unit.cells:
            if scope == "all" or float(cell.gamma) >= 1.0 - 1e-9:
                parameters.append(cell.W0)
    return parameters


def _prompt_batches(prompts: list[list[int]], device) -> list[torch.Tensor]:
    grouped: dict[int, list[list[int]]] = {}
    for prompt in prompts:
        if prompt:
            grouped.setdefault(len(prompt), []).append(prompt)
    return [
        torch.tensor(items, dtype=torch.long, device=device)
        for _length, items in sorted(grouped.items())
    ]


def teach_w0(model, prompt_ids: list[int], target_id: int, device, *, lr: float = 0.01,
             steps: int = 1, max_fraction: float = 0.01,
             anchor_prompts: list[list[int]] | None = None,
             anchor_weight: float = 0.0, anchor_mode: str = "kl",
             scope: str = "all", preserve_prompts: list[list[int]] | None = None,
             preserve_weight: float = 0.0,
             anchor_forbidden_ids: list[int] | None = None) -> W0TeachingResult:
    """Teach one confirmed next token by gradient-updating only persistent ``W0`` matrices.

    A scalar rating can authorize and scale this operation, but the next-token prediction error
    supplies the missing vector direction. General model parameters remain frozen. Callers are
    responsible for policy, snapshots, rollback, and anchor evaluation before durable acceptance.
    """
    if not prompt_ids:
        raise ValueError("cannot teach from an empty prompt")
    if steps < 1:
        raise ValueError("steps must be positive")
    if lr < 0.0 or max_fraction < 0.0 or anchor_weight < 0.0 or preserve_weight < 0.0:
        raise ValueError(
            "learning rate, max fraction, anchor weight, and preserve weight must be non-negative")
    if anchor_mode not in ("kl", "negative", "margin"):
        raise ValueError("anchor mode must be 'kl', 'negative', or 'margin'")
    persistent = _w0_parameters(model, scope)
    persistent_ids = {id(parameter) for parameter in persistent}
    trainable = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    was_training = model.training
    event_units = [unit for unit, _depths in _depths_by_unit(model)]
    event_settings = [(unit, unit.event_algebra) for unit in event_units]
    cfg_event_algebra = model.cfg.event_algebra
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    target = torch.tensor([int(target_id)], dtype=torch.long, device=device)
    forbidden_ids = sorted(
        {int(target_id)} | {int(value) for value in (anchor_forbidden_ids or [])})
    total_sq = 0.0
    initial_loss = final_loss = float("nan")
    final_anchor_kl = 0.0
    final_preserve_kl = 0.0
    try:
        model.eval()
        for parameter, _required in trainable:
            parameter.requires_grad_(id(parameter) in persistent_ids)
        # Eligibility is not consumed by supervised W0 teaching. The ordinary delta write is
        # numerically equivalent without it, while the chunk kernel is substantially faster.
        model.cfg.event_algebra = False
        for unit in event_units:
            unit.event_algebra = False
        anchors = _prompt_batches(anchor_prompts or [], device)
        preserve = _prompt_batches(preserve_prompts or [], device)
        with torch.no_grad():
            anchor_targets = [model(anchor)[0][:, -1].float().softmax(dim=-1) for anchor in anchors]
            preserve_targets = [
                model(prompt)[0][:, -1].float().softmax(dim=-1) for prompt in preserve
            ]
        for step in range(int(steps)):
            with torch.enable_grad():
                logits, _loss, _states, _delta = model(idx)
                loss = F.cross_entropy(logits[:, -1], target)
                if anchors and anchor_weight:
                    anchor_losses = []
                    for anchor, reference in zip(anchors, anchor_targets):
                        anchor_logits = model(anchor)[0][:, -1].float()
                        if anchor_mode == "margin":
                            forbidden = anchor_logits[:, forbidden_ids]
                            alternatives = anchor_logits.clone()
                            alternatives[:, forbidden_ids] = -torch.inf
                            best_other = alternatives.max(dim=-1).values
                            anchor_losses.append(
                                F.relu(forbidden - best_other[:, None] + 0.1).mean(dim=-1))
                        elif anchor_mode == "negative":
                            probability = anchor_logits.softmax(dim=-1)[:, forbidden_ids]
                            anchor_losses.append(
                                -torch.log1p(-probability.clamp(max=1 - 1e-6)).mean(dim=-1))
                        else:
                            log_probability = anchor_logits.log_softmax(dim=-1)
                            anchor_losses.append(
                                (reference * (reference.clamp_min(1e-12).log() - log_probability)).sum(-1))
                    loss = loss + float(anchor_weight) * torch.cat(anchor_losses).mean()
                if preserve and preserve_weight:
                    preserve_losses = []
                    for prompt, reference in zip(preserve, preserve_targets):
                        prompt_logits = model(prompt)[0][:, -1].float()
                        log_probability = prompt_logits.log_softmax(dim=-1)
                        preserve_losses.append(
                            (reference * (
                                reference.clamp_min(1e-12).log() - log_probability
                            )).sum(-1))
                    loss = loss + float(preserve_weight) * torch.cat(preserve_losses).mean()
                gradients = torch.autograd.grad(loss, persistent)
            if step == 0:
                initial_loss = float(loss.detach())
            with torch.no_grad():
                for parameter, gradient in zip(persistent, gradients):
                    delta = _bounded(-float(lr) * gradient, parameter, max_fraction)
                    parameter.add_(delta)
                    total_sq += float(delta.float().square().sum())
        with torch.no_grad():
            logits, _loss, _states, _delta = model(idx)
            final_loss = float(F.cross_entropy(logits[:, -1], target))
            if anchors:
                anchor_losses = []
                for anchor, reference in zip(anchors, anchor_targets):
                    anchor_logits = model(anchor)[0][:, -1].float()
                    log_probability = anchor_logits.log_softmax(dim=-1)
                    anchor_losses.append(
                        (reference * (reference.clamp_min(1e-12).log() - log_probability)).sum(-1))
                final_anchor_kl = float(torch.cat(anchor_losses).mean())
            if preserve:
                preserve_losses = []
                for prompt, reference in zip(preserve, preserve_targets):
                    prompt_logits = model(prompt)[0][:, -1].float()
                    log_probability = prompt_logits.log_softmax(dim=-1)
                    preserve_losses.append(
                        (reference * (
                            reference.clamp_min(1e-12).log() - log_probability
                        )).sum(-1))
                final_preserve_kl = float(torch.cat(preserve_losses).mean())
    finally:
        model.cfg.event_algebra = cfg_event_algebra
        for unit, enabled in event_settings:
            unit.event_algebra = enabled
        for parameter, required in trainable:
            parameter.requires_grad_(required)
        model.train(was_training)
    return W0TeachingResult(
        update_norm=total_sq ** 0.5,
        initial_loss=initial_loss,
        final_loss=final_loss,
        anchor_kl=final_anchor_kl,
        preserve_kl=final_preserve_kl,
    )


def w0_snapshot(model) -> list[list[torch.Tensor]]:
    """Return a CPU snapshot of each stored block's long-term associative weights."""
    return [[cell.W0.detach().cpu().clone() for cell in block.unit.cells]
            for block in model.blocks]


@torch.no_grad()
def restore_w0(model, snapshot) -> None:
    if len(snapshot) != len(model.blocks):
        raise ValueError("W0 overlay block count does not match the model")
    for block, scales in zip(model.blocks, snapshot):
        if len(scales) != len(block.unit.cells):
            raise ValueError("W0 overlay scale count does not match the model")
        for cell, value in zip(block.unit.cells, scales):
            if tuple(value.shape) != tuple(cell.W0.shape):
                raise ValueError("W0 overlay tensor shape does not match the model")
            cell.W0.copy_(value.to(cell.W0))


def save_w0(path, model) -> None:
    """Atomically persist only the consolidated W0 overlay, not a duplicate full checkpoint."""
    from fractal.persist import atomic_torch_save
    atomic_torch_save({"schema_version": 1, "w0": w0_snapshot(model)}, path)


def load_w0(path, model) -> bool:
    """Load a validated tensor-only W0 overlay. Return False when no overlay exists."""
    if not Path(path).exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid W0 overlay schema")
    restore_w0(model, payload.get("w0"))
    return True


def append_event(path, event: dict) -> None:
    """Append one private, fsynced feedback event for a concurrently running trainer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        line = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_events(path) -> list[dict]:
    """Read complete JSONL records; a concurrently incomplete final record is ignored."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event_id"), str):
            events.append(value)
    return events


def save_consolidation_state(path, model, seen_event_ids) -> None:
    """Atomically couple W0 and consumed IDs so queue replay is exactly-once after restart."""
    from fractal.persist import atomic_torch_save
    atomic_torch_save({"schema_version": 1, "w0": w0_snapshot(model),
                       "seen_event_ids": sorted(set(seen_event_ids))}, path)


def load_consolidation_state(path, model) -> set[str]:
    if not Path(path).exists():
        return set()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid feedback consolidation state")
    restore_w0(model, payload.get("w0"))
    return set(payload.get("seen_event_ids") or [])


@torch.no_grad()
def consolidate_w0(model, evidence_states, credit: float, lr: float = 0.01,
                   max_fraction: float = 0.01) -> float:
    """Commit strong evidence into W0, averaging tied-depth traces exactly once per unit."""
    total_sq = 0.0
    for unit, depths in _depths_by_unit(model):
        for scale, cell in enumerate(unit.cells):
            traces = []
            for depth in depths:
                eligibility = evidence_states[depth].eligibility
                if eligibility is not None:
                    traces.append(eligibility[scale].mean(dim=0))
            if not traces:
                continue
            trace = torch.stack(traces).mean(dim=0).to(cell.W0)
            delta = _bounded(float(lr) * float(credit) * trace, cell.W0, max_fraction)
            cell.W0.add_(delta)
            total_sq += float(delta.float().square().sum())
    return total_sq ** 0.5


@torch.no_grad()
def apply_rating(model, states, evidence_states, rating: int, *, fast_lr: float = 0.25,
                 w0_lr: float = 0.01, consolidate: bool = True) -> FeedbackResult:
    """Apply one idempotence-checked rating; callers must reject duplicate revisions first."""
    credit = credit_for_rating(rating)
    fast_norm = apply_to_state(states, evidence_states, credit, lr=fast_lr) if credit else 0.0
    do_consolidate = bool(consolidate and rating != 3)
    w0_norm = consolidate_w0(model, evidence_states, credit, lr=w0_lr) if do_consolidate else 0.0
    return FeedbackResult(rating=int(rating), credit=credit, fast_update_norm=fast_norm,
                          w0_update_norm=w0_norm, consolidated=do_consolidate)
