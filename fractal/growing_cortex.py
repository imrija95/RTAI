"""Append-only, function-preserving skill growth for the fractal model.

The growing cortex is one shared residual organ reused at every recurrent depth.  It owns a
content-addressed bank of low-rank skill experts and a compiler that maps a short task
specification to a candidate expert.  Candidates are born disabled, so appending one does not
change the model function.  Only an explicit commit makes an expert routable.

The module deliberately separates:

* specification compilation -- a meta-trained mapping from task features to an expert,
* read routing -- selecting a mature expert for a complete sample or sticky trajectory,
* write routing -- updating only one candidate expert,
* lifecycle policy -- deciding when evidence is strong enough to birth or commit an expert.

The number of stored experts may grow, while active compute remains the shared base plus at most
one low-rank residual expert.  Runtime fast-weight memory remains independent of conversation
length.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVE_STATUSES = frozenset({"juvenile", "mature"})
KNOWN_STATUSES = frozenset({"candidate", "juvenile", "mature", "quarantined", "dormant", "pruned"})


@dataclass
class CompiledSkill:
    """Differentiable candidate generated from a task representation."""

    key: torch.Tensor
    down: torch.Tensor
    up: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the generated low-rank residual to ``x``."""
        if self.down.ndim == 2:
            return F.linear(F.gelu(F.linear(x, self.down)), self.up)
        if self.down.ndim != 3 or x.shape[0] != self.down.shape[0]:
            raise ValueError("batched compiled skills must match the input batch")
        hidden = torch.einsum("bte,bre->btr", x, self.down)
        return torch.einsum("btr,ber->bte", F.gelu(hidden), self.up)

    def repeat_interleave(self, repeats: int) -> "CompiledSkill":
        if self.down.ndim == 2:
            if repeats == 1:
                return self
            return CompiledSkill(
                key=self.key[None].expand(repeats, -1),
                down=self.down[None].expand(repeats, -1, -1),
                up=self.up[None].expand(repeats, -1, -1),
            )
        return CompiledSkill(
            key=self.key.repeat_interleave(repeats, dim=0),
            down=self.down.repeat_interleave(repeats, dim=0),
            up=self.up.repeat_interleave(repeats, dim=0),
        )


@dataclass(frozen=True)
class LocalTeachingResult:
    """Measurements from bounded local refinement of one expert."""

    initial_loss: float
    final_loss: float
    anchor_kl: float
    update_norm: float
    steps: int


@dataclass(frozen=True)
class GrowthEvidence:
    """Evidence used by the append-only birth policy."""

    fingerprint: str
    error: float
    nearest_similarity: float
    gradient_cosine: float
    existing_expert_improved: bool


class SkillCompiler(nn.Module):
    """Map a short specification representation to a low-rank residual program."""

    def __init__(self, n_embd: int, rank: int, key_dim: int | None = None):
        super().__init__()
        self.n_embd = int(n_embd)
        self.rank = int(rank)
        self.key_dim = int(key_dim or n_embd)
        self.norm = nn.LayerNorm(n_embd)
        self.sequence_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.key_proj = nn.Linear(n_embd, self.key_dim, bias=False)
        self.down_proj = nn.Linear(n_embd, rank * n_embd, bias=True)
        self.up_proj = nn.Linear(n_embd, n_embd * rank, bias=True)

    def forward(self, task_features: torch.Tensor) -> CompiledSkill:
        if task_features.ndim == 3:
            length = task_features.shape[1]
            weights = torch.linspace(
                -1.0, 1.0, length, device=task_features.device,
                dtype=task_features.dtype,
            ).view(1, length, 1)
            mean = task_features.mean(dim=1)
            ordered = (task_features * weights).mean(dim=1)
            task_features = self.sequence_proj(torch.cat((mean, ordered), dim=-1))
        if task_features.ndim != 2:
            raise ValueError("skill compilation requires a batch of task specifications")
        z = self.norm(task_features)
        key = F.normalize(self.key_proj(z), dim=-1)
        # Candidate execution is always shadowed until verification, so the compiler can use a
        # materially expressive residual without perturbing the active model at birth.
        scale = 1.0 / math.sqrt(max(self.rank, 1))
        down = torch.tanh(self.down_proj(z)).view(-1, self.rank, self.n_embd) * scale
        up = torch.tanh(self.up_proj(z)).view(-1, self.n_embd, self.rank) * scale
        if z.shape[0] == 1:
            key, down, up = key[0], down[0], up[0]
        return CompiledSkill(key=key, down=down, up=up)


class FactorizedAddressEncoder(nn.Module):
    """Small address encoder used by the production skill bank."""

    def __init__(self, n_embd: int, address_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.down = nn.Linear(n_embd, address_dim, bias=False)
        self.up = nn.Linear(address_dim, address_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            features = features.mean(dim=1)
        if features.ndim != 2:
            raise ValueError("skill address features must have shape (batch, time, embedding)")
        return self.up(F.gelu(self.down(self.norm(features))))


class SkillExpert(nn.Module):
    """One persistent low-rank hemisphere with a stable content address."""

    def __init__(self, n_embd: int, rank: int, address_dim: int, *, expert_id: int,
                 status: str = "candidate", parent_id: int | None = None, created_step: int = 0,
                 name: str = "", synopsis: str = ""):
        super().__init__()
        if status not in KNOWN_STATUSES:
            raise ValueError(f"unknown skill expert status: {status}")
        self.down = nn.Parameter(torch.empty(rank, n_embd))
        self.up = nn.Parameter(torch.zeros(n_embd, rank))
        nn.init.normal_(self.down, mean=0.0, std=1.0 / math.sqrt(n_embd))
        self.register_buffer("key", torch.zeros(address_dim))
        self.expert_id = int(expert_id)
        self.status = status
        self.parent_id = None if parent_id is None else int(parent_id)
        self.created_step = int(created_step)
        self.name = str(name)
        self.synopsis = str(synopsis)
        self.confidence = 0.0
        self.usage = 0
        self.last_score = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.gelu(F.linear(x, self.down)), self.up)

    @torch.no_grad()
    def load_compiled(self, compiled: CompiledSkill) -> None:
        self.key.copy_(compiled.key.detach().to(self.key))
        self.down.copy_(compiled.down.detach().to(self.down))
        self.up.copy_(compiled.up.detach().to(self.up))

    @torch.no_grad()
    def clone_from(self, parent: "SkillExpert") -> None:
        self.key.copy_(parent.key)
        self.down.copy_(parent.down)
        self.up.copy_(parent.up)

    def metadata(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "status": self.status,
            "parent_id": self.parent_id,
            "created_step": self.created_step,
            "name": self.name,
            "synopsis": self.synopsis,
            "confidence": float(self.confidence),
            "usage": int(self.usage),
            "last_score": float(self.last_score),
        }


class GrowingCortex(nn.Module):
    """Shared append-only skill bank with content-addressed top-1 routing."""

    def __init__(self, n_embd: int, rank: int = 8, router_threshold: float = 0.25,
                 auto_route: bool = False, compiler_mode: str = "none",
                 address_dim: int = 64):
        super().__init__()
        self.n_embd = int(n_embd)
        self.rank = int(rank)
        if compiler_mode not in ("none", "full"):
            raise ValueError(f"unknown skill compiler mode: {compiler_mode}")
        self.compiler_mode = compiler_mode
        self.address_dim = self.n_embd if compiler_mode == "full" else int(address_dim)
        self.router_threshold = float(router_threshold)
        self.auto_route = bool(auto_route)
        self.compiler = (
            SkillCompiler(n_embd, rank, self.address_dim) if compiler_mode == "full" else None)
        self.query_proj = (
            nn.Linear(n_embd, self.address_dim, bias=False)
            if compiler_mode == "full"
            else FactorizedAddressEncoder(n_embd, self.address_dim)
        )
        self.experts = nn.ModuleList()
        self._compiled_override: CompiledSkill | None = None
        self._forced_expert: int | None = None
        self._last_selected: list[int | None] = []
        self._last_scores: list[float] = []

    def compile(self, task_features: torch.Tensor) -> CompiledSkill:
        if self.compiler is None:
            raise ValueError("skill compiler is disabled; create and teach a local candidate")
        return self.compiler(task_features)

    def address(self, task_features: torch.Tensor) -> torch.Tensor:
        if task_features.ndim == 3:
            task_features = task_features.mean(dim=1)
        return F.normalize(self.query_proj(task_features), dim=-1)

    def _append_empty(self, metadata: dict | None = None) -> SkillExpert:
        metadata = metadata or {}
        expert_id = len(self.experts)
        if int(metadata.get("expert_id", expert_id)) != expert_id:
            raise ValueError("skill expert manifest must be append ordered")
        expert = SkillExpert(
            self.n_embd, self.rank, self.address_dim, expert_id=expert_id,
            status=str(metadata.get("status", "candidate")),
            parent_id=metadata.get("parent_id"),
            created_step=int(metadata.get("created_step", 0)),
            name=str(metadata.get("name", "")),
            synopsis=str(metadata.get("synopsis", "")),
        )
        reference = next(self.query_proj.parameters())
        expert.to(device=reference.device, dtype=reference.dtype)
        expert.confidence = float(metadata.get("confidence", 0.0))
        expert.usage = int(metadata.get("usage", 0))
        expert.last_score = float(metadata.get("last_score", 0.0))
        self.experts.append(expert)
        return expert

    @torch.no_grad()
    def birth(self, task_features: torch.Tensor | None = None, *,
              compiled: CompiledSkill | None = None, parent_id: int | None = None,
              created_step: int = 0, name: str = "", synopsis: str = "") -> int:
        """Append a disabled candidate.  The active model function is unchanged."""
        if compiled is None and task_features is not None and self.compiler is not None:
            compiled = self.compile(task_features)
        expert = self._append_empty({
            "parent_id": parent_id, "created_step": created_step, "status": "candidate",
            "name": name, "synopsis": synopsis,
        })
        if parent_id is not None:
            expert.clone_from(self.experts[int(parent_id)])
            if compiled is not None:
                expert.key.copy_(compiled.key.detach().to(expert.key))
        elif compiled is not None:
            expert.load_compiled(compiled)
        elif task_features is not None:
            key = self.address(task_features)
            if key.shape[0] != 1:
                raise ValueError("one locally taught skill must have exactly one address")
            expert.key.copy_(key[0].detach().to(expert.key))
        return expert.expert_id

    def restore_structure(self, manifest: Iterable[dict]) -> None:
        if len(self.experts):
            raise ValueError("cannot restore a growing cortex over an existing expert bank")
        for metadata in manifest:
            self._append_empty(dict(metadata))

    def manifest(self) -> list[dict]:
        return [expert.metadata() for expert in self.experts]

    def expert(self, expert_id: int) -> SkillExpert:
        try:
            return self.experts[int(expert_id)]
        except (IndexError, ValueError, TypeError) as exc:
            raise ValueError(f"unknown skill expert: {expert_id}") from exc

    @torch.no_grad()
    def commit(self, expert_id: int, *, confidence: float = 1.0, mature: bool = False) -> None:
        expert = self.expert(expert_id)
        if expert.status not in ("candidate", "juvenile"):
            raise ValueError(f"cannot commit an expert in state {expert.status}")
        expert.status = "mature" if mature else "juvenile"
        expert.confidence = float(confidence)

    @torch.no_grad()
    def mature(self, expert_id: int) -> None:
        expert = self.expert(expert_id)
        if expert.status != "juvenile":
            raise ValueError("only a juvenile expert can mature")
        expert.status = "mature"

    @torch.no_grad()
    def quarantine(self, expert_id: int) -> None:
        self.expert(expert_id).status = "quarantined"
        if self._forced_expert == int(expert_id):
            self._forced_expert = None

    @torch.no_grad()
    def set_dormant(self, expert_id: int) -> None:
        self.expert(expert_id).status = "dormant"

    @torch.no_grad()
    def prune(self, expert_id: int) -> None:
        expert = self.expert(expert_id)
        expert.status = "pruned"
        expert.down.zero_()
        expert.up.zero_()
        if self._forced_expert == int(expert_id):
            self._forced_expert = None

    @torch.no_grad()
    def discard_candidate(self, expert_id: int) -> None:
        """Remove an uncommitted tail candidate that was never written to the durable skill bank."""
        expert = self.expert(expert_id)
        if expert.expert_id != len(self.experts) - 1 or expert.status != "candidate":
            raise ValueError("only the newest uncommitted candidate can be discarded")
        del self.experts[-1]
        if self._forced_expert == expert.expert_id:
            self._forced_expert = None

    def nearest(self, task_key: torch.Tensor, *, active_only: bool = True) -> tuple[int | None, float]:
        candidates = [
            expert for expert in self.experts
            if (not active_only or expert.status in ACTIVE_STATUSES)
        ]
        if not candidates:
            return None, -1.0
        key = F.normalize(task_key.detach().reshape(-1), dim=0)
        scores = torch.stack([torch.dot(key, F.normalize(expert.key, dim=0))
                              for expert in candidates])
        pos = int(scores.argmax())
        return candidates[pos].expert_id, float(scores[pos])

    def _route(self, x: torch.Tensor) -> list[int | None]:
        active = [expert for expert in self.experts if expert.status in ACTIVE_STATUSES]
        if not active:
            self._last_scores = [-1.0] * x.shape[0]
            return [None] * x.shape[0]
        query = self.address(x)
        keys = torch.stack([F.normalize(expert.key, dim=0) for expert in active])
        confidence = query.new_tensor([expert.confidence for expert in active])
        scores = query @ keys.T + 0.05 * confidence[None]
        values, positions = scores.max(dim=-1)
        selected = [
            active[int(position)].expert_id if float(value) >= self.router_threshold else None
            for value, position in zip(values.detach(), positions.detach())
        ]
        self._last_scores = [float(value) for value in values.detach()]
        return selected

    @contextmanager
    def use_compiled(self, compiled: CompiledSkill):
        previous = self._compiled_override
        self._compiled_override = compiled
        try:
            yield
        finally:
            self._compiled_override = previous

    @contextmanager
    def force(self, expert_id: int | None):
        if expert_id is not None:
            self.expert(expert_id)
        previous = self._forced_expert
        self._forced_expert = None if expert_id is None else int(expert_id)
        try:
            yield
        finally:
            self._forced_expert = previous

    @contextmanager
    def suspend(self):
        """Temporarily disable both automatic and forced skill execution."""
        previous_auto = self.auto_route
        previous_forced = self._forced_expert
        previous_override = self._compiled_override
        self.auto_route = False
        self._forced_expert = None
        self._compiled_override = None
        try:
            yield
        finally:
            self.auto_route = previous_auto
            self._forced_expert = previous_forced
            self._compiled_override = previous_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_override is not None:
            self._last_selected = [-2] * x.shape[0]
            self._last_scores = [1.0] * x.shape[0]
            return self._compiled_override.apply(x)
        if self._forced_expert is not None:
            selected = [self._forced_expert] * x.shape[0]
            self._last_scores = [1.0] * x.shape[0]
        elif self.auto_route:
            selected = self._route(x)
        else:
            selected = [None] * x.shape[0]
            self._last_scores = [-1.0] * x.shape[0]
        self._last_selected = selected
        out = torch.zeros_like(x)
        for expert_id in sorted({value for value in selected if value is not None}):
            positions = [i for i, value in enumerate(selected) if value == expert_id]
            index = torch.tensor(positions, device=x.device, dtype=torch.long)
            expert = self.expert(expert_id)
            residual = expert(x.index_select(0, index))
            out = out.index_copy(0, index, residual)
            expert.usage += len(positions)
            expert.last_score = max(
                self._last_scores[i] for i in positions) if positions else expert.last_score
        return out

    def inference_parameter_count(self) -> int:
        """Parameters touched for routing plus at most one active expert."""
        active = [expert for expert in self.experts if expert.status in ACTIVE_STATUSES]
        router = sum(parameter.numel() for parameter in self.query_proj.parameters()) if active else 0
        one_expert = max(
            (sum(parameter.numel() for parameter in expert.parameters()) for expert in active),
            default=0,
        )
        return router + one_expert

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "rank": self.rank,
            "compiler": self.compiler_mode,
            "address_dim": self.address_dim,
            "router_threshold": self.router_threshold,
            "auto_route": self.auto_route,
            "forced_expert": self._forced_expert,
            "stored_experts": len(self.experts),
            "active_experts": sum(expert.status in ACTIVE_STATUSES for expert in self.experts),
            "selected": list(self._last_selected),
            "scores": list(self._last_scores),
            "experts": [
                {
                    **expert.metadata(),
                    "key_norm": float(expert.key.norm()),
                    "update_norm": float(torch.sqrt(
                        expert.down.detach().float().square().sum()
                        + expert.up.detach().float().square().sum())),
                }
                for expert in self.experts
            ],
        }


class GrowthController:
    """Conservative evidence accumulator for autonomous append-only births."""

    def __init__(self, *, error_threshold: float = 1.0, similarity_threshold: float = 0.35,
                 collision_threshold: float = 0.0, patience: int = 3, cooldown: int = 32):
        self.error_threshold = float(error_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.collision_threshold = float(collision_threshold)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.streaks: dict[str, int] = {}
        self.last_birth_step = -10 ** 9
        self.events: list[dict] = []

    def observe(self, evidence: GrowthEvidence, step: int) -> bool:
        qualifies = (
            evidence.error >= self.error_threshold
            and evidence.nearest_similarity <= self.similarity_threshold
            and evidence.gradient_cosine <= self.collision_threshold
            and not evidence.existing_expert_improved
        )
        streak = self.streaks.get(evidence.fingerprint, 0)
        streak = streak + 1 if qualifies else 0
        self.streaks[evidence.fingerprint] = streak
        decision = streak >= self.patience and step - self.last_birth_step >= self.cooldown
        event = {
            "step": int(step), "fingerprint": evidence.fingerprint,
            "error": float(evidence.error),
            "nearest_similarity": float(evidence.nearest_similarity),
            "gradient_cosine": float(evidence.gradient_cosine),
            "existing_expert_improved": bool(evidence.existing_expert_improved),
            "streak": streak, "birth": bool(decision),
        }
        self.events.append(event)
        if decision:
            self.last_birth_step = int(step)
            self.streaks[evidence.fingerprint] = 0
        return decision


def expert_snapshot(expert: SkillExpert) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in expert.state_dict().items()}


@torch.no_grad()
def restore_expert(expert: SkillExpert, snapshot: dict[str, torch.Tensor]) -> None:
    expert.load_state_dict({name: value.to(expert.down.device) for name, value in snapshot.items()})


def teach_expert(model, expert_id: int,
                 batches: list[tuple[torch.Tensor, torch.Tensor] |
                               tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 *, steps: int = 64, min_steps: int = 16, patience: int = 8, lr: float = 1e-2,
                 anchor_batches: list[torch.Tensor] | None = None,
                 anchor_weight: float = 1.0) -> LocalTeachingResult:
    """Refine one candidate while freezing the base model and every other expert."""
    if model.skill_cortex is None:
        raise ValueError("model has no growing cortex")
    if not batches:
        raise ValueError("local expert teaching requires at least one batch")
    cortex = model.skill_cortex
    expert = cortex.expert(expert_id)
    required = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    trainable_ids = {id(parameter) for parameter in expert.parameters()}
    was_training = model.training
    anchor_batches = anchor_batches or []
    update_before = torch.cat([expert.down.detach().flatten(), expert.up.detach().flatten()])
    initial_loss = final_loss = float("nan")
    anchor_kl = 0.0
    best_loss = float("inf")
    best_state = expert_snapshot(expert)
    stale = 0
    completed_steps = 0
    try:
        model.eval()
        for parameter, _old in required:
            parameter.requires_grad_(id(parameter) in trainable_ids)
        with torch.no_grad(), cortex.suspend():
            anchor_targets = [
                model(batch)[0].float().softmax(dim=-1) for batch in anchor_batches
            ]
        optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=0.0)
        for step in range(int(steps)):
            losses = []
            optimizer.zero_grad(set_to_none=True)
            with cortex.force(expert_id):
                for batch in batches:
                    idx, targets, *weights = batch
                    _logits, loss, _states, _delta = model(
                        idx, targets=targets,
                        loss_weight=(weights[0] if weights else None))
                    losses.append(loss)
                loss = torch.stack(losses).mean()
                if anchor_batches and anchor_weight:
                    penalties = []
                    for batch, reference in zip(anchor_batches, anchor_targets):
                        logits = model(batch)[0].float()
                        penalties.append(F.kl_div(
                            logits.log_softmax(dim=-1), reference, reduction="batchmean"))
                    loss = loss + float(anchor_weight) * torch.stack(penalties).mean()
            if step == 0:
                initial_loss = float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            optimizer.step()
            completed_steps = step + 1
            observed = float(loss.detach())
            if observed < best_loss - 1e-5:
                best_loss = observed
                best_state = expert_snapshot(expert)
                stale = 0
            else:
                stale += 1
            if completed_steps >= min_steps and stale >= patience:
                break
        restore_expert(expert, best_state)
        with torch.no_grad(), cortex.force(expert_id):
            losses = []
            for batch in batches:
                idx, targets, *weights = batch
                losses.append(model(
                    idx, targets=targets,
                    loss_weight=(weights[0] if weights else None))[1])
            final_loss = float(torch.stack(losses).mean())
            if anchor_batches:
                penalties = []
                for batch, reference in zip(anchor_batches, anchor_targets):
                    logits = model(batch)[0].float()
                    penalties.append(F.kl_div(
                        logits.log_softmax(dim=-1), reference, reduction="batchmean"))
                anchor_kl = float(torch.stack(penalties).mean())
    finally:
        for parameter, old in required:
            parameter.requires_grad_(old)
        model.train(was_training)
    update_after = torch.cat([expert.down.detach().flatten(), expert.up.detach().flatten()])
    return LocalTeachingResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        anchor_kl=anchor_kl,
        update_norm=float((update_after - update_before).float().norm()),
        steps=completed_steps,
    )
