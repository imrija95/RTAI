"""Evolvable one-pass local learning for the fractal model.

The learner deliberately does not use autograd or an optimizer.  It observes the
next token, constructs an output-space prediction error from the tied embedding,
projects that signal back through deterministic random feedback, and applies one
rank-one local update per observed block.  The ordinary fast-weight delta rule
continues to update the persistent FractalState during the same forward stream.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


GROUPS = ("embedding", "qk", "routing", "projection", "mlp")


@dataclass
class RuleGenes:
    log10_lr: float = -3.5
    error: float = 1.0
    hebbian: float = 0.0
    oja: float = 0.01
    decay: float = 0.0001


@dataclass
class PlasticityGenome:
    """A small, named and versioned learning law rather than trained model weights."""

    version: int = 1
    feedback_seed: int = 17
    feedback_scale: float = 1.0
    clip_ratio: float = 4.0
    surprise_threshold: float = 0.0
    eligibility_decay: float = 0.9
    fast_base: float = 0.0
    fast_surprise: float = 0.1
    fast_usage: float = 0.1
    rules: dict[str, RuleGenes] = field(
        default_factory=lambda: {name: RuleGenes() for name in GROUPS})

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported plasticity genome version: {self.version}")
        if set(self.rules) != set(GROUPS):
            raise ValueError(f"rules must contain exactly: {', '.join(GROUPS)}")
        vals = self.to_vector()
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("genome contains a non-finite value")
        for value, (lo, hi) in zip(vals, self.bounds()):
            if not lo <= value <= hi:
                raise ValueError(f"gene {value} outside [{lo}, {hi}]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PlasticityGenome":
        raw = dict(data)
        raw["rules"] = {name: RuleGenes(**genes) for name, genes in raw["rules"].items()}
        genome = cls(**raw)
        genome.validate()
        return genome

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlasticityGenome":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_vector(self) -> list[float]:
        head = [self.feedback_scale, self.clip_ratio, self.surprise_threshold,
                self.eligibility_decay,
                self.fast_base, self.fast_surprise, self.fast_usage]
        body = []
        for name in GROUPS:
            rule = self.rules[name]
            body.extend([rule.log10_lr, rule.error, rule.hebbian, rule.oja, rule.decay])
        return head + body

    @classmethod
    def from_vector(cls, values, feedback_seed: int = 17) -> "PlasticityGenome":
        values = [float(v) for v in values]
        if len(values) != 7 + 5 * len(GROUPS):
            raise ValueError(f"expected {7 + 5 * len(GROUPS)} genes, got {len(values)}")
        genome = cls(feedback_seed=feedback_seed,
                     feedback_scale=values[0], clip_ratio=values[1],
                     surprise_threshold=values[2], eligibility_decay=values[3],
                     fast_base=values[4], fast_surprise=values[5], fast_usage=values[6])
        pos = 7
        genome.rules = {}
        for name in GROUPS:
            genome.rules[name] = RuleGenes(*values[pos:pos + 5])
            pos += 5
        genome.validate()
        return genome

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        head = [(0.05, 3.0), (1.1, 10.0), (0.0, 8.0), (0.0, 0.999),
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        rule = [(-6.0, -1.0), (-3.0, 3.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 0.1)]
        return head + rule * len(GROUPS)


def _group_for(name: str) -> str | None:
    if name.endswith("to_qk"):
        return "qk"
    if name.endswith("to_beta") or name.endswith("gate") or name.endswith("to_f"):
        return "routing"
    if name.endswith("unit.proj"):
        return "projection"
    if ".mlp." in name and (name.endswith("fc") or name.endswith("proj")):
        return "mlp"
    return None


class OnlinePlasticLearner:
    """Apply a PlasticityGenome to a live model with explicit in-place updates."""

    def __init__(self, model, genome: PlasticityGenome):
        genome.validate()
        self.model = model
        self.genome = genome
        self.captures: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.modules: dict[str, nn.Linear] = {}
        self.feedback: dict[tuple[str, int, int, str], torch.Tensor] = {}
        self.handles = []
        self.sample_ids: set[str] = set()
        self.last_update_norms = {name: 0.0 for name in GROUPS}
        self.last_fast_update_norms: list[list[float]] = []
        self.total_tokens = 0
        self.last_loss = float("nan")
        self.stable = True

        for name, module in model.named_modules():
            group = _group_for(name)
            if group is not None and isinstance(module, nn.Linear):
                self.modules[name] = module
                self.captures[name] = []
                self.handles.append(module.register_forward_hook(self._capture(name)))
            elif name == "head" and isinstance(module, nn.Linear):
                self.modules[name] = module
                self.captures[name] = []
                self.handles.append(module.register_forward_hook(self._capture(name)))

        self.initial_norms = {
            name: max(float(module.weight.detach().norm()), 1e-8) for name, module in self.modules.items()
            if name != "head"
        }
        self.initial_embedding_norm = max(float(model.tok_emb.weight.detach().norm()), 1e-8)
        for block in model.blocks:
            block.unit._log_share = True

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture(self, name):
        def hook(_module, args, output):
            self.captures[name].append((args[0].detach(), output.detach()))
        return hook

    def _feedback(self, name: str, source: int, target: int, device, dtype) -> torch.Tensor:
        key = (name, source, target, str(device))
        if key not in self.feedback:
            if source == target:
                self.feedback[key] = (torch.eye(source, device=device, dtype=dtype)
                                      * self.genome.feedback_scale)
                return self.feedback[key]
            digest = hashlib.sha256(f"{self.genome.feedback_seed}:{name}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF
            generator = torch.Generator(device="cpu").manual_seed(seed)
            matrix = torch.randint(0, 2, (source, target), generator=generator,
                                   dtype=torch.float32).mul_(2).sub_(1)
            matrix.mul_(self.genome.feedback_scale / math.sqrt(max(source, 1)))
            self.feedback[key] = matrix.to(device=device, dtype=dtype)
        return self.feedback[key]

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        dims = tuple(range(value.ndim - 1))
        denom = weight.expand_as(value).sum(dims).clamp_min(1.0)
        return (value * weight).sum(dims) / denom

    def _clip_module(self, name: str, module: nn.Linear) -> None:
        limit = self.initial_norms[name] * self.genome.clip_ratio
        norm = float(module.weight.norm())
        if not math.isfinite(norm):
            self.stable = False
            return
        if norm > limit:
            module.weight.mul_(limit / max(norm, 1e-12))

    def _update_linear(self, name, module, global_error, token_weight) -> tuple[str, float]:
        group = _group_for(name)
        rule = self.genome.rules[group]
        lr = 10.0 ** rule.log10_lr
        total = torch.zeros_like(module.weight)
        bias_total = torch.zeros_like(module.bias) if module.bias is not None else None
        count = 0
        for inputs, outputs in self.captures[name]:
            if inputs.ndim != 3 or outputs.ndim != 3 or inputs.shape[:2] != global_error.shape[:2]:
                continue
            fb = self._feedback(name, global_error.shape[-1], outputs.shape[-1],
                                outputs.device, outputs.dtype)
            local_error = global_error.to(outputs.dtype) @ fb
            if name.endswith("mlp.fc"):
                local_error = local_error * torch.sigmoid(1.702 * outputs)
            ebar = self._weighted_mean(local_error, token_weight)
            running = torch.zeros_like(inputs[:, 0])
            normalizer = 0.0
            traces = []
            for t in range(inputs.shape[1]):
                running = self.genome.eligibility_decay * running + inputs[:, t]
                normalizer = self.genome.eligibility_decay * normalizer + 1.0
                traces.append((running / normalizer).unsqueeze(1))
            eligibility = torch.cat(traces, dim=1)
            xbar = self._weighted_mean(eligibility, token_weight)
            ybar = self._weighted_mean(outputs, token_weight)
            direction = (-rule.error * torch.outer(ebar, xbar)
                         + rule.hebbian * torch.outer(ybar, xbar)
                         - rule.oja * ybar.square().mean() * module.weight
                         - rule.decay * module.weight)
            total.add_(direction)
            if bias_total is not None:
                bias_total.add_(-rule.error * ebar - rule.decay * module.bias)
            count += 1
        if count:
            delta = lr * total / count
            module.weight.add_(delta)
            if bias_total is not None:
                module.bias.add_(lr * bias_total / count)
            self._clip_module(name, module)
            return group, float(delta.norm())
        return group, 0.0

    def _update_embedding(self, idx, targets, hidden, probs, global_error, token_weight,
                          explicit_mask: bool) -> float:
        rule = self.genome.rules["embedding"]
        lr = 10.0 ** rule.log10_lr
        emb = self.model.tok_emb.weight
        before = emb.clone()
        weight = token_weight.reshape(-1)
        flat_idx = idx.reshape(-1)
        flat_tgt = targets.reshape(-1)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_error = global_error.reshape(-1, global_error.shape[-1])
        flat_probs = probs.reshape(-1, probs.shape[-1])
        active = weight > 0
        if active.any():
            a_idx = flat_idx[active]
            a_tgt = flat_tgt[active]
            a_hidden = flat_hidden[active]
            a_error = flat_error[active]
            a_probs = flat_probs[active]
            w = weight[active].unsqueeze(-1)
            emb.index_add_(0, a_idx, -lr * rule.error * w * a_error)
            p_tgt = a_probs.gather(1, a_tgt[:, None])
            emb.index_add_(0, a_tgt, lr * rule.error * w * (1.0 - p_tgt) * a_hidden)
            pred = a_probs.argmax(dim=-1)
            p_pred = a_probs.gather(1, pred[:, None])
            emb.index_add_(0, pred, -lr * rule.error * w * p_pred * a_hidden)
            if explicit_mask:
                # Delayed answer error reinforces earlier observed symbols through the same
                # decaying eligibility trace used by the linear modules.
                B, T = idx.shape
                for b in range(B):
                    for t in token_weight[b].nonzero(as_tuple=False).flatten().tolist():
                        norm = sum(self.genome.eligibility_decay ** distance
                                   for distance in range(t + 1))
                        for s in range(t, -1, -1):
                            coeff = self.genome.eligibility_decay ** (t - s)
                            emb[idx[b, s]].add_(-lr * rule.error * coeff / norm * global_error[b, t])
        emb.mul_(max(0.0, 1.0 - lr * rule.decay))
        limit = self.initial_embedding_norm * self.genome.clip_ratio
        norm = float(emb.norm())
        if not math.isfinite(norm):
            self.stable = False
        elif norm > limit:
            emb.mul_(limit / max(norm, 1e-12))
        return float((emb - before).norm())

    def _modulate_fast_weights(self, mean_surprise: float) -> None:
        for block in self.model.blocks:
            share = block.unit._last_share
            for scale, cell in enumerate(block.unit.cells):
                usage = (float(share[scale]) if share is not None else 1.0 / len(block.unit.cells))
                log_gain = (self.genome.fast_base
                            + self.genome.fast_surprise * math.log1p(max(mean_surprise, 0.0))
                            + self.genome.fast_usage * usage)
                cell.set_beta_gain(math.exp(max(-1.5, min(1.5, log_gain))))

    @torch.no_grad()
    def learn_block(self, idx: torch.Tensor, targets: torch.Tensor, states, sample_id: str,
                    loss_weight: torch.Tensor | None = None):
        """Observe one unique block once and return logits, new state, and scalar loss."""
        if sample_id in self.sample_ids:
            raise ValueError(f"training sample replayed: {sample_id}")
        self.sample_ids.add(sample_id)
        for values in self.captures.values():
            values.clear()

        self.model.eval()
        logits, new_states = self.model.forward_stream(idx, states)
        probs = logits.float().softmax(dim=-1)
        surprise = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), reduction="none").view_as(targets)
        self.last_loss = float(surprise.mean())
        explicit_mask = loss_weight is not None
        token_weight = (loss_weight.to(logits.dtype) if explicit_mask else
                        (surprise >= self.genome.surprise_threshold).to(logits.dtype))
        if token_weight.sum() == 0:
            token_weight.reshape(-1)[surprise.argmax()] = 1.0

        embedding = self.model.tok_emb.weight.detach().float()
        global_error = probs @ embedding - embedding.index_select(0, targets.reshape(-1)).view(
            *targets.shape, embedding.shape[-1])
        if not self.captures.get("head"):
            raise RuntimeError("head input was not captured")
        hidden = self.captures["head"][-1][0].float()

        norms = {name: 0.0 for name in GROUPS}
        norms["embedding"] = self._update_embedding(
            idx, targets, hidden, probs, global_error, token_weight, explicit_mask)
        for name, module in self.modules.items():
            if name == "head":
                continue
            group, norm = self._update_linear(name, module, global_error, token_weight)
            norms[group] += norm
        self.last_update_norms = norms
        self.last_fast_update_norms = []
        for old, new in zip(states, new_states):
            self.last_fast_update_norms.append([
                float((new_w - old_w).norm()) for old_w, new_w in zip(old.W, new.W)
            ])
        self._modulate_fast_weights(self.last_loss)
        self.total_tokens += idx.numel()
        self.stable &= all(math.isfinite(v) for v in norms.values())
        return logits, new_states, self.last_loss
