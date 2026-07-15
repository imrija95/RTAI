"""Model as a GENERATING RULE, not a stored grid.

Instead of a stack of N distinct layers we have ONE shared block, unrolled to depth
`depth` (weight-tied recursion, à la Universal Transformer). Parameters are shared,
but the state (fast weights) is separate for each unrolling. Consequences:

  - size = how far you unroll the rule → `depth` is a runtime knob (unrolled at run time),
  - orders of magnitude fewer distinct parameters → higher sample efficiency,
  - position is held by the recurrent state → no positional embeddings, length is unbounded.

Block = LayerNorm → FractalUnit (fractal memory) → LayerNorm → a small SHARED MLP.
The large per-layer MLP (von-Neumann-style static scaffolding) is gone; there is only
one nonlinearity, shared across depth.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from fractal.unit import FractalUnit


@dataclass
class Config:
    vocab_size: int = 8000
    n_embd: int = 384
    n_head: int = 8
    depth: int = 6                 # how many times to unroll the shared block (weight-tied)
    n_scales: int = 4              # number of fractal levels (time scales)
    tau0: float = 16.0             # shortest time constant
    rho: float = 4.0               # geometric ratio of the ladder (τ_ℓ = τ₀·ρ^ℓ)
    chunk_size: int = 64
    mlp_ratio: int = 2             # small shared MLP (not 4× as in the classic design)
    dropout: float = 0.0
    bias: bool = False
    high_pass_keys: bool = False   # key/query from novelty (feature − running average) → separable keys
    selective: bool = False        # LEARNED dynamics: decay = data-dependent forget gate (not a fixed γ)
    untie: bool = False            # a separate block per depth (more parameters; erodes the "one rule")
    n_experts: int = 1             # functional MoE experts in the MLP (1 = full MLP; >1 = router + K disjoint MLPs)
    moe_mode: str = "soft"         # soft = all experts; top1 = true sparse token dispatch
    event_budget: float = 1.0       # fraction of positions sent through the expensive memory unit

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.mlp_ratio * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.proj = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class MoEMLP(nn.Module):
    """Functional MoE experts (K disjoint MLPs) + a content-based router. Soft mixture (dense — all
    experts weighted by softmax) → fully differentiable, and mitosis (−ln2 on the router column) is EXACT.
    Router usage is logged: load-balance (neg-entropy of the mean → uniform) + per-task analysis (hemispheres).
    This creates a functional symmetry (interchangeable experts) that load-balance + a separable input BREAK."""

    def __init__(self, cfg: "Config", n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.mode = getattr(cfg, "moe_mode", "soft")
        if self.mode not in ("soft", "top1"):
            raise ValueError(f"unknown MoE mode: {self.mode}")
        # Mitosis: every expert starts as the same function. Sparse capacity is added without
        # shocking the network; specialization can emerge only after routing begins.
        stem = MLP(cfg)
        self.experts = nn.ModuleList([stem] + [copy.deepcopy(stem) for _ in range(n_experts - 1)])
        self.router = nn.Linear(cfg.n_embd, n_experts, bias=True)
        self._log = False
        self._log_buf: list = []
        self._last_usage = None                       # (B,T,K) last router softmax (per-task analysis)

    def forward(self, x):
        g = self.router(x).softmax(dim=-1)            # (B,T,K)
        if self.mode == "top1":
            chosen = g.argmax(dim=-1)                 # (B,T), hard routing in the real forward
            self._last_usage = F.one_hot(chosen, self.n_experts).to(g.dtype).detach()
        else:
            chosen = None
            self._last_usage = g.detach()
        if self._log:
            self._log_buf.append(g.mean(dim=(0, 1)))  # (K,) mean usage
        if self.mode == "top1":
            xf = x.reshape(-1, x.shape[-1])
            gf = g.reshape(-1, self.n_experts)
            cf = chosen.reshape(-1)
            out = torch.zeros_like(xf)
            for e, expert in enumerate(self.experts):
                pos = (cf == e).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                p = gf.index_select(0, pos)[:, e:e + 1]
                # Forward scale is exactly one, so cloned experts reproduce the dense stem.
                # Backward still reaches the router through p.
                straight_through_one = p / p.detach().clamp_min(1e-9)
                y = expert(xf.index_select(0, pos)) * straight_through_one
                out = out.index_add(0, pos, y)
            return out.view_as(x)
        out = 0
        for e in range(self.n_experts):
            out = out + g[..., e:e + 1] * self.experts[e](x)
        return out

    def pop_balance_loss(self):
        if not self._log_buf:
            return None
        g = torch.stack(self._log_buf).mean(0)        # (K,)
        self._log_buf = []
        return (g * (g + 1e-9).log()).sum()           # neg-entropy; min → uniform expert usage


class Block(nn.Module):
    """A single shared rule block (applied several times via unrolling)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.unit = FractalUnit(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MoEMLP(cfg, cfg.n_experts) if getattr(cfg, "n_experts", 1) > 1 else MLP(cfg)
        self.event_budget = float(getattr(cfg, "event_budget", 1.0))
        if not 0.0 < self.event_budget <= 1.0:
            raise ValueError("event_budget must be in (0, 1]")
        self._last_event_share = 1.0

    def _event_patches(self, z, state):
        """Compress completed causal patches into one global event each.

        A patch is summarized only after its final token arrives, so neither full nor streaming
        execution can leak future information. Unfinished patches remain in persistent state.
        """
        B, T, E = z.shape
        if self.event_budget >= 1.0:
            return None, None, state
        if state is None:
            state = self.unit.init_state(B, z.device, z.dtype)
        stride = max(2, round(1.0 / self.event_budget))
        total = state.event_sum if state.event_sum is not None else z.new_zeros(B, E)
        count = state.event_count
        summaries, positions = [], []
        for t in range(T):
            total = total + z[:, t]
            count += 1
            if count == stride:
                summaries.append((total / stride).unsqueeze(1))
                positions.append(t)
                total = z.new_zeros(B, E)
                count = 0
        state.event_sum, state.event_count = total, count
        state.event_prev = z[:, -1]
        state.event_n += T
        self._last_event_share = len(positions) / T
        if not summaries:
            return z[:, :0], z.new_empty((0,), dtype=torch.long), state
        return torch.cat(summaries, dim=1), torch.tensor(positions, device=z.device), state

    def forward(self, x, state=None, return_delta=False):
        z = self.ln1(x)
        patches, idx, state = self._event_patches(z, state)
        if patches is None:
            y, state, dn = self.unit(z, state, return_delta)
        elif patches.shape[1] == 0:
            y, dn = torch.zeros_like(x), None
        else:
            partial_sum, partial_count, event_n = state.event_sum, state.event_count, state.event_n
            ys, state, dn = self.unit(patches, state, return_delta)
            y = torch.zeros_like(x).scatter(
                1, idx.view(1, -1, 1).expand(x.shape[0], -1, x.shape[-1]),
                ys.to(x.dtype))
            state.event_sum, state.event_count, state.event_n = partial_sum, partial_count, event_n
            state.event_prev = z[:, -1]
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, state, dn

    def step(self, x, state):
        z = self.ln1(x)
        patches, idx, state = self._event_patches(z, state)
        if patches is None:
            y, state = self.unit.step(z, state)
        elif patches.shape[1] == 0:
            y = torch.zeros_like(x)
        else:
            partial_sum, partial_count, event_n = state.event_sum, state.event_count, state.event_n
            ys, state = self.unit.step(patches, state)
            y = torch.zeros_like(x).scatter(
                1, idx.view(1, -1, 1).expand(x.shape[0], -1, x.shape[-1]),
                ys.to(x.dtype))
            state.event_sum, state.event_count, state.event_n = partial_sum, partial_count, event_n
            state.event_prev = z[:, -1]
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, state


class FractalLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # untie=False → one shared block (fractal, weight-tied); untie=True → a separate block per depth
        n_blocks = cfg.depth if cfg.untie else 1
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_blocks)])
        self.block = self.blocks[0]                 # alias (compat + shared path)
        self.grad_ckpt = False                      # runtime switch (train.py), does not change the model
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight      # weight tying
        self.apply(self._init_weights)
        # Module-wide initialization randomizes copied experts independently. Restore exact
        # function-preserving mitosis after it; loading a checkpoint overwrites these values.
        for block in self.blocks:
            if isinstance(block.mlp, MoEMLP):
                stem = block.mlp.experts[0].state_dict()
                for expert in block.mlp.experts[1:]:
                    expert.load_state_dict(stem)
        n_all = sum(p.numel() for p in self.parameters())
        print(f"[FractalLM] params: {n_all/1e6:.2f}M | depth={cfg.depth}"
              f"{' (untied)' if cfg.untie else ' (weight-tied)'} | n_scales={cfg.n_scales}"
              f" | n_embd={cfg.n_embd} γ={[round(g,3) for g in self.block.unit.gammas]}"
              f" | moe={getattr(cfg, 'moe_mode', 'soft')}"
              f" | event_budget={getattr(cfg, 'event_budget', 1.0):.2f}")

    def parameter_counts(self) -> tuple[int, int]:
        """Return stored and actually active parameters for one token."""
        stored = sum(p.numel() for p in self.parameters())
        inactive = 0
        for block in self.blocks:
            mlp = block.mlp
            if isinstance(mlp, MoEMLP) and mlp.mode == "top1":
                inactive += sum(p.numel() for expert in mlp.experts[1:]
                                for p in expert.parameters())
        return stored, stored - inactive

    def event_share(self) -> float:
        return sum(b._last_event_share for b in self.blocks) / len(self.blocks)

    def block_at(self, d):
        return self.blocks[d] if self.cfg.untie else self.blocks[0]

    def set_moe_log(self, on: bool):
        for b in self.blocks:
            if hasattr(b.mlp, "_log"):
                b.mlp._log = on

    def pop_moe_balance(self):
        """Mean load-balance loss (neg-entropy of usage) across MoE blocks; None if there are no MoE."""
        ls = [b.mlp.pop_balance_loss() for b in self.blocks if hasattr(b.mlp, "pop_balance_loss")]
        ls = [x for x in ls if x is not None]
        return sum(ls) / len(ls) if ls else None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def init_states(self, batch_size, device, dtype=torch.float32, depth=None):
        """One FractalState per unrolling (parameters shared, states separate)."""
        depth = depth or self.cfg.depth
        return [self.block_at(d).unit.init_state(batch_size, device, dtype) for d in range(depth)]

    def forward(self, idx, targets=None, states=None, return_delta=False, depth=None,
                loss_weight=None):
        """idx:(B,T). states: a list of length depth (persistence) or None (start from W0).
        depth: runtime unrolling (default cfg.depth) — the model can be unrolled deeper/shallower.
        loss_weight: (B,T) per-token weights for CE (recall: answer weighted heavily, filler 0)."""
        depth = depth or self.cfg.depth
        x = self.drop(self.tok_emb(idx))
        new_states, all_dn = [], ([] if return_delta else None)
        ckpt = self.grad_ckpt and self.training and states is None and not return_delta
        for d in range(depth):
            blk = self.block_at(d)
            if ckpt:                                   # gradient checkpointing (saves VRAM)
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, b=blk: b(inp, None, False)[0], x, use_reentrant=False)
            else:
                st = states[d] if states is not None else None
                x, st, dn = blk(x, st, return_delta)
                new_states.append(st)
                if return_delta:
                    all_dn.append(dn)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            if loss_weight is None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                       ignore_index=-1)
            else:                                              # weighted CE (recall curriculum)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                     ignore_index=-1, reduction="none")
                w = loss_weight.reshape(-1)
                loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        return logits, loss, new_states, all_dn

    # ---- streaming (persistent) run for chat ----
    def forward_stream(self, idx, states, depth=None):
        depth = depth or self.cfg.depth
        x = self.drop(self.tok_emb(idx))
        new_states = []
        for d in range(depth):
            x, st = self.block_at(d).step(x, states[d])
            new_states.append(st)
        return self.head(self.ln_f(x)), new_states

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, states, temperature=1.0, top_k=None, depth=None):
        """Persistent generation: W is carried forward (not reset) → memory persists.
        Returns (generated_tokens, states) — save the state for the next turn/session."""
        self.eval()
        logits, states = self.forward_stream(idx, states, depth)
        gen = []
        for _ in range(max_new_tokens):
            lg = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, dim=-1), num_samples=1)
            gen.append(nxt)
            logits, states = self.forward_stream(nxt, states, depth)
        gen = torch.cat(gen, dim=1) if gen else idx[:, :0]
        return gen, states
