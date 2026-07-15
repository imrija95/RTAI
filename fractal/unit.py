"""Fractal compute unit — replaces "attention".

FractalUnit = a geometric ladder of fast-weight cells with SHARED projections and ONE
rule, differing only in the decay γ (time scale). Key/query are learned projections
(context via conv); the VALUE is the raw state (token identity) — so that a read decodes
back to the correct token through the tied head, even for UNSEEN tokens (generalizable copying).

    τ_ℓ = τ₀ · ρ^ℓ,   γ_ℓ = exp(−1/τ_ℓ),   last level γ = 1 (permanent memory)

The fine level (small γ) forgets quickly → local fluency; the top one (γ→1) holds facts
forever. All read at full resolution (no pooling → precise keys for precise recall).
A learned gate (softmax over levels, per token/head) mixes the reads:

    o_t = Σ_ℓ g_ℓ · (W_ℓ q_t)

The ComputeUnit interface is a swap point: FractalUnit can be substituted for any
non-fractal unit. n_scales=1 gives the baseline (a single DeltaNet, γ=1).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fractal.cell import FastWeightCell


@dataclass
class FractalState:
    """Opaque mutable unit state (per layer). Neither the model nor the chat look inside
    — they only thread it through and store it. State = fast weights per level + a causal conv
    buffer + a running average (for high-pass keys) — all for streaming."""
    W: list[torch.Tensor]            # len n_scales, each (B, n_head, hd, hd)
    conv: torch.Tensor | None = None  # (B, conv_k-1, n_embd) — only during streaming
    hp_sum: torch.Tensor | None = None  # (B, n_embd) running sum of features (high-pass)
    hp_n: int = 0                     # number of positions counted
    # Event-cortex bookkeeping. These fields are inert for the default dense path and therefore
    # keep old checkpoints and the ordinary FractalUnit numerically unchanged.
    event_prev: torch.Tensor | None = None  # previous local feature, including skipped tokens
    event_n: int = 0                        # number of local positions seen by this depth
    event_sum: torch.Tensor | None = None   # unfinished causal patch sum
    event_count: int = 0                    # positions accumulated in the unfinished patch

    def _map(self, fn_w, fn_c):
        return FractalState([fn_w(w) for w in self.W],
                            None if self.conv is None else fn_c(self.conv),
                            None if self.hp_sum is None else fn_c(self.hp_sum), self.hp_n,
                            None if self.event_prev is None else fn_c(self.event_prev), self.event_n,
                            None if self.event_sum is None else fn_c(self.event_sum), self.event_count)

    def to(self, device):  return self._map(lambda w: w.to(device), lambda c: c.to(device))
    def cpu(self):         return self._map(lambda w: w.cpu(), lambda c: c.cpu())
    def detach(self):      return self._map(lambda w: w.detach(), lambda c: c.detach())
    def clone(self):       return self._map(lambda w: w.clone(), lambda c: c.clone())


class ComputeUnit(nn.Module, ABC):
    """Interchangeable compute unit (both fractal and non-fractal implementations)."""

    @abstractmethod
    def init_state(self, batch_size: int, device, dtype=torch.float32) -> FractalState: ...

    @abstractmethod
    def forward(self, x, state: FractalState | None = None, return_delta: bool = False):
        """Full pass (training/parallel). x:(B,T,E) → (out:(B,T,E), new_state, aux|None)."""

    @abstractmethod
    def step(self, x, state: FractalState):
        """Streaming step (chat). x:(B,T,E) linked via state.conv → (out, new_state)."""


def _make_gammas(n_scales: int, tau0: float, rho: float) -> list[float]:
    """Geometric decay ladder; last level = 1.0 (permanent memory)."""
    gammas = []
    for l in range(n_scales):
        if l == n_scales - 1:
            gammas.append(1.0)
        else:
            tau = tau0 * (rho ** l)
            gammas.append(math.exp(-1.0 / tau))
    return gammas


class FractalUnit(ComputeUnit):
    conv_k = 8        # the window must reach the name (key) so multiple facts can be distinguished

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.head_dim
        self.n_embd = cfg.n_embd
        self.n_scales = cfg.n_scales
        self.high_pass = cfg.high_pass_keys
        self.selective = cfg.selective
        self.tau0 = cfg.tau0                 # ladder parameters — needed also for growth (neurogenesis)
        self.rho = cfg.rho
        self.chunk_size = cfg.chunk_size
        self.gammas = _make_gammas(cfg.n_scales, cfg.tau0, cfg.rho)

        # shared causal (depthwise) convolution — a position sees a few preceding tokens
        # (key→value binding, as in DeltaNet/Mamba). Shared across all levels.
        self.conv = nn.Conv1d(cfg.n_embd, cfg.n_embd, kernel_size=self.conv_k,
                              groups=cfg.n_embd, bias=True)
        # shared slow projections: key+query learned (value = raw state, see _project)
        self.to_qk = nn.Linear(cfg.n_embd, 2 * cfg.n_embd, bias=cfg.bias)
        self.to_beta = nn.Linear(cfg.n_embd, cfg.n_head, bias=True)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        # gate over levels (per token, per head)
        self.gate = nn.Linear(cfg.n_embd, cfg.n_head * cfg.n_scales, bias=True)

        # ladder of cells — one rule, different γ
        self.cells = nn.ModuleList([
            FastWeightCell(cfg.n_head, cfg.head_dim, gamma=g, chunk_size=cfg.chunk_size)
            for g in self.gammas
        ])

        # selective (learned) dynamics: forget gate per (level, head).
        # Init: weight=0, bias=logit(γ_ℓ) → at start f = σ(bias) = γ_ℓ (a purely geometric ladder
        # as the prior), then the model learns to modulate the decay by content.
        if self.selective:
            self.to_f = nn.Linear(cfg.n_embd, cfg.n_scales * cfg.n_head, bias=True)
            with torch.no_grad():
                self.to_f.weight.zero_()
                bias = torch.empty(cfg.n_scales, cfg.n_head)
                for l, g in enumerate(self.gammas):
                    gg = min(max(g, 1e-4), 1 - 1e-4)
                    bias[l].fill_(math.log(gg / (1 - gg)))     # logit(γ_ℓ)
                self.to_f.bias.copy_(bias.reshape(-1))

        # instrumentation / gate load-balancing (against collapse onto a single level)
        self._log_gate = False
        self._gate_log: list = []
        # routing prior: None → neg-entropy (target uniform); p∈(0,1) → CE to a prior that holds
        # the PERMANENT scale at p (recall wants concentration on permanent, not uniform — see A/B)
        self.gate_prior_perm = None
        # telemetry for neurogenesis: last mean gate share per scale (L,) — growth trigger
        self._log_share = False
        self._last_share = None
        # DIAGNOSTICS (inference-time ablation): when != None, the gate reads ONLY from this scale
        # (one-hot). A free test of "do the fast scales dilute recall?" without training. Off by default.
        self.force_scale = None

    # ---- helpers ----
    def _project(self, x, xc, hp_state=None):
        """x:(B,T,E) raw state (→ value), xc:(B,T,E) after conv (→ β; and key/query).
        high_pass: key/query from novelty feat = xc − causal running average (removes the
        common-mode template → separable keys). hp_state=(sum,n) for streaming.
        Returns q,k,v (B,H,T,hd), beta (B,H,T), new_hp_state."""
        B, T, _ = xc.shape
        H, hd = self.n_head, self.hd
        feat, new_hp = xc, None
        if self.high_pass:
            s0 = hp_state[0].unsqueeze(1) if hp_state is not None else 0.0
            n0 = hp_state[1] if hp_state is not None else 0
            csum = s0 + xc.cumsum(dim=1)                                # (B,T,E) running sum
            cnt = (n0 + torch.arange(1, T + 1, device=xc.device)).to(xc.dtype).view(1, T, 1)
            feat = xc - csum / cnt                                      # novelty = deviation from the mean
            new_hp = (csum[:, -1], n0 + T)
        q, k = self.to_qk(feat).split(H * hd, dim=2)
        q = F.normalize(q.view(B, T, H, hd), dim=-1).transpose(1, 2)
        k = F.normalize(k.view(B, T, H, hd), dim=-1).transpose(1, 2)
        v = x.view(B, T, H, hd).transpose(1, 2)                        # value = raw state (token identity)
        beta = torch.sigmoid(self.to_beta(xc)).transpose(1, 2)         # (B,H,T)
        return q, k, v, beta, new_hp

    def _combine(self, outs, xc):
        """Mix reads over levels with the learned gate. outs: list L × (B,H,T,hd)."""
        B, T, _ = xc.shape
        H, L = self.n_head, self.n_scales
        g = self.gate(xc).view(B, T, H, L).softmax(dim=-1).permute(0, 2, 1, 3)  # (B,H,T,L)
        if self.force_scale is not None:                     # ablation: read only from one scale
            g = torch.zeros_like(g)
            g[..., self.force_scale] = 1.0
        if self.training and self._log_gate:
            self._gate_log.append(g.mean(dim=(0, 1, 2)))               # (L,) level usage
        if self._log_share:
            self._last_share = g.mean(dim=(0, 1, 2)).detach()          # (L,) share per scale (growth trigger)
        o = torch.zeros_like(outs[0])
        for l in range(L):
            o = o + g[..., l].unsqueeze(-1) * outs[l]                  # (B,H,T,hd)
        return o.transpose(1, 2).reshape(B, T, self.n_embd)            # (B,T,E)

    def pop_gate_balance_loss(self):
        """Gate routing regularization (mean level usage g, shape (L,)).
        gate_prior_perm=None (default): neg-entropy → target UNIFORM (even use of the ladder).
        gate_prior_perm=p: cross-entropy to a prior π that holds the mass of the PERMANENT scale
          (last index) at p and spreads the rest evenly over the fast ones — for recall, uniform is a bad
          target (it dilutes permanent); this keeps permanent ~constant regardless of the number of scales.
          L=2,p=0.5 → π=[0.5,0.5] = exactly today's uniform (reproduces the working baseline)."""
        if not self._gate_log:
            return None
        g = torch.stack(self._gate_log).mean(0)                        # (L,)
        self._gate_log = []
        if self.gate_prior_perm is None:
            return (g * (g + 1e-9).log()).sum()                        # = −entropy; min → uniform
        L = g.shape[0]
        p = float(self.gate_prior_perm)
        pi = g.new_full((L,), (1.0 - p) / max(L - 1, 1))
        pi[-1] = p                                                     # permanent scale = index L−1
        return -(pi * (g + 1e-9).log()).sum()                          # CE(π,g); min at g=π

    @torch.no_grad()
    def grow_scale(self, birth_beta_gain: float = 2.0, demote: bool = True):
        """NEUROGENESIS: sprout a new scale (γ=1, empty W0, high plasticity).

        demote=True (default): the old permanent scale (γ=1) is "demoted" to a finite γ<1 →
          the invariant gammas==_make_gammas(n) holds (reload without saving γ). BUT it destroys the memory
          in the original permanent scale (see A/B v2 — destructive).
        demote=False (gentle v3): existing scales UNCHANGED → several permanent (γ=1) stores
          side by side (fresh stores à la dentate gyrus). Preserves learned memory. The ladder is no longer
          _make_gammas(n) → a grown checkpoint does NOT reload cleanly (fine for A/B, eval is in-process).

        Writes are unconditional → the newborn fills up right away; the gate (new rows = 0) LEARNS to read.
        Returns (γ_newborn, list of new parameters) — new params → opt.add_param_group (WITHOUT resetting
        the optimizer for the old weights; only gate/to_f necessarily get new objects due to the shape change).
        """
        H, L = self.n_head, self.n_scales
        dev = self.gate.weight.device
        dt = self.gate.weight.dtype

        if demote:                                  # (destructive) demote the old permanent scale to γ<1
            tau_old = self.tau0 * (self.rho ** (L - 1))
            self.gammas[L - 1] = math.exp(-1.0 / tau_old)
            self.cells[L - 1].gamma = self.gammas[L - 1]

        # newborn permanent scale (γ=1, empty W0, greedy plasticity)
        newcell = FastWeightCell(H, self.hd, gamma=1.0, chunk_size=self.chunk_size).to(dev, dt)
        newcell.set_beta_gain(birth_beta_gain)
        self.cells.append(newcell)
        self.gammas.append(1.0)

        # extend the gate by a scale. Output is (B,T,H,L) → row = h*L+l (L innermost) →
        # you CANNOT simply append; reweave into (H, L+1) and give the new scale a neutral 0.
        E = self.gate.in_features
        gw = self.gate.weight.data.view(H, L, E)
        gb = self.gate.bias.data.view(H, L)
        new_gate = nn.Linear(E, H * (L + 1), bias=True).to(dev, dt)
        nw = gw.new_zeros(H, L + 1, E); nw[:, :L] = gw
        nb = gb.new_zeros(H, L + 1); nb[:, :L] = gb
        new_gate.weight.data.copy_(nw.reshape(H * (L + 1), E))
        new_gate.bias.data.copy_(nb.reshape(H * (L + 1)))
        self.gate = new_gate
        new_params = list(newcell.parameters()) + list(new_gate.parameters())

        # selective forget-gate (if present): output (B,T,L,H) → row = l*H+h (scale-major)
        # → the new scale is simply APPENDed at the end; bias = logit(γ=1) (like the permanent init).
        if self.selective:
            Ef = self.to_f.in_features
            fw = self.to_f.weight.data
            fb = self.to_f.bias.data
            new_f = nn.Linear(Ef, (L + 1) * H, bias=True).to(dev, dt)
            nfw = fw.new_zeros((L + 1) * H, Ef); nfw[:L * H] = fw
            nfb = fb.new_zeros((L + 1) * H); nfb[:L * H] = fb
            gg = 1 - 1e-4
            nfb[L * H:] = math.log(gg / (1 - gg))
            new_f.weight.data.copy_(nfw)
            new_f.bias.data.copy_(nfb)
            self.to_f = new_f
            new_params += list(new_f.parameters())

        self.n_scales = L + 1
        return 1.0, new_params

    def _mode(self, cell) -> str:
        return "chunk"        # the decay-chunk kernel handles any γ (see cell.delta_chunk_decay)

    # ---- ComputeUnit API ----
    def init_state(self, batch_size, device, dtype=torch.float32) -> FractalState:
        return FractalState(W=[c.init_state(batch_size, device, dtype) for c in self.cells])

    def forward(self, x, state: FractalState | None = None, return_delta: bool = False):
        B, T, _ = x.shape
        xc = x.transpose(1, 2)
        xc = F.pad(xc, (self.conv_k - 1, 0))
        xc = self.conv(xc).transpose(1, 2)                             # (B,T,E)
        q, k, v, beta, _ = self._project(x, xc)                        # v = raw x; high-pass keys fresh per pass
        f_all = torch.sigmoid(self.to_f(xc)).view(B, T, self.n_scales, self.n_head) if self.selective else None

        Ws = state.W if state is not None else [None] * self.n_scales
        outs, new_W, dnorms = [], [], ([] if return_delta else None)
        for l, cell in enumerate(self.cells):
            W_l = Ws[l] if Ws[l] is not None else cell.start(B, x.device, x.dtype)
            f_l = f_all[:, :, l, :].transpose(1, 2) if f_all is not None else None   # (B,H,T)
            o_l, W_l, dn = cell.scan(q, k, v, beta, W_l, self._mode(cell), return_delta, f=f_l)
            outs.append(o_l)
            new_W.append(W_l)
            if return_delta:
                dnorms.append(dn)
        out = self.drop(self.proj(self._combine(outs, xc)))
        return out, FractalState(
            W=new_W,
            event_prev=None if state is None else state.event_prev,
            event_n=0 if state is None else state.event_n,
            event_sum=None if state is None else state.event_sum,
            event_count=0 if state is None else state.event_count,
        ), dnorms

    def step(self, x, state: FractalState):
        """Streaming: link conv to the stored context, update W of all levels (recurrent)."""
        B, T, E = x.shape
        conv_state = state.conv
        if conv_state is None:
            conv_state = x.new_zeros(B, self.conv_k - 1, E)
        x_full = torch.cat([conv_state, x], dim=1)                     # (B, k-1+T, E)
        xc = self.conv(x_full.transpose(1, 2)).transpose(1, 2)         # (B, T, E)
        new_conv = x_full[:, -(self.conv_k - 1):]
        hp = (state.hp_sum, state.hp_n) if state.hp_sum is not None else None
        q, k, v, beta, new_hp = self._project(x, xc, hp)               # v = raw x; high-pass keys (carry the average)
        f_all = torch.sigmoid(self.to_f(xc)).view(B, T, self.n_scales, self.n_head) if self.selective else None

        outs, new_W = [], []
        for l, cell in enumerate(self.cells):
            f_l = f_all[:, :, l, :].transpose(1, 2) if f_all is not None else None
            o_l, W_l, _ = cell.scan(q, k, v, beta, state.W[l], mode="recurrent", f=f_l)
            outs.append(o_l)
            new_W.append(W_l)
        out = self.drop(self.proj(self._combine(outs, xc)))
        hp_sum, hp_n = new_hp if new_hp is not None else (None, 0)
        return out, FractalState(
            W=new_W, conv=new_conv, hp_sum=hp_sum, hp_n=hp_n,
            event_prev=state.event_prev, event_n=state.event_n,
            event_sum=state.event_sum, event_count=state.event_count,
        )
