"""The atom of the fractal — a fast-weight cell with the delta rule.

A single "fast weight" W (per head, per sample) modifies itself DURING the forward pass:

    v̂ = W k                 # recall what you hold for this key
    W ← γ·W + β (v − v̂) kᵀ    # correct yourself by the difference  ← SELF-MODIFICATION
    o = W q                 # read (after the write)

This is at the same time one step of online learning (LMS, minimizes ‖v − W k‖²) — so the
model literally runs an optimizer inside the forward pass.

γ (decay) controls how fast the memory forgets: small γ → a short-term scratchpad,
γ→1 → permanent memory. The fractal stacks several cells with different γ on a geometric ladder.

Two numerically EQUIVALENT paths (see tests/test_equiv.py):
  - delta_recurrent   — a readable step-by-step loop; any γ; can log ‖ΔW‖,
  - delta_chunk_decay — parallel block-wise delta-rule (gated DeltaNet), ~8× faster,
                        works for any γ∈(0,1] (delta_chunk is its γ=1 special case).

Scan inputs: q,k,v (B,H,T,hd), beta (B,H,T), W (B,H,hd,hd). q,k are normalized outside.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def delta_recurrent(q, k, v, beta, W, gamma: float = 1.0, return_delta: bool = False, f=None,
                    eligibility=None, eligibility_decay: float | None = None):
    """Step by step, W = decay·W + β(v − Wk)kᵀ, read after write.
    decay = scalar γ, OR (selective mode) a per-step forget gate f (B,H,T) — learned dynamics.
    Returns (out (B,H,T,hd), W_final, dnorms|None)."""
    T = q.shape[2]
    outs = []
    dnorms = [] if return_delta else None
    track_eligibility = eligibility_decay is not None
    if track_eligibility and eligibility is None:
        eligibility = torch.zeros_like(W)
    for t in range(T):
        k_t, v_t, q_t = k[:, :, t], v[:, :, t], q[:, :, t]        # (B,H,hd)
        b_t = beta[:, :, t].unsqueeze(-1)                          # (B,H,1)
        v_hat = torch.einsum("bhij,bhj->bhi", W, k_t)              # W_{t-1} k_t
        dW = torch.einsum("bhi,bhj->bhij", b_t * (v_t - v_hat), k_t)
        decay = f[:, :, t, None, None] if f is not None else gamma  # (B,H,1,1) or scalar
        W = decay * W + dW                                         # SELF-MODIFICATION
        if track_eligibility:
            eligibility = eligibility_decay * eligibility + dW
        outs.append(torch.einsum("bhij,bhj->bhi", W, q_t))        # read after write
        if return_delta:
            dnorms.append(dW.flatten(1).norm(dim=1))              # (B,)
    dn = torch.stack(dnorms, dim=1) if return_delta else None      # (B,T)
    result = (torch.stack(outs, dim=2), W, dn)                     # (B,H,T,hd)
    return (*result, eligibility) if track_eligibility else result


def delta_chunk(q, k, v, beta, W, chunk_size: int, gamma: float = 1.0):
    """Parallel block-wise delta-rule (DeltaNet). The recurrence is sequential only ACROSS
    blocks; within a block these are batched matmuls → markedly faster training.

    Within a block, the intra-block dependencies are solved by a single lower-triangular
    system:  (I + A) U = β (V − S·Kᵀ),  out = Q·Sᵀ + tril(Q·Kᵀ)·U,  S ← S + Uᵀ·K.
    """
    assert gamma == 1.0, "the chunk path currently supports only γ=1 (use recurrent for the decay ladder)"
    B, H, T, hd = q.shape
    C = chunk_size
    S = W                                                          # (B,H,hd,hd): v = S·k
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    outs = []
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        Kc, Vc, Qc = k[:, :, s:e], v[:, :, s:e], q[:, :, s:e]      # (B,H,c,hd)
        Bc = beta[:, :, s:e]                                       # (B,H,c)
        A = torch.tril(Bc.unsqueeze(-1) * (Kc @ Kc.transpose(-1, -2)), diagonal=-1)
        M = A + eye[:c, :c]                                        # unit-lower-triangular
        rhs = Bc.unsqueeze(-1) * (Vc - Kc @ S.transpose(-1, -2))   # (B,H,c,hd)
        U = torch.linalg.solve_triangular(M, rhs, upper=False, unitriangular=True)
        QKt = torch.tril(Qc @ Kc.transpose(-1, -2), diagonal=0)    # j ≤ t (read after write)
        outs.append(Qc @ S.transpose(-1, -2) + QKt @ U)           # (B,H,c,hd)
        S = S + U.transpose(-1, -2) @ Kc                           # advance the state to the end of the block
    return torch.cat(outs, dim=2), S                               # (B,H,T,hd), (B,H,hd,hd)


def delta_chunk_decay(q, k, v, beta, W, chunk_size: int, gamma: float = 1.0):
    """Chunk-parallel delta-rule WITH DECAY γ<1 (gated DeltaNet). A generalization of delta_chunk.

    The inter-position decay γ^{i−j} factors into γ^i·γ^{−j} and is rescaled away by the substitution
    ũ_i = γ^{−i} u_i → the system has the same structure as γ=1, just with powers of γ:
      M   = I + tril((β/γ) ⊙ K Kᵀ, −1)
      R   = β ⊙ (γ^{−i} ⊙ V − K Sᵀ)          Ũ = M⁻¹ R  (triangular solve)
      out = γ^i ⊙ (γ·Q Sᵀ + tril(Q Kᵀ,0)·Ũ)
      S'  = γ^c·S + γ^{c−1}·(Ũᵀ K)            ← γ^c: boundary decay (needed across blocks!)
    Numerically identical to delta_recurrent (verified in tests/test_equiv.py). γ=1 → delta_chunk.
    """
    B, H, T, hd = q.shape
    C = chunk_size
    S = W                                                          # (B,H,hd,hd)
    sd = torch.float32                                             # always solve in fp32
    outs = []
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        if gamma < 1.0:                                            # γ^{−(c−1)} must not overflow
            assert (c - 1) * (-math.log(gamma)) <= 30.0, \
                "chunk_size too large relative to τ (γ^-c overflows) — reduce chunk_size"
        Kc, Vc, Qc = k[:, :, s:e], v[:, :, s:e], q[:, :, s:e]      # (B,H,c,hd)
        Bc = beta[:, :, s:e]                                       # (B,H,c)
        idx = torch.arange(c, device=q.device, dtype=q.dtype)
        lam = (gamma ** idx).view(1, 1, c, 1)                      # γ^i  (≤1)
        linv = (gamma ** (-idx)).view(1, 1, c, 1)                  # γ^-i (amplification)

        A = torch.tril((Bc / gamma).unsqueeze(-1) * (Kc @ Kc.transpose(-1, -2)), diagonal=-1)
        M = (A + torch.eye(c, device=q.device, dtype=q.dtype)).to(sd)   # unit-lower-tri
        R = Bc.unsqueeze(-1) * (linv * Vc - Kc @ S.transpose(-1, -2))    # (B,H,c,hd)
        U = torch.linalg.solve_triangular(M, R.to(sd), upper=False, unitriangular=True).to(q.dtype)

        QKt = torch.tril(Qc @ Kc.transpose(-1, -2), diagonal=0)    # j ≤ t (read after write)
        outs.append(lam * (gamma * (Qc @ S.transpose(-1, -2)) + QKt @ U))   # (B,H,c,hd)
        S = (gamma ** c) * S + (gamma ** (c - 1)) * (U.transpose(-1, -2) @ Kc)
    return torch.cat(outs, dim=2), S                               # (B,H,T,hd), (B,H,hd,hd)


class FastWeightCell(nn.Module):
    """A single fractal scale: a fast weight W with a learned initial ("birth") W0 and
    a fixed decay γ. It does NOT own the projections (q,k,v,β) — it receives them from outside and shares
    them across scales (that is the self-similarity). Supports both parallel and streaming runs."""

    def __init__(self, n_head: int, head_dim: int, gamma: float = 1.0, chunk_size: int = 64):
        super().__init__()
        self.n_head = n_head
        self.hd = head_dim
        self.gamma = gamma
        self.chunk_size = chunk_size
        # learned initial fast weight (the start during training; gradient flows into it)
        self.W0 = nn.Parameter(torch.zeros(n_head, head_dim, head_dim))
        # plasticity (neurogenesis): a newborn scale starts "greedy" (beta_gain>1) and
        # gradually matures toward 1.0. Driven by a growth rule OUTSIDE the gradient → a buffer, not
        # a parameter. persistent=False: does not add a key to state_dict → old checkpoints
        # load unchanged; in a fully trained model it is ~1.0 anyway (a mature scale).
        self.register_buffer("beta_gain", torch.tensor(1.0), persistent=False)
        # compile-safe Python mirror of beta_gain. scan() reads THIS, not the buffer: float(tensor)
        # in the hot path calls Tensor.item(), which is a torch.compile graph break — and because
        # scan() runs inside the per-scale loop, Dynamo reports "graph break in loop" and falls the
        # WHOLE unit forward back to eager (no fusion). Kept in sync via set_beta_gain().
        self._beta_gain_f = 1.0

    def set_beta_gain(self, gain: float):
        """Set plasticity gain (neurogenesis), keeping the buffer and its compile-safe Python mirror
        in sync. The mirror is what the hot path reads (see __init__); the buffer stays for any tensor
        reader (e.g. neurogenesis maturation)."""
        g = float(gain)
        self.beta_gain = torch.tensor(g, device=self.beta_gain.device)
        self._beta_gain_f = g

    def init_state(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return self.W0.detach().to(device=device, dtype=dtype)[None].repeat(batch_size, 1, 1, 1)

    def start(self, batch_size: int, device, dtype) -> torch.Tensor:
        """W for training, derived from W0 so that gradient flows into it."""
        return self.W0[None].expand(batch_size, self.n_head, self.hd, self.hd).contiguous().to(dtype)

    def scan(self, q, k, v, beta, W, mode: str = "chunk", return_delta: bool = False, f=None):
        """Full pass over time. f=None → fixed γ, chunk fast. f (B,H,T) → SELECTIVE
        (learned per-step dynamics) → recurrent (the chunk kernel cannot do per-step decay yet)."""
        bg = self._beta_gain_f                                # Python mirror → no .item() / graph break
        if bg != 1.0:                                        # neurogenesis: the newborn writes more strongly
            # β_eff = 1 − (1−β)^gain — monotonic, bounded in (0,1) (gain>1 → closer to a full write,
            # gain<1 → dampens); stable for the delta rule (β_eff never exceeds 1 → no overshoot).
            # the float exponent preserves the dtype of β (bf16 autocast does not promote the kernel to fp32).
            beta = 1.0 - (1.0 - beta).clamp(min=1e-6).pow(bg)
        if f is None and mode == "chunk" and not return_delta:
            out, W = delta_chunk_decay(q, k, v, beta, W, self.chunk_size, self.gamma)
            return out, W, None
        return delta_recurrent(q, k, v, beta, W, self.gamma, return_delta, f=f)

    def scan_eligibility(self, q, k, v, beta, W, eligibility, eligibility_decay: float,
                         return_delta: bool = False, f=None):
        """Recurrent event-algebra path with a chunking-invariant delayed-credit trace."""
        bg = self._beta_gain_f
        if bg != 1.0:
            beta = 1.0 - (1.0 - beta).clamp(min=1e-6).pow(bg)
        return delta_recurrent(q, k, v, beta, W, self.gamma, return_delta, f=f,
                               eligibility=eligibility, eligibility_decay=eligibility_decay)
