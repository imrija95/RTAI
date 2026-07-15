"""Reusable mechanisms for the small efficiency tournament.

The experiment runner lives separately from the production trainer, but these primitives are kept
small and testable: verified unified-format tool episodes and the sparse local-credit update. No
benchmark result is hard-coded here.
"""

from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F

from fractal import chat_format as cf


def weighted_ce(logits, targets, weight=None):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                         reduction="none")
    if weight is None:
        return ce.mean()
    w = weight.reshape(-1).to(ce.dtype)
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def local_credit_loss(model, x, y, weight, selected_depth: int):
    """Train one shared recurrence from a local target; preceding recurrences run without a graph.

    Because the block is weight-tied, rotating selected_depth still updates the same generating
    rule at every position. The shared embedding receives a gradient only when depth zero is
    selected; the shared output head always receives one.
    """
    h = model.drop(model.tok_emb(x))
    for d in range(selected_depth):
        with torch.no_grad():
            h, _, _ = model.block_at(d)(h, None, False)
    if selected_depth:
        h = h.detach()
    h, _, _ = model.block_at(selected_depth)(h, None, False)
    logits = model.head(model.ln_f(h))
    return weighted_ce(logits, y, weight)


def _encode_weighted(tok, pieces):
    """Encode (text, loss-weight) pieces without retokenizing or modifying the tokenizer."""
    ids, weights, first = [], [], True
    for text, weight in pieces:
        enc = tok.encode((" " + text) if first else text).ids
        first = False
        ids.extend(enc)
        weights.extend([float(weight)] * len(enc))
    return ids, weights


class VerifiedToolGen:
    """Execution-verified calc/echo/time trajectories in the production chat protocol."""

    def __init__(self, tok, seed: int = 0):
        self.tok = tok
        self.rng = random.Random(seed)

    def episode(self):
        kind = self.rng.choices(("calc", "echo", "time"), weights=(5, 3, 2), k=1)[0]
        if kind == "calc":
            a, b = self.rng.randint(0, 99), self.rng.randint(1, 30)
            op = self.rng.choice(("+", "-", "*"))
            user = f"calculate {a} {op} {b}"
            args = {"expression": f"{a} {op} {b}"}
        elif kind == "echo":
            text = " ".join(self.rng.sample(
                ("amber", "river", "quiet", "stone", "forest", "signal", "orbit", "copper"), 3))
            user, args = f"echo {text}", {"text": text}
        else:
            user, args = self.rng.choice(("what time is it", "give the current time")), {}
        payload = json.dumps({"name": kind, "arguments": args}, separators=(",", ":"))
        pieces = cf.render_pieces([("user", user), ("tool_call", payload)])
        # The compiler values the typed action more than ordinary assistant prose. Structural
        # context remains zero and <|end|> stays at the normal weight.
        weighted = []
        for text, trainable in pieces:
            w = 4.0 if trainable and cf.TOOL_CALL in text else (1.0 if trainable else 0.0)
            weighted.append((text, w))
        return user, kind, args, weighted

    def batch(self, batch_size, seq_len, device):
        rows, weights = [], []
        for _ in range(batch_size):
            _, _, _, pieces = self.episode()
            ids, w = _encode_weighted(self.tok, pieces)
            ids = (ids + [0] * (seq_len + 1))[:seq_len + 1]
            # Token mask is aligned with input tokens; next-token loss consumes mask[1:].
            w = (w + [0.0] * (seq_len + 1))[:seq_len + 1]
            rows.append(ids)
            weights.append(w[1:])
        t = torch.tensor(rows, dtype=torch.long, device=device)
        wt = torch.tensor(weights, dtype=torch.float32, device=device)
        return t[:, :-1], t[:, 1:], wt
