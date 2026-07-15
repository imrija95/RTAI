"""Synthetic associative recall task (MQAR-lite).

The model is given key→value pairs and then queries; it must recall the correct value.
This teaches the SRWM layer to store associations into the fast matrix W and read them back
— exactly the capability later used at deploy time for PERSISTENT learning (learn in one
session, recall in the next).

Sequence layout (teacher-forced):
    k1 v1 k2 v2 ... kP vP  |  QUERY q1 a1  QUERY q2 a2 ...
We only score the positions of the queried keys q_j (predicting their value a_j).

Keys and values live in separate token ranges → the role is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecallVocab:
    n_keys: int = 32
    n_vals: int = 32

    @property
    def KEY0(self) -> int:
        return 0

    @property
    def VAL0(self) -> int:
        return self.n_keys

    @property
    def QUERY(self) -> int:
        return self.n_keys + self.n_vals

    @property
    def PAD(self) -> int:
        return self.n_keys + self.n_vals + 1

    @property
    def size(self) -> int:
        return self.n_keys + self.n_vals + 2

    def key_tok(self, i: int) -> int:
        return self.KEY0 + i

    def val_tok(self, i: int) -> int:
        return self.VAL0 + i

    def val_index(self, tok: int) -> int:
        """From a value token back to a value index (0..n_vals-1)."""
        return tok - self.VAL0


def make_recall_batch(batch: int, n_pairs: int, n_queries: int,
                      vocab: RecallVocab, device, generator=None):
    """Return (x, y): x=(B,T) tokens, y=(B,T) targets (-1 = not scored)."""
    T = 2 * n_pairs + 3 * n_queries
    x = torch.full((batch, T), vocab.PAD, dtype=torch.long)
    y = torch.full((batch, T), -1, dtype=torch.long)

    def randint(high):
        return int(torch.randint(high, (1,), generator=generator).item())

    for b in range(batch):
        # P distinct keys, each with a random value
        keys = torch.randperm(vocab.n_keys, generator=generator)[:n_pairs].tolist()
        vals = [randint(vocab.n_vals) for _ in range(n_pairs)]
        pos = 0
        for ki, vi in zip(keys, vals):
            x[b, pos] = vocab.key_tok(ki); pos += 1
            x[b, pos] = vocab.val_tok(vi); pos += 1
        # Q queries on random keys from that set
        for _ in range(n_queries):
            j = randint(n_pairs)
            x[b, pos] = vocab.QUERY; pos += 1
            x[b, pos] = vocab.key_tok(keys[j])
            # target: at the queried key's position, predict its value
            y[b, pos] = vocab.val_tok(vals[j]); pos += 1
            x[b, pos] = vocab.val_tok(vals[j]); pos += 1  # teacher forcing
    return x.to(device), y.to(device)


def encode_teach(pairs: list[tuple[int, int]], vocab: RecallVocab, device):
    """'Learn' sequence: k1 v1 k2 v2 ... (no queries). (1,T) tokens."""
    toks = []
    for ki, vi in pairs:
        toks += [vocab.key_tok(ki), vocab.val_tok(vi)]
    return torch.tensor([toks], dtype=torch.long, device=device)


def encode_query(key: int, vocab: RecallVocab, device):
    """Query sequence: QUERY key. Take the prediction from the logits at the last position. (1,2)."""
    return torch.tensor([[vocab.QUERY, vocab.key_tok(key)]],
                        dtype=torch.long, device=device)
