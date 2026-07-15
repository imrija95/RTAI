"""Synthetic recall episodes in natural language — teach the model to store a fact into W and
recall it from a distance.

Episode: a few facts ("Tom's favorite color is blue.") → long filler (sentences from
TinyStories, creating distance) → query ("Tom's favorite color is") and we score the value
prediction. Because the model has no global attention (convolution only k=4), the info for the
query can flow ONLY through the fast weights W → this teaches W to become long-term memory.

Mixed with plain TinyStories windows (multi-task: recall + fluency).
"""

from __future__ import annotations

import os
import numpy as np
import torch

NAMES = ["Tom", "Lily", "Max", "Anna", "Ben", "Mia", "Sam", "Emma", "Leo", "Zoe",
         "Jack", "Rose", "Tim", "Sara", "Nick", "Ella"]
ATTRS = {
    "color": ["red", "blue", "green", "yellow", "pink", "black", "white", "brown", "purple", "orange"],
    "animal": ["dog", "cat", "fox", "bird", "fish", "frog", "bear", "duck", "cow", "pig"],
    "toy": ["ball", "doll", "kite", "drum", "car", "train", "robot", "teddy", "puzzle", "top"],
    "food": ["cake", "apple", "bread", "soup", "pie", "candy", "cheese", "rice", "egg", "jam"],
}
ATTR_KEYS = list(ATTRS)


def _rng_choice(seq, g):
    return seq[int(torch.randint(len(seq), (1,), generator=g).item())]


class RecallGen:
    """Generator of recall episodes + mixed batcher (recall + TinyStories)."""

    def __init__(self, tok, data_dir="data", split="train"):
        self.tok = tok
        self.eot = tok.token_to_id("<|endoftext|>")
        self.data = np.memmap(os.path.join(data_dir, f"{split}.bin"),
                              dtype=np.uint16, mode="r")

    def _enc(self, s):
        return self.tok.encode(s).ids

    def _filler(self, n):
        i = int(torch.randint(len(self.data) - n - 1, (1,)).item())
        return [int(t) for t in self.data[i:i + n]]

    def episode(self, block_size, n_facts=3, n_queries=2, g=None):
        """Return (toks, score) of length block_size+1. score[i]=True → score the prediction toks[i+1]."""
        # pick n_facts distinct names, each with an (attribute, value)
        names = list(NAMES)
        chosen = []
        for _ in range(n_facts):
            nm = names.pop(int(torch.randint(len(names), (1,), generator=g).item()))
            attr = _rng_choice(ATTR_KEYS, g)
            val = _rng_choice(ATTRS[attr], g)
            chosen.append((nm, attr, val))

        toks, score = [], []
        for nm, attr, val in chosen:                      # facts (not scored)
            for t in self._enc(f"{nm}'s favorite {attr} is {val}. "):
                toks.append(t); score.append(False)

        # queries (assembled ahead so we know the length)
        q_blocks = []
        qs = [chosen[int(torch.randint(len(chosen), (1,), generator=g).item())]
              for _ in range(n_queries)]
        for nm, attr, val in qs:
            q_ids = self._enc(f"{nm}'s favorite {attr} is")
            v_ids = self._enc(f" {val}")
            q_blocks.append((q_ids, v_ids))
        q_len = sum(len(q) + len(v) + 2 for q, v in q_blocks)   # +2 reserve for ". "

        filler_len = max(16, (block_size + 1) - len(toks) - q_len)
        toks += self._filler(filler_len)
        score += [False] * filler_len

        for q_ids, v_ids in q_blocks:                     # query + scored value
            for t in q_ids:
                toks.append(t); score.append(False)
            score[-1] = True                              # last query token → 1st value token
            for k, t in enumerate(v_ids):
                toks.append(t); score.append(k < len(v_ids) - 1)
            for t in self._enc(". "):
                toks.append(t); score.append(False)

        toks = toks[:block_size + 1]
        score = score[:block_size + 1]
        if len(toks) < block_size + 1:                    # pad with eot (not scored)
            pad = block_size + 1 - len(toks)
            toks += [self.eot] * pad; score += [False] * pad
        return toks, score

    def _story_window(self, block_size):
        i = int(torch.randint(len(self.data) - block_size - 1, (1,)).item())
        return [int(t) for t in self.data[i:i + block_size + 1]]

    def batch(self, block_size, batch_size, recall_ratio, device, g=None):
        xs, ys = [], []
        for _ in range(batch_size):
            if torch.rand(1, generator=g).item() < recall_ratio:
                toks, score = self.episode(block_size, g=g)
                y = [toks[i + 1] if score[i] else -1 for i in range(block_size)]
            else:
                toks = self._story_window(block_size)
                y = toks[1:block_size + 1]                # fluency: score everything
            xs.append(toks[:block_size]); ys.append(y)
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return x, y

    @torch.no_grad()
    def recall_accuracy(self, model, block_size, device, trials=64, g=None):
        """Fraction of correctly recalled values on held-out recall episodes (start from W0)."""
        model.eval()
        xs, ys = [], []
        for _ in range(trials):
            toks, score = self.episode(block_size, g=g)
            xs.append(toks[:block_size])
            ys.append([toks[i + 1] if score[i] else -1 for i in range(block_size)])
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        logits, _, _, _ = model(x, states=None)
        mask = y != -1
        pred = logits.argmax(dim=-1)
        return (pred[mask] == y[mask]).float().mean().item()
