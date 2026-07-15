"""Recall curriculum: MULTI-FACT associative recall (MQAR-style, copying from context).

Lessons from failures:
  1) closed value pool → the model memorizes instead of copying → pool = the whole vocabulary,
  2) value = learned projection → does not decode unseen tokens → value = RAW state
     (token identity, see unit._project),
  3) 1 fact per episode → the model can't tell apart multiple stored facts → MORE facts per episode,
     same template, different names → the only discriminator is the name → forces keying on the key.

Episode: "{A} ... is {vA}. {B} ... is {vB}. ..." → filler → "{X} ... is" → v_X.
Test on HELD-OUT values (never trained) = general recall, not memorization.
"""

from __future__ import annotations

import random

import numpy as np
import torch

NAMES = ["Tom", "Lily", "Ben", "Anna", "Max", "Mia", "Sam", "Ella", "Leo", "Nina",
         "Kate", "Jack", "Rosa", "Finn", "Zoe", "Ivan", "Lucy", "Milo", "Nora", "Otto"]

PREFIXES = [
    " {n}'s favorite color is",
    " {n}'s favorite thing is",
    " {n}'s favorite animal is",
    " {n} likes",
    " {n} has a",
]


class RecallGen:
    def __init__(self, tok, val_bin: str = "fractal_data/val.bin", seed: int = 0, n_names: int = 0):
        self.tok = tok
        self.data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        # value pool = single-token lowercase words (thousands), split train/held-out;
        # at the same time collect single-token words with a CAPITAL letter → name (key) candidates
        pool, caps = [], []
        for i in range(tok.get_vocab_size()):
            s = tok.decode([i])
            if len(s) > 3 and s[0] == " " and s[1:].isalpha():
                if s[1:].islower():
                    pool.append(i)
                elif s[1].isupper():
                    caps.append(s.strip())
        random.Random(seed).shuffle(pool)
        cut = int(len(pool) * 0.85)
        self.train_vals, self.test_vals = pool[:cut], pool[cut:]
        if n_names > 0:
            # LARGE name (key) pool: the model can't memorize them → it must handle
            # GENERAL key separation under capacity pressure (that's the goal of neurogenesis)
            random.Random(seed + 1).shuffle(caps)
            self.names = caps[:n_names]
        else:
            # only 6 single-token names (easy — clean baseline metric)
            self.names = [nm for nm in NAMES if len(tok.encode(" " + nm).ids) == 1]
        print(f"[RecallGen] value pool: {len(pool)} tokens "
              f"(train {len(self.train_vals)} / held-out {len(self.test_vals)}) | names: {len(self.names)}"
              f"{' (large pool, no memorization)' if n_names > 0 else ''}")

    def _e(self, s: str):
        return self.tok.encode(s).ids

    def _filler(self, n: int):
        if n <= 0:
            return []
        i = random.randint(0, len(self.data) - n - 1)
        return [int(t) for t in self.data[i:i + n]]

    def _episode(self, seq_len: int, n_facts: int, held_out: bool):
        """Returns (seq, ans_tok, A) — A is the index of the answer in seq (seq[A] == ans)."""
        tpl = random.choice(PREFIXES)
        names = random.sample(self.names, min(n_facts, len(self.names)))
        pool = self.test_vals if held_out else self.train_vals
        vals = [random.choice(pool) for _ in names]
        facts = []
        for nm, vt in zip(names, vals):
            facts += self._e(tpl.format(n=nm)) + [vt]
        j = random.randrange(len(names))                       # which fact we ask about
        query = self._e(tpl.format(n=names[j]))
        ans = vals[j]
        F, Q = len(facts), len(query)
        D = random.randint(1, max(1, seq_len - F - Q - 1))
        seq = facts + self._filler(D) + query + [ans]
        return seq, ans, F, D, Q, F + D + Q

    def batch(self, batch_size: int, seq_len: int, device, w_ans: float = 5.0, max_facts: int = 4):
        """Multi-fact recall batch, episode ≤ seq_len. (x, y, w); w weights the answer (w_ans), facts/query=1, filler/pad=0."""
        rows, wts = [], []
        for _ in range(batch_size):
            M = random.randint(1, min(max_facts, len(self.names)))     # cap by the ACTUAL name pool
            seq, _, F, D, Q, A = self._episode(seq_len, M, held_out=False)
            seq = (seq + [0] * (seq_len + 1))[:seq_len + 1]
            w = [0.0] * seq_len
            if 0 <= A - 1 < seq_len:
                w[A - 1] = w_ans          # train ONLY the answer (pure recall); facts are only READ into W, fluency comes from the story batch
            rows.append(seq); wts.append(w)
        t = torch.tensor(rows, dtype=torch.long, device=device)
        wt = torch.tensor(wts, dtype=torch.float32, device=device)
        return t[:, :-1], t[:, 1:], wt

    @torch.no_grad()
    def accuracy(self, model, distance: int, device, n: int = 32, held_out: bool = True, n_facts: int = 1) -> float:
        """Recall accuracy: n_facts facts, query for one, across `distance` filler. held_out=True → unseen values."""
        was_training = model.training
        model.eval()
        ok = 0
        for _ in range(n):
            tpl = random.choice(PREFIXES)
            names = random.sample(self.names, min(n_facts, len(self.names)))
            pool = self.test_vals if held_out else self.train_vals
            vals = [random.choice(pool) for _ in names]
            facts = []
            for nm, vt in zip(names, vals):
                facts += self._e(tpl.format(n=nm)) + [vt]
            j = random.randrange(len(names))
            prompt = facts + self._filler(distance) + self._e(tpl.format(n=names[j]))
            idx = torch.tensor([prompt], dtype=torch.long, device=device)
            logits, _, _, _ = model(idx)                    # forward (chunk) — fast, equivalent to streaming
            ok += int(logits[0, -1].argmax().item() == vals[j])
        model.train(was_training)
        return ok / n
