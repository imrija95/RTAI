"""Train a fresh byte-level BPE tokenizer on a SAMPLE of the quality mix — SAFE / non-destructive.

Writes a NEW file (default fractal_tokenizer_32k.json); it never touches fractal_tokenizer.json.
A representative sample of the mix (a few hundred MB of text) is enough to learn a good vocab; a
larger vocab (32k) compresses educational text and especially code far better than the 16k tokenizer
(which was trained on simple stories), so more real content fits per token.

Run:  uv run python -m fractal.train_tokenizer --vocab_size 32000 --out fractal_tokenizer_32k.json
"""

from __future__ import annotations

import argparse
import os

from fractal import tokenizer as tk
from fractal.data_mix import MIX, Source


def sample_iter(sources, max_chars):
    """Yield documents from the mix sources, weight-interleaved, up to a character budget."""
    total_w = sum(s.weight for s in sources)
    targets = {s.name: s.weight / total_w * max_chars for s in sources}
    seen = {s.name: 0 for s in sources}
    total = 0
    while total < max_chars and not all(s.exhausted for s in sources):
        live = [s for s in sources if not s.exhausted]
        s = max(live, key=lambda s: (targets[s.name] - seen[s.name]) / max(targets[s.name], 1.0))
        t = s.next_text()
        if t is None:
            continue
        seen[s.name] += len(t)
        total += len(t)
        if total % 20_000_000 < 8192:
            print(f"  …{total:,} chars sampled", flush=True)
        yield t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--out", default="fractal_tokenizer_32k.json")
    ap.add_argument("--max_chars", type=int, default=120_000_000)   # ~120 MB sample → good 32k vocab
    a = ap.parse_args()
    sources = [Source(n, c, k, w) for n, c, k, w in MIX]
    print(f"training {a.vocab_size} byte-level BPE on ~{a.max_chars:,} chars of the mix → {a.out}",
          flush=True)
    tk.train_bpe(sample_iter(sources, a.max_chars), a.vocab_size, a.out)
    print(f"done → {a.out}", flush=True)
    os._exit(0)      # skip datasets' streaming-thread teardown race at exit
