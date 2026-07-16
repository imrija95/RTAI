"""Train a fresh byte-level BPE tokenizer on a sample of a selected corpus recipe.

The output must not exist. Natural Cortex uses a 24k vocabulary and atomic chat/tool/teaching
markers; older recipes remain available for archived experiments.

Run: uv run python -m fractal.train_tokenizer --recipe natural --vocab_size 24000 \
         --out natural_tokenizer_24k.json
"""

from __future__ import annotations

import argparse
import os

from fractal import tokenizer as tk
from fractal.data_mix import RECIPES, Source


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
    ap.add_argument("--vocab_size", type=int, default=24000)
    ap.add_argument("--out", default="natural_tokenizer_24k.json")
    ap.add_argument("--max_chars", type=int, default=120_000_000)
    ap.add_argument("--recipe", default="natural", choices=["natural", *sorted(RECIPES)])
    a = ap.parse_args()
    if a.recipe == "natural":
        from fractal.natural_data import train_tokenizer
        train_tokenizer(a.out, a.vocab_size, a.max_chars)
        print(f"done → {a.out}", flush=True)
        raise SystemExit(0)
    sources = [Source(n, c, k, w) for n, c, k, w in RECIPES[a.recipe]]
    print(f"training {a.vocab_size} byte-level BPE on ~{a.max_chars:,} chars of the mix → {a.out}",
          flush=True)
    tk.train_bpe(sample_iter(sources, a.max_chars), a.vocab_size, a.out)
    print(f"done → {a.out}", flush=True)
    os._exit(0)      # skip datasets' streaming-thread teardown race at exit
