"""FineWeb-Edu → fractal_data_web/ using the EXISTING 16k tokenizer — SAFE.

Unlike data_web.py, this does NOT train a new tokenizer and does NOT overwrite
fractal_tokenizer.json or fractal_data/ (TinyStories). It reuses the existing 16k tokenizer
and writes to a SEPARATE fractal_data_web/ directory → train via `--data_dir fractal_data_web`.
(Analogous to data_chat_ft.py.)

Run:  uv run python -m fractal.data_web_ft --max_tokens 100000000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset

from fractal import tokenizer as tk

OUT_DIR = "fractal_data_web"
TOK_PATH = "fractal_tokenizer.json"        # EXISTING 16k — read only
EOT = "<|endoftext|>"


def prepare(dataset, config, max_tokens, retries=12):
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = tk.load(TOK_PATH)
    eot = tok.token_to_id(EOT)
    assert eot is not None, "16k tokenizer must have <|endoftext|>"
    print(f"tokenizer: {TOK_PATH} (vocab {tok.get_vocab_size()}) | source: {dataset}/{config}", flush=True)

    val_n = max(50_000, max_tokens // 100)
    need = max_tokens + val_n
    ids: list[int] = []
    docs_seen = 0                                    # for reconnect+skip (avoid writing duplicate data)
    attempt = 0
    while len(ids) < need and attempt <= retries:
        try:
            ds = (load_dataset(dataset, config, split="train", streaming=True) if config
                  else load_dataset(dataset, split="train", streaming=True))
            if docs_seen:
                ds = ds.skip(docs_seen)              # resume where the stream dropped (no duplicate output)
            for ex in ds:
                docs_seen += 1
                t = (ex.get("text") or ex.get("content") or "").strip()
                if not t:
                    continue
                ids.extend(tok.encode(" " + t).ids)
                ids.append(eot)
                if len(ids) % 5_000_000 < 4096:
                    print(f"  …{len(ids):,} tokens", flush=True)
                if len(ids) >= need:
                    break
            break                                    # stream finished without error
        except Exception as e:                       # stream dropout (Starlink) → reconnect + skip
            attempt += 1
            print(f"  [dropout {attempt}/{retries}: {type(e).__name__}] have {len(ids):,} tok, reconnect+skip {docs_seen}", flush=True)

    if len(ids) < 2 * val_n:
        raise SystemExit(f"too little data ({len(ids):,} tok) — try again")
    arr = np.array(ids[:need] if len(ids) >= need else ids, dtype=np.uint16)
    vn = min(val_n, len(arr) // 20)
    arr[:vn].tofile(f"{OUT_DIR}/val.bin")
    arr[vn:].tofile(f"{OUT_DIR}/train.bin")
    print(f"{OUT_DIR}/val.bin: {vn:,} | {OUT_DIR}/train.bin: {len(arr)-vn:,} tokens", flush=True)
    print("done. Training: --data_dir fractal_data_web", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--max_tokens", type=int, default=100_000_000)
    prepare(**vars(ap.parse_args()))
    os._exit(0)      # sidesteps the datasets streaming-thread race at finalization
