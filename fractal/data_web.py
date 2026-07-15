"""Large general corpus for scaling (FineWeb-Edu / C4 …) via datasets streaming + larger BPE.

Writes to the standard paths (fractal_data/{train,val}.bin + fractal_tokenizer.json) → train.py
works unchanged (only --vocab_size must match this, default 16000).

Test (quick):  uv run python -m fractal.data_web --max_tokens 2000000 --bpe_docs 3000
Full:          uv run python -m fractal.data_web --max_tokens 300000000
Other source:  uv run python -m fractal.data_web --dataset allenai/c4 --config en
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DATA_DIR = "fractal_data"
TOK_PATH = "fractal_tokenizer.json"
EOT = "<|endoftext|>"


def _stream(dataset, config):
    """Stream of texts from an HF dataset (streaming → does not download the whole thing)."""
    ds = (load_dataset(dataset, config, split="train", streaming=True) if config
          else load_dataset(dataset, split="train", streaming=True))
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if t and t.strip():
            yield t.strip()


def prepare(dataset, config, vocab_size, max_tokens, bpe_docs):
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"BPE ({vocab_size}) on {bpe_docs} documents from {dataset}…", flush=True)
    def bpe_iter():
        for i, t in enumerate(_stream(dataset, config)):
            if i >= bpe_docs:
                break
            yield t
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(bpe_iter(), trainer=trainer)
    tok.save(TOK_PATH)
    print(f"vocab_size = {tok.get_vocab_size()}", flush=True)

    eot = tok.token_to_id(EOT)
    val_n = max(50_000, max_tokens // 100)          # ~1 % for validation
    need = max_tokens + val_n
    ids = []
    for t in _stream(dataset, config):
        ids.extend(tok.encode(" " + t).ids)
        ids.append(eot)
        if len(ids) >= need:
            break
        if len(ids) % 5_000_000 < 4096:
            print(f"  …{len(ids):,} tokens", flush=True)
    arr = np.array(ids[:need], dtype=np.uint16)
    arr[:val_n].tofile(f"{DATA_DIR}/val.bin")
    arr[val_n:].tofile(f"{DATA_DIR}/train.bin")
    print(f"{DATA_DIR}/val.bin: {val_n:,} | {DATA_DIR}/train.bin: {len(arr)-val_n:,} tokens", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--max_tokens", type=int, default=100_000_000)
    ap.add_argument("--bpe_docs", type=int, default=10000)
    prepare(**vars(ap.parse_args()))
    os._exit(0)   # hard exit — sidesteps the datasets streaming-thread race at finalization (data already written)
