"""TinyStories data pipeline: download → train BPE → tokenize into a binary.

Run:  uv run python -m fractal.data --train_mb 400 --vocab_size 8000

Saves fractal_data/{train.bin,val.bin} (np.uint16) and fractal_tokenizer.json.
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from fractal import tokenizer as tk

DATA_DIR = "fractal_data"
TOK_PATH = "fractal_tokenizer.json"
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE}/TinyStoriesV2-GPT4-train.txt"
VAL_URL = f"{BASE}/TinyStoriesV2-GPT4-valid.txt"


def _download(url: str, dst: str, max_mb: int | None) -> None:
    if os.path.exists(dst):
        print(f"  {dst} already exists, skipping")
        return
    cmd = ["curl", "-L", "-o", dst]
    if max_mb is not None:
        cmd += ["-r", f"0-{max_mb * 1024 * 1024 - 1}"]   # Range: only the first max_mb MB
    cmd.append(url)
    print(f"  downloading {url} → {dst}" + (f" (first {max_mb} MB)" if max_mb else ""))
    subprocess.run(cmd, check=True)


def _stories(path: str):
    """Generator of stories separated by <|endoftext|>."""
    buf: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == tk.EOT:
                if buf:
                    yield "".join(buf)
                    buf = []
            else:
                buf.append(line)
    if buf:
        yield "".join(buf)


def _encode_to_bin(tok, txt_path: str, bin_path: str) -> None:
    eot = tok.token_to_id(tk.EOT)
    ids: list[int] = []
    for story in _stories(txt_path):
        ids.extend(tok.encode(story).ids)
        ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"  {bin_path}: {len(arr):,} tokens")


def prepare(train_mb: int, vocab_size: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    train_txt = os.path.join(DATA_DIR, "train.txt")
    val_txt = os.path.join(DATA_DIR, "valid.txt")
    _download(TRAIN_URL, train_txt, train_mb)
    _download(VAL_URL, val_txt, None)

    if os.path.exists(TOK_PATH):
        print(f"  {TOK_PATH} already exists, skipping BPE training")
        tok = tk.load(TOK_PATH)
    else:
        print("  training BPE on the validation sample…")
        tok = tk.train_bpe(_stories(val_txt), vocab_size, TOK_PATH)
    print(f"  vocab_size = {tok.get_vocab_size()}")

    _encode_to_bin(tok, train_txt, os.path.join(DATA_DIR, "train.bin"))
    _encode_to_bin(tok, val_txt, os.path.join(DATA_DIR, "val.bin"))
    print("done.")


def get_batch(split: str, batch_size: int, seq_len: int, device, data_dir: str | None = None):
    """Random spans of length seq_len (+1 for the target). seq_len = n_segments × block_size.
    data_dir: where to read the .bin from (default fractal_data) — enables fine-tune on a different corpus."""
    path = os.path.join(data_dir or DATA_DIR, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
    import torch
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def get_masked_batch(split: str, batch_size: int, seq_len: int, device, data_dir: str | None = None):
    """Like get_batch, plus a per-target loss weight w from a parallel uint8 `{split}.mask.bin`
    (1 = train on this token). Used by Phase-2 masked training (loss on assistant/<|tool_call|>/<|end|>
    only). w is aligned to the TARGET y, i.e. w[t] weights predicting y[t]."""
    d = data_dir or DATA_DIR
    data = np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint16, mode="r")
    mask = np.memmap(os.path.join(d, f"{split}.mask.bin"), dtype=np.uint8, mode="r")
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
    w = np.stack([mask[i + 1:i + 1 + seq_len].astype(np.float32) for i in ix])
    import torch
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(w).to(device))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mb", type=int, default=400)
    ap.add_argument("--vocab_size", type=int, default=8000)
    prepare(**vars(ap.parse_args()))
