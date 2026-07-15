"""TinyStories: download, train a BPE tokenizer, and tokenize into binary .bin files.

TinyStories = a dataset of simple children's stories (Microsoft) on which even small models
can produce coherent English. Stories are separated by the <|endoftext|> marker.

    uv run python -m rtai.data_text            # download + prepare (default subset)
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import torch

from .tokenizer import EOT, train_bpe, load as load_tok

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
VALID_URL = f"{BASE}/TinyStoriesV2-GPT4-valid.txt"
TRAIN_URL = f"{BASE}/TinyStoriesV2-GPT4-train.txt"


def _download(url: str, path: str, max_bytes: int | None = None):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[data] downloading {os.path.basename(path)}"
          + (f" (first {max_bytes // 1024 // 1024} MB)" if max_bytes else ""))
    cmd = ["curl", "-Ls", "-o", path]
    if max_bytes:
        cmd += ["-r", f"0-{max_bytes - 1}"]   # only the start of the file (Range request)
    subprocess.run(cmd + [url], check=True)


def _encode_to_bin(txt_path: str, bin_path: str, tok, eot: int):
    text = open(txt_path, encoding="utf-8", errors="ignore").read()
    stories = [s.strip() for s in text.split(EOT) if s.strip()]
    stories = stories[:-1] if len(stories) > 1 else stories  # the last one may be truncated
    ids: list[int] = []
    for i in range(0, len(stories), 10000):                  # batched for speed
        for enc in tok.encode_batch(stories[i:i + 10000]):
            ids.extend(enc.ids)
            ids.append(eot)
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    print(f"[data] {os.path.basename(bin_path)}: {len(ids):,} tokens")


def prepare(data_dir="data", tok_path="tokenizer.json", vocab_size=8000, train_mb=400):
    valid_txt = os.path.join(data_dir, "valid.txt")
    train_txt = os.path.join(data_dir, "train.txt")
    _download(VALID_URL, valid_txt)
    _download(TRAIN_URL, train_txt, max_bytes=train_mb * 1024 * 1024)

    if not os.path.exists(tok_path):
        print(f"[data] training BPE tokenizer (vocab={vocab_size})…")
        train_bpe([valid_txt], vocab_size, tok_path)   # a sample from valid is enough
    tok = load_tok(tok_path)
    eot = tok.token_to_id(EOT)

    for split, txt in [("train", train_txt), ("val", valid_txt)]:
        binp = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(binp):
            print(f"[data] tokenizing {split}…")
            _encode_to_bin(txt, binp, tok, eot)
    return tok


def get_batch(split, block_size, batch_size, device, data_dir="data"):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mb", type=int, default=400)
    ap.add_argument("--vocab_size", type=int, default=8000)
    args = ap.parse_args()
    prepare(train_mb=args.train_mb, vocab_size=args.vocab_size)
    print("[data] done.")
