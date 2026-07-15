"""Conversational fine-tune data — SAFE for a running pretraining.

Unlike data_dialog.py: does NOT train a new tokenizer and does NOT overwrite fractal_data/
or fractal_tokenizer.json. It reuses the EXISTING 16k tokenizer (the one the large model is
built on) and writes to a SEPARATE fractal_data_chat/ directory.

The 16k tokenizer only has <|endoftext|>, not <|eot|>. Turn-taking is therefore taught with
TEXTUAL roles "User:/Assistant:" (BPE encodes them as ordinary tokens) + <|endoftext|>
between dialogs. Chat then generates after "Assistant:" and stops at the next "User:".

Run:  uv run python -m fractal.data_chat_ft
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset

from fractal import tokenizer as tk

OUT_DIR = "fractal_data_chat"
TOK_PATH = "fractal_tokenizer.json"        # EXISTING 16k — read only
EOT = "<|endoftext|>"
ROLES = ("User", "Assistant")


def _dialogs(split: str):
    """Dialog as alternating User/Assistant turns (ConvLab/dailydialog, streaming)."""
    ds = load_dataset("ConvLab/dailydialog", split=split, streaming=True)
    for ex in ds:
        turns = ex.get("turns") or []
        utts = [t["utterance"].strip() for t in turns
                if isinstance(t, dict) and t.get("utterance", "").strip()]
        if len(utts) >= 2:
            lines = [f"{ROLES[i % 2]}: {u}" for i, u in enumerate(utts)]
            yield "\n".join(lines)


def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = tk.load(TOK_PATH)
    eot = tok.token_to_id(EOT)
    assert eot is not None, "16k tokenizer must have <|endoftext|>"
    print(f"tokenizer: {TOK_PATH} (vocab {tok.get_vocab_size()}), <|endoftext|>={eot}", flush=True)

    print("downloading dialogs (ConvLab/dailydialog, stream)…", flush=True)
    dialogs = list(_dialogs("train"))
    n_val = max(300, len(dialogs) // 20)
    val_d, train_d = dialogs[:n_val], dialogs[n_val:]
    print(f"dialogs: {len(dialogs)} (train {len(train_d)} / val {len(val_d)})", flush=True)

    for name, ds in [("train", train_d), ("val", val_d)]:
        ids: list[int] = []
        for d in ds:
            ids.extend(tok.encode(d).ids)
            ids.append(eot)
        np.array(ids, dtype=np.uint16).tofile(f"{OUT_DIR}/{name}.bin")
        print(f"{OUT_DIR}/{name}.bin: {len(ids):,} tokens", flush=True)
    print("done. Fine-tune reads via train.py --data_dir fractal_data_chat", flush=True)


if __name__ == "__main__":
    prepare()
    os._exit(0)      # sidesteps the datasets streaming-thread race at finalization
