"""DailyDialog pipeline — conversational training (alternating turns).

Format: turns within a dialog separated by `<|eot|>`, dialogs by `<|endoftext|>`. Writes to
the STANDARD paths (fractal_data/{train,val}.bin + fractal_tokenizer.json), so
train.py / chat.py / live.py work unchanged (just a retrained tokenizer + data).

Run:  uv run python -m fractal.data_dialog
"""

from __future__ import annotations

import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

DATA_DIR = "fractal_data"
TOK_PATH = "fractal_tokenizer.json"
EOT = "<|endoftext|>"
EOTURN = "<|eot|>"


def _dialogs(split: str):
    """Generator of dialogs as text: turns joined by <|eot|>.
    ConvLab/dailydialog via STREAMING (sidesteps the deprecated script and the verification error).
    Each example has 'turns' = [{'speaker','utterance',...}, ...]."""
    ds = load_dataset("ConvLab/dailydialog", split=split, streaming=True)
    for ex in ds:
        turns = ex.get("turns") or []
        utts = [t["utterance"].strip() for t in turns
                if isinstance(t, dict) and t.get("utterance", "").strip()]
        if len(utts) >= 2:
            yield (f" {EOTURN} ").join(utts)


def prepare(vocab_size: int = 8000):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("downloading dialogs (ConvLab/dailydialog, stream)…", flush=True)
    dialogs = list(_dialogs("train"))                 # the only available split; small (~13k)
    n_val = max(300, len(dialogs) // 20)              # 5 % held out for validation
    val_d, train_d = dialogs[:n_val], dialogs[n_val:]
    print(f"dialogs: {len(dialogs)} (train {len(train_d)} / val {len(val_d)})", flush=True)

    print("training BPE…", flush=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT, EOTURN],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_d, trainer=trainer)
    tok.save(TOK_PATH)
    print(f"vocab_size = {tok.get_vocab_size()}", flush=True)

    eot = tok.token_to_id(EOT)
    for name, ds in [("train", train_d), ("val", val_d)]:
        ids = []
        for d in ds:
            ids.extend(tok.encode(" " + d).ids)
            ids.append(eot)
        np.array(ids, dtype=np.uint16).tofile(f"{DATA_DIR}/{name}.bin")
        print(f"{DATA_DIR}/{name}.bin: {len(ids):,} tokens", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    prepare()
