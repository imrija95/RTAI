"""Byte-level BPE tokenizer with atomic chat, tool, and teaching markers."""

from __future__ import annotations

import os
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

EOT = "<|endoftext|>"
SPECIAL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|skill|>",
    "<|teach|>",
    "<|end|>",
    EOT,
)


def train_bpe(text_iter, vocab_size: int, save_path: str, *,
              special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
              overwrite: bool = False) -> Tokenizer:
    """Train and atomically save a tokenizer without replacing an existing artifact by default."""
    destination = Path(save_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing tokenizer: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    assert_atomic_special_tokens(tok, special_tokens)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        tok.save(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return tok


def load(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def assert_atomic_special_tokens(tok: Tokenizer,
                                 special_tokens: tuple[str, ...] = SPECIAL_TOKENS) -> None:
    """Raise when a required marker is absent or encodes as more than one token."""
    failures = []
    for marker in special_tokens:
        token_id = tok.token_to_id(marker)
        encoded = tok.encode(marker, add_special_tokens=False).ids
        if token_id is None or encoded != [token_id]:
            failures.append(f"{marker} -> id={token_id}, encoded={encoded}")
    if failures:
        raise ValueError("special tokens are not atomic: " + "; ".join(failures))
