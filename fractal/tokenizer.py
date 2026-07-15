"""Byte-level BPE tokenizer (reusing the ready-made `tokenizers` framework).

Byte-level → no <unk>, every byte covered. `<|endoftext|>` separates stories.
"""

from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

EOT = "<|endoftext|>"


def train_bpe(text_iter, vocab_size: int, save_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    tok.save(save_path)
    return tok


def load(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)
