"""A small BPE tokenizer (splits text into frequent word pieces) via the `tokenizers` library.

BPE = Byte-Pair Encoding: starts from individual bytes and repeatedly merges the most
frequent pairs → a vocabulary of "subwords". Shorter sequences than per-character → faster
training and nicer text.
"""

from __future__ import annotations

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT = "<|endoftext|>"   # end-of-story marker


def train_bpe(files: list[str], vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # covers all bytes → no <unk>
    )
    tok.train(files, trainer)
    tok.save(out_path)
    return tok


def load(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)
