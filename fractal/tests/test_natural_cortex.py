"""Natural Cortex production-path invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fractal import persist
from fractal import tokenizer as tk
from fractal.growing_cortex import teach_expert
from fractal.model import Config, FractalLM
from fractal.natural_data import (
    NaturalCorpus,
    document_digest,
    normalize_text,
    split_for_digest,
)
from fractal.natural_runtime import NaturalRuntimeSession, SkillBank
from fractal.natural_train import copy_dense_stem, natural_config


def _tokenizer(path: Path):
    corpus = [
        "Hello world. This is a compact English tokenizer sample.",
        "<|user|> Teach a skill. <|assistant|> Done. <|end|>",
        "Dogs, plants, water, books, colors, arithmetic, and simple conversation.",
    ] * 20
    return tk.train_bpe(iter(corpus), 512, str(path))


def _local_model(vocab_size=128):
    torch.manual_seed(7)
    return FractalLM(Config(
        vocab_size=vocab_size, n_embd=32, n_head=4, depth=2, n_scales=2, chunk_size=8,
        growing_cortex=True, skill_rank=4, skill_compiler="none", skill_address_dim=8,
        skill_router_threshold=0.6, skill_auto_route=False,
    ))


def test_special_tokens_are_atomic_and_tokenizer_is_non_destructive(tmp_path: Path):
    path = tmp_path / "tokenizer.json"
    tokenizer = _tokenizer(path)
    tk.assert_atomic_special_tokens(tokenizer)
    for marker in tk.SPECIAL_TOKENS:
        assert tokenizer.encode(marker, add_special_tokens=False).ids == [tokenizer.token_to_id(marker)]
    with pytest.raises(FileExistsError):
        tk.train_bpe(iter(["new corpus"]), 512, str(path))


def test_local_candidate_birth_is_zero_output_but_has_gradient():
    model = _local_model().train()
    address = model.tok_emb(torch.randint(0, model.cfg.vocab_size, (1, 5)))
    expert_id = model.skill_cortex.birth(task_features=address, name="test", synopsis="test")
    expert = model.skill_cortex.expert(expert_id)
    assert expert.up.detach().count_nonzero() == 0
    assert float(expert.down.detach().norm()) > 0
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    targets = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.no_grad(), model.skill_cortex.suspend():
        base = model(ids)[0]
    with torch.no_grad(), model.skill_cortex.force(expert_id):
        candidate = model(ids)[0]
    assert torch.equal(base, candidate)
    with model.skill_cortex.force(expert_id):
        loss = model(ids, targets=targets)[1]
    loss.backward()
    assert expert.up.grad is not None
    assert float(expert.up.grad.norm()) > 0


def test_local_teaching_updates_only_candidate():
    model = _local_model().train()
    address = model.tok_emb(torch.randint(0, model.cfg.vocab_size, (1, 5)))
    expert_id = model.skill_cortex.birth(task_features=address)
    base_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("skill_cortex.experts.")
    }
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    targets = torch.randint(0, model.cfg.vocab_size, (1, 8))
    result = teach_expert(model, expert_id, [(ids, targets)], steps=16, min_steps=16)
    assert result.update_norm > 0
    for name, expected in base_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_document_hash_split_is_stable_and_cross_source_dedup_is_possible():
    left = document_digest("source-a", "A  document\r\nwith spaces.")
    right = document_digest("source-b", "A document\nwith spaces.")
    assert normalize_text("A  document\r\nwith spaces.") == "A document\nwith spaces."
    assert left == right
    assert split_for_digest("source-a", left) == split_for_digest("source-a", left)


def test_sharded_sampler_alignment_and_fixed_validation(tmp_path: Path):
    token_file = tmp_path / "val-00000.tokens.bin"
    mask_file = tmp_path / "val-00000.mask.bin"
    source_file = tmp_path / "val-00000.source.bin"
    docs_file = tmp_path / "val-00000.docs.jsonl"
    tokens = np.arange(256, dtype=np.uint16)
    mask = np.zeros(256, dtype=np.uint8)
    mask[::7] = 1
    source = np.zeros(256, dtype=np.uint8)
    tokens.tofile(token_file)
    mask.tofile(mask_file)
    source.tofile(source_file)
    docs_file.write_text('{"start": 0, "end": 256, "source_id": 0, "digest": "x"}\n')
    shard = {
        "tokens": token_file.name, "mask": mask_file.name, "source": source_file.name,
        "documents": docs_file.name, "count": 256, "document_count": 1,
    }
    manifest = {
        "kind": "natural-cortex-corpus",
        "seed": 9,
        "splits": {
            "train": {"tokens": 256, "documents": 1, "shards": [shard]},
            "val": {"tokens": 256, "documents": 1, "shards": [shard]},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    corpus = NaturalCorpus(tmp_path)
    first = corpus.batch("val", 3, 24, "cpu", np.random.RandomState(11))
    second = corpus.batch("val", 3, 24, "cpu", np.random.RandomState(11))
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert torch.all(first[2].sum(dim=1) > 0)


def test_skill_bank_and_fast_weights_round_trip(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _tokenizer(tokenizer_path)
    model = _local_model(tokenizer.get_vocab_size()).eval()
    checkpoint = tmp_path / "base.pt"
    state_path = tmp_path / "fast.pt"
    persist.save_model(str(checkpoint), model)
    bank = SkillBank(tmp_path / "bank", model, str(checkpoint))
    runtime = NaturalRuntimeSession(
        model, tokenizer, bank, "cpu", model.init_states(1, "cpu"), str(state_path))
    result = runtime.teach(
        "polite greeting", "Reply to a greeting with a short friendly greeting.",
        [{"user": "Hello", "assistant": "Hello!"}], confirmed=True, steps=16)
    assert result["update_norm"] > 0
    assert runtime.rate(5)["action"] == "committed"
    verification = runtime.restart_verification(str(checkpoint), str(state_path))
    assert verification["complete"]
    assert bank.manifest["router"]["threshold"] == 0.6


def test_rating_three_leaves_no_durable_skill(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _tokenizer(tokenizer_path)
    model = _local_model(tokenizer.get_vocab_size()).eval()
    checkpoint = tmp_path / "base.pt"
    persist.save_model(str(checkpoint), model)
    bank = SkillBank(tmp_path / "bank", model, str(checkpoint))
    runtime = NaturalRuntimeSession(model, tokenizer, bank, "cpu")
    runtime.teach(
        "neutral", "A candidate that is not committed.",
        [{"user": "Test", "assistant": "Test."}], confirmed=True, steps=16)
    assert runtime.rate(3)["action"] == "discarded"
    assert bank.manifest["experts"] == []
    assert len(model.skill_cortex.experts) == 0


def test_production_preset_excludes_refuted_paths():
    dense = natural_config("dense")
    moe = natural_config("moe")
    for cfg in (dense, moe):
        assert not cfg.selective
        assert cfg.event_budget == 1.0
        assert not cfg.event_algebra
        assert not cfg.untie
        assert cfg.growing_cortex
        assert cfg.skill_compiler == "none"
        assert not cfg.skill_auto_route
    assert dense.n_experts == 1
    assert moe.n_experts == 4 and moe.moe_mode == "top1"


def test_dense_moe_share_identical_initial_stem():
    dense_cfg = Config(vocab_size=64, n_embd=32, n_head=4, depth=2, n_scales=2,
                       n_experts=1)
    moe_cfg = Config(vocab_size=64, n_embd=32, n_head=4, depth=2, n_scales=2,
                     n_experts=4, moe_mode="top1")
    torch.manual_seed(5)
    dense = FractalLM(dense_cfg)
    torch.manual_seed(19)
    moe = FractalLM(moe_cfg)
    copy_dense_stem(dense, moe)
    torch.testing.assert_close(dense.tok_emb.weight, moe.tok_emb.weight, rtol=0, atol=0)
    for expert in moe.block.mlp.experts:
        torch.testing.assert_close(
            dense.block.mlp.fc.weight, expert.fc.weight, rtol=0, atol=0)
