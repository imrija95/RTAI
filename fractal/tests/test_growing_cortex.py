"""Append-only Growing Cortex invariants and persistence tests."""

from __future__ import annotations

import tempfile

import torch

from fractal import persist
from fractal.growing_cortex import GrowthController, GrowthEvidence
from fractal.model import Config, FractalLM


def _model():
    torch.manual_seed(41)
    return FractalLM(Config(
        vocab_size=64, n_embd=32, n_head=4, depth=2, n_scales=2, chunk_size=8,
        growing_cortex=True, skill_rank=4, skill_compiler="full", skill_address_dim=32,
        skill_router_threshold=0.2,
    )).eval()


def test_candidate_birth_is_exactly_function_preserving():
    model = _model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 9))
    with torch.no_grad():
        before = model(ids)[0]
    compiled = model.compile_skill(ids)
    expert_id = model.skill_cortex.birth(compiled=compiled)
    with torch.no_grad():
        after = model(ids)[0]
    assert expert_id == 0
    assert model.skill_cortex.expert(0).status == "candidate"
    assert torch.equal(before, after)


def test_compiler_receives_outer_execution_gradient():
    model = _model().train()
    instruction = torch.randint(0, model.cfg.vocab_size, (1, 7))
    query = torch.randint(0, model.cfg.vocab_size, (3, 5))
    target = torch.randint(0, model.cfg.vocab_size, (3,))
    compiled = model.compile_skill(instruction)
    with model.skill_cortex.use_compiled(compiled):
        logits = model(query)[0][:, -1]
        loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert model.skill_cortex.compiler.up_proj.weight.grad is not None
    assert float(model.skill_cortex.compiler.up_proj.weight.grad.norm()) > 0.0


def test_quarantined_expert_returns_to_base_function():
    model = _model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    compiled = model.compile_skill(ids)
    expert_id = model.skill_cortex.birth(compiled=compiled)
    model.skill_cortex.commit(expert_id, mature=True)
    with torch.no_grad(), model.skill_cortex.suspend():
        base = model(ids)[0]
    with torch.no_grad(), model.skill_cortex.force(expert_id):
        adapted = model(ids)[0]
    assert not torch.equal(base, adapted)
    model.skill_cortex.quarantine(expert_id)
    with torch.no_grad():
        restored = model(ids)[0]
    assert torch.equal(base, restored)


def test_growing_cortex_round_trip_restores_structure_and_logits():
    model = _model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    compiled = model.compile_skill(ids)
    expert_id = model.skill_cortex.birth(compiled=compiled, created_step=7)
    model.skill_cortex.commit(expert_id, confidence=0.4, mature=True)
    with torch.no_grad(), model.skill_cortex.force(expert_id):
        expected = model(ids)[0]
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/model.pt"
        persist.save_model(path, model)
        loaded = persist.load_model(path, "cpu").eval()
    assert loaded.skill_cortex.manifest() == model.skill_cortex.manifest()
    with torch.no_grad(), loaded.skill_cortex.force(expert_id):
        actual = loaded(ids)[0]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_growth_policy_requires_repeated_novel_colliding_failure():
    controller = GrowthController(patience=3, cooldown=10)
    evidence = GrowthEvidence(
        fingerprint="skill-a", error=2.0, nearest_similarity=0.1,
        gradient_cosine=-0.3, existing_expert_improved=False,
    )
    assert not controller.observe(evidence, 1)
    assert not controller.observe(evidence, 2)
    assert controller.observe(evidence, 3)
    assert not controller.observe(evidence, 4)
