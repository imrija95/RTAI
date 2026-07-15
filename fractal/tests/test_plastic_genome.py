"""Load-bearing checks for explicit one-pass Plasticity Genome learning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from fractal.exp_plastic_genome import (OBJECTIVE_NAMES, _elite_indices, _island_vectors,
                                        _model, _parse_cpu_list, process_restart_accuracy,
                                        recall_metrics)
from fractal.plastic_genome import PlasticityGenome, OnlinePlasticLearner, _group_for


def _zero_genome():
    genome = PlasticityGenome()
    for rule in genome.rules.values():
        rule.error = rule.hebbian = rule.oja = rule.decay = 0.0
    return genome


def test_schema_round_trip():
    assert _parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    genome = PlasticityGenome()
    rebuilt = PlasticityGenome.from_vector(genome.to_vector(), genome.feedback_seed)
    assert rebuilt.to_dict() == genome.to_dict()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "genome.json"
        genome.save(path)
        assert PlasticityGenome.load(path).to_dict() == genome.to_dict()


def test_no_grad_one_pass_and_all_surfaces():
    torch.manual_seed(0)
    model = _model(0, "cpu")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    learner = OnlinePlasticLearner(model, PlasticityGenome())
    x = torch.arange(24).remainder(model.cfg.vocab_size)[None]
    y = torch.arange(1, 25).remainder(model.cfg.vocab_size)[None]
    states = model.init_states(1, "cpu")
    _, new_states, _ = learner.learn_block(x, y, states, "unique-block")
    assert all(parameter.grad is None for parameter in model.parameters())
    assert learner.total_tokens == x.numel()
    assert learner.last_fast_update_norms
    assert any(value > 0 for row in learner.last_fast_update_norms for value in row)
    changed = set()
    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter):
            if name == "tok_emb.weight":
                changed.add("embedding")
            module_name = name.rsplit(".", 1)[0]
            group = _group_for(module_name)
            if group:
                changed.add(group)
    assert changed == {"embedding", "qk", "routing", "projection", "mlp"}
    try:
        learner.learn_block(x, y, new_states, "unique-block")
    except ValueError as exc:
        assert "replayed" in str(exc)
    else:
        raise AssertionError("sample replay was accepted")
    learner.close()


def test_zero_genome_keeps_slow_weights_fixed():
    model = _model(1, "cpu")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    learner = OnlinePlasticLearner(model, _zero_genome())
    x = torch.randint(0, model.cfg.vocab_size, (1, 16))
    y = torch.randint(0, model.cfg.vocab_size, (1, 16))
    learner.learn_block(x, y, model.init_states(1, "cpu"), "zero")
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    learner.close()


def test_scale_transfer_and_process_restart():
    genome = PlasticityGenome()
    model = _model(2, "cpu", width=64, depth=4)
    learner = OnlinePlasticLearner(model, genome)
    x = torch.randint(0, model.cfg.vocab_size, (1, 16))
    y = torch.randint(0, model.cfg.vocab_size, (1, 16))
    learner.learn_block(x, y, model.init_states(1, "cpu"), "scaled")
    learner.close()
    metrics = recall_metrics(model, "cpu", 7, 1, trials=4)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["memory_advantage"]))
    with tempfile.TemporaryDirectory() as directory:
        result = process_restart_accuracy(model, "cpu", Path(directory) / "roundtrip", trials=2)
    assert "accuracy" in result and 0.0 <= result["accuracy"] <= 1.0


def test_objective_niches_survive_elite_selection():
    base = {"score": 0.1, "recall_1fact": 0.0, "recall_3fact": 0.0,
            "recall_1fact_memory_advantage": 0.0, "recall_3fact_memory_advantage": 0.0,
            "recall_1fact_update_accuracy": 0.0, "recall_3fact_update_accuracy": 0.0,
            "relative_loss_improvement": 0.0}
    metrics = []
    for field in ("score", "recall_1fact", "recall_3fact",
                  "recall_1fact_memory_advantage", "recall_1fact_update_accuracy",
                  "relative_loss_improvement"):
        candidate = dict(base)
        candidate[field] = 1.0
        metrics.append(candidate)
    metrics.extend(dict(base, score=0.9) for _ in range(10))
    assert set(_elite_indices(metrics, 6)) == set(range(6))


def test_islands_keep_separate_populations():
    genome = PlasticityGenome()
    mean = torch.tensor(genome.to_vector()).numpy()
    sigma = torch.ones_like(torch.tensor(mean)).numpy() * 0.01
    means = {name: mean.copy() for name in OBJECTIVE_NAMES}
    sigmas = {name: sigma.copy() for name in OBJECTIVE_NAMES}
    vectors, labels = _island_vectors(means, sigmas, 64, np.random.default_rng(7))
    assert vectors.shape == (64, len(mean))
    assert set(labels) == set(OBJECTIVE_NAMES)
    assert max(labels.count(name) for name in OBJECTIVE_NAMES) <= 10


def main():
    test_schema_round_trip()
    test_no_grad_one_pass_and_all_surfaces()
    test_zero_genome_keeps_slow_weights_fixed()
    test_scale_transfer_and_process_restart()
    test_objective_niches_survive_elite_selection()
    test_islands_keep_separate_populations()
    print("OK - plastic genomes are explicit, one-pass, scale-free, and restart-persistent")


if __name__ == "__main__":
    main()
