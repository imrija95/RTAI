"""Bounded falsification experiment for one-pass Plasticity Genome learning.

The search evolves a tiny named law, not model weights.  Every candidate starts
from the same random FractalLM, consumes each training sample once, and updates
without autograd.  AdamW appears only as a separately measured control.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fractal import persist
from fractal.model import Config, FractalLM
from fractal.plastic_genome import PlasticityGenome, OnlinePlasticLearner


OBJECTIVE_NAMES = ("score", "recall1", "recall3", "recall_total", "memory",
                   "overwrite", "sequence")
_WORKER_NUMA = "unbound"


def _parse_cpu_list(spec: str) -> set[int]:
    cpus = set()
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    return cpus


def _pin_worker(threads: int, numa: str) -> None:
    """Pin each process-pool worker to one Linux NUMA node when available."""
    global _WORKER_NUMA
    torch.set_num_threads(max(1, threads))
    if numa == "off" or not hasattr(os, "sched_setaffinity"):
        return
    nodes = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"),
                   key=lambda path: int(path.name[4:]))
    nodes = [path for path in nodes if (path / "cpulist").exists()]
    if not nodes:
        return
    identity = multiprocessing.current_process()._identity
    worker_index = (identity[0] - 1) if identity else os.getpid()
    node = nodes[worker_index % len(nodes)]
    cpus = _parse_cpu_list((node / "cpulist").read_text(encoding="utf-8"))
    allowed = cpus & set(os.sched_getaffinity(0))
    if allowed:
        os.sched_setaffinity(0, allowed)
        _WORKER_NUMA = node.name


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _config(width=32, depth=2):
    return Config(vocab_size=32, n_embd=width, n_head=4, depth=depth, n_scales=2,
                  tau0=8.0, rho=4.0, chunk_size=16, dropout=0.0)


def _model(seed, device, width=32, depth=2):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return FractalLM(_config(width, depth)).to(device).eval()


def transition_stream(seed: int, length: int, vocab: int = 32):
    """A deterministic but unseen-start transition system for the first gate."""
    rng = np.random.default_rng(seed)
    mapping = rng.permutation(vocab)
    data = np.empty(length + 1, dtype=np.int64)
    data[0] = int(rng.integers(vocab))
    for i in range(length):
        data[i + 1] = mapping[data[i]]
        if i and i % 97 == 0:                      # multiple basins, same transition law
            data[i + 1] = int(rng.integers(vocab))
    return torch.from_numpy(data)


def _blocks(stream, block):
    for start in range(0, len(stream) - 1, block):
        pair = stream[start:start + block + 1]
        if len(pair) > 1:
            yield pair[:-1][None], pair[1:][None], start


@torch.no_grad()
def stream_loss(model, stream, device, block=32):
    states = model.init_states(1, device)
    losses = []
    for x, y, _ in _blocks(stream, block):
        logits, states = model.forward_stream(x.to(device), states)
        losses.append(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                      y.to(device).reshape(-1)).item())
    return sum(losses) / max(len(losses), 1)


def _episode_parts(rng, n_facts, values, filler=8, update=False):
    keys = rng.sample(list(range(4, 12)), n_facts)
    vals = rng.sample(list(values), n_facts)
    store = []
    for key, value in zip(keys, vals):
        store.extend([1, key, 2, value, 3])
    query = rng.randrange(n_facts)
    answer = vals[query]
    if update:
        alternatives = [value for value in values if value != answer]
        answer = rng.choice(alternatives)
        store.extend([0] * max(1, filler // 2) + [1, keys[query], 2, answer, 3])
    store.extend([0] * filler)
    query_tokens = [1, keys[query], 2]
    return (torch.tensor(store, dtype=torch.long),
            torch.tensor(query_tokens, dtype=torch.long), answer)


def _episode(rng, n_facts, values, filler=8, update=False):
    store, query, answer = _episode_parts(rng, n_facts, values, filler, update)
    return torch.cat([store, query, torch.tensor([answer])])


def learn_recall_curriculum(model, learner, device, seed, episodes):
    rng = random.Random(seed)
    for i in range(episodes):
        seq = _episode(rng, 1 + i % 3, range(12, 24), filler=4 + i % 8,
                       update=(i % 4 == 3))
        states = model.init_states(1, device)
        weight = torch.zeros(1, len(seq) - 1, device=device)
        weight[0, -1] = 1.0
        learner.learn_block(seq[:-1][None].to(device), seq[1:][None].to(device), states,
                            f"recall:{seed}:{i}", loss_weight=weight)


@torch.no_grad()
def recall_metrics(model, device, seed, n_facts, trials=24, held_out=True):
    rng = random.Random(seed)
    values = range(24, 32) if held_out else range(12, 24)
    correct = 0
    losses = []
    no_memory_losses = []
    update_correct = 0
    for trial in range(trials):
        store, query, answer = _episode_parts(rng, n_facts, values, update=(trial % 4 == 3))
        states = model.init_states(1, device)
        _, stored = model.forward_stream(store[None].to(device), states)
        ablated = [state.clone() for state in stored]
        for state in ablated:
            state.W = [torch.zeros_like(weight) for weight in state.W]
        logits, _ = model.forward_stream(query[None].to(device), stored)
        no_memory_logits, _ = model.forward_stream(query[None].to(device), ablated)
        correct += int(logits[0, -1].argmax().item() == answer)
        if trial % 4 == 3:
            update_correct += int(logits[0, -1].argmax().item() == answer)
        losses.append(F.cross_entropy(logits[0, -1].float(),
                                      torch.tensor(answer, device=device)).item())
        no_memory_losses.append(F.cross_entropy(no_memory_logits[0, -1].float(),
                                                torch.tensor(answer, device=device)).item())
    loss = sum(losses) / max(len(losses), 1)
    no_memory_loss = sum(no_memory_losses) / max(len(no_memory_losses), 1)
    return {"accuracy": correct / trials, "loss": loss,
            "memory_advantage": no_memory_loss - loss,
            "update_accuracy": update_correct / max(trials // 4, 1),
            "no_memory_loss": no_memory_loss}


def recall_accuracy(model, device, seed, n_facts, trials=24, held_out=True):
    return recall_metrics(model, device, seed, n_facts, trials, held_out)["accuracy"]


def evaluate_genome(vector, seed, device="cpu", train_tokens=768, recall_episodes=24,
                    width=32, depth=2, threads=1, return_model=False):
    torch.set_num_threads(max(1, threads))
    genome = PlasticityGenome.from_vector(vector, feedback_seed=17)
    model = _model(seed, device, width, depth)
    learner = OnlinePlasticLearner(model, genome)
    train = transition_stream(seed + 100, train_tokens)
    validation = transition_stream(seed + 100, 256)[-129:]
    frozen = _model(seed, device, width, depth)
    frozen_loss = stream_loss(frozen, validation, device)
    started = time.monotonic()
    states = model.init_states(1, device)
    for x, y, start in _blocks(train, 32):
        _, states, _ = learner.learn_block(x.to(device), y.to(device), states,
                                           f"transition:{seed}:{start}")
    learn_recall_curriculum(model, learner, device, seed + 200, recall_episodes)
    elapsed = time.monotonic() - started
    post_loss = stream_loss(model, validation, device)
    recall1_metrics = recall_metrics(model, device, seed + 300, 1)
    recall3_metrics = recall_metrics(model, device, seed + 301, 3)
    recall1, recall1_loss = recall1_metrics["accuracy"], recall1_metrics["loss"]
    recall3, recall3_loss = recall3_metrics["accuracy"], recall3_metrics["loss"]
    improvement = (frozen_loss - post_loss) / max(frozen_loss, 1e-9)
    uniform_loss = math.log(model.cfg.vocab_size)
    recall1_loss_improvement = (uniform_loss - recall1_loss) / uniform_loss
    recall3_loss_improvement = (uniform_loss - recall3_loss) / uniform_loss
    stable = learner.stable and math.isfinite(post_loss)
    # Staged fitness: sequence learning is rewarded only up to its 20% gate. Beyond that,
    # continuous recall loss supplies selection pressure even while exact accuracy is still flat.
    score = (min(improvement, 0.20)
             + 0.15 * (recall1_loss_improvement + recall3_loss_improvement)
             + 0.30 * math.tanh(recall1_metrics["memory_advantage"]
                                + recall3_metrics["memory_advantage"])
             + 0.25 * (recall1 + recall3)) if stable else -1e9
    metrics = {
        "seed": seed, "score": score, "stable": stable,
        "frozen_loss": frozen_loss, "post_loss": post_loss,
        "delta_only_control_loss": frozen_loss,
        "relative_loss_improvement": improvement,
        "recall_1fact": recall1, "recall_3fact": recall3,
        "recall_1fact_loss": recall1_loss, "recall_3fact_loss": recall3_loss,
        "recall_1fact_loss_improvement": recall1_loss_improvement,
        "recall_3fact_loss_improvement": recall3_loss_improvement,
        "recall_1fact_memory_advantage": recall1_metrics["memory_advantage"],
        "recall_3fact_memory_advantage": recall3_metrics["memory_advantage"],
        "recall_1fact_update_accuracy": recall1_metrics["update_accuracy"],
        "recall_3fact_update_accuracy": recall3_metrics["update_accuracy"],
        "learning_seconds": elapsed,
        "learning_tokens_per_second": learner.total_tokens / max(elapsed, 1e-9),
        "observed_tokens": learner.total_tokens,
        "unique_samples": len(learner.sample_ids),
        "update_norms": learner.last_update_norms,
        "fast_update_norms": learner.last_fast_update_norms,
        "all_gradients_none": all(p.grad is None for p in model.parameters()),
        "numa_node": _WORKER_NUMA,
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    learner.close()
    return (metrics, model) if return_model else metrics


def _worker(payload):
    vector, seed, train_tokens, recall_episodes, threads = payload
    return evaluate_genome(vector, seed, "cpu", train_tokens, recall_episodes,
                           threads=threads)


def _adam_control(seed, device, train_tokens=768):
    model = _model(seed, device)
    train = transition_stream(seed + 100, train_tokens)
    validation = transition_stream(seed + 100, 256)[-129:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    started = time.monotonic()
    seen = 0
    for x, y, _ in _blocks(train, 32):
        logits, loss, _, _ = model(x.to(device), targets=y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        seen += x.numel()
    elapsed = time.monotonic() - started
    return {"post_loss": stream_loss(model.eval(), validation, device),
            "learning_seconds": elapsed, "observed_tokens": seen,
            "learning_tokens_per_second": seen / max(elapsed, 1e-9)}


def _candidate_vectors(mean, sigma, population, rng):
    bounds = np.asarray(PlasticityGenome.bounds(), dtype=np.float64)
    half = (population + 1) // 2
    noise = rng.standard_normal((half, len(mean)))
    noise = np.concatenate([noise, -noise], axis=0)[:population]
    return np.clip(mean[None] + sigma[None] * noise, bounds[:, 0], bounds[:, 1])


def _metric_view(metric):
    """Use fixed-seed GPU evidence when it is available."""
    return metric.get("verification", metric)


def _objective_values(metric):
    view = _metric_view(metric)
    return {
        "score": view["score"],
        "recall1": view["recall_1fact"],
        "recall3": view["recall_3fact"],
        "recall_total": view["recall_1fact"] + view["recall_3fact"],
        "memory": (view["recall_1fact_memory_advantage"]
                   + view["recall_3fact_memory_advantage"]),
        "overwrite": (view["recall_1fact_update_accuracy"]
                      + view["recall_3fact_update_accuracy"]),
        "sequence": min(view["relative_loss_improvement"], 0.20),
    }


def _island_vectors(means, sigmas, population, rng):
    """Sample one antithetic subpopulation for every objective island."""
    base, extra = divmod(population, len(OBJECTIVE_NAMES))
    vectors = []
    labels = []
    for position, name in enumerate(OBJECTIVE_NAMES):
        count = base + int(position < extra)
        if count == 0:
            continue
        sampled = _candidate_vectors(means[name], sigmas[name], count, rng)
        vectors.extend(sampled)
        labels.extend([name] * count)
    return np.asarray(vectors), labels


def _elite_indices(metrics, count):
    """Keep objective niches instead of collapsing all useful evidence to one scalar."""
    count = min(count, len(metrics))

    def values(metric):
        view = _metric_view(metric)
        score = view["score"]
        return tuple((value, score) for value in _objective_values(view).values())

    rankings = []
    for objective in range(len(_objective_values(metrics[0]))):
        rankings.append(sorted(range(len(metrics)),
                               key=lambda i: values(metrics[i])[objective], reverse=True))
    selected = []
    for rank in range(len(metrics)):
        for ranking in rankings:
            candidate = ranking[rank]
            if candidate not in selected:
                selected.append(candidate)
                if len(selected) == count:
                    return selected
    return selected


def _run_population(vectors, seed, args):
    payloads = [(v.tolist(), seed, args.train_tokens, args.recall_episodes,
                 args.threads_per_worker) for v in vectors]
    if args.workers <= 1 or args.search_device == "cuda":
        return [evaluate_genome(v, seed, args.search_device, args.train_tokens,
                                args.recall_episodes, threads=args.threads_per_worker)
                for v in vectors]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_pin_worker,
            initargs=(args.threads_per_worker, args.numa)) as pool:
        return list(pool.map(_worker, payloads))


def search(args):
    if args.islands and args.population < len(OBJECTIVE_NAMES):
        raise ValueError(f"island search needs at least {len(OBJECTIVE_NAMES)} candidates")
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    history_path = out.with_suffix(".jsonl")
    state_path = out.with_suffix(".search.json")
    genome_path = out.with_suffix(".genome.json")
    telemetry_path = out.with_suffix(".tele.json")
    rng = np.random.default_rng(args.seed)
    bounds = np.asarray(PlasticityGenome.bounds(), dtype=np.float64)
    mean = np.asarray(PlasticityGenome().to_vector(), dtype=np.float64)
    sigma = 0.12 * (bounds[:, 1] - bounds[:, 0])
    island_means = {name: mean.copy() for name in OBJECTIVE_NAMES}
    island_sigmas = {name: sigma.copy() for name in OBJECTIVE_NAMES}
    generation = 0
    best = None
    archive = {}
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        mean, sigma = np.asarray(state["mean"]), np.asarray(state["sigma"])
        generation, best = state["generation"], state.get("best")
        archive = state.get("archive", {})
        if args.islands and "island_means" in state:
            island_means = {name: np.asarray(values)
                            for name, values in state["island_means"].items()}
            island_sigmas = {name: np.asarray(values)
                             for name, values in state["island_sigmas"].items()}
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]

    total_seconds = args.budget_minutes * 60
    reserve = min(600.0, total_seconds * 0.20)
    deadline = time.monotonic() + max(1.0, total_seconds - reserve)
    max_generations = 2 if args.phase == "smoke" else args.max_generations
    while generation < max_generations and time.monotonic() < deadline:
        if args.islands:
            vectors, island_labels = _island_vectors(
                island_means, island_sigmas, args.population, rng)
        else:
            vectors = _candidate_vectors(mean, sigma, args.population, rng)
            island_labels = ["shared"] * len(vectors)
        metrics = _run_population(vectors, args.seed + generation, args)
        if args.islands:
            island_elites = {}
            proposal_ids = []
            for name in OBJECTIVE_NAMES:
                members = [i for i, label in enumerate(island_labels) if label == name]
                ranked = sorted(
                    members,
                    key=lambda i: (_objective_values(metrics[i])[name], metrics[i]["score"]),
                    reverse=True,
                )
                keep = max(2, len(ranked) // 4)
                island_elites[name] = ranked[:keep]
                proposal_ids.append(ranked[0])
            for candidate in _elite_indices(metrics, args.elites):
                if candidate not in proposal_ids and len(proposal_ids) < args.elites:
                    proposal_ids.append(candidate)
            elite_ids = proposal_ids[:args.elites]
        else:
            island_elites = {}
            elite_ids = _elite_indices(metrics, args.elites)

        # The broad population may run on CPU. Verify objective-niche elites against one
        # fixed seed so global-best comparisons do not confuse progress with seed difficulty.
        if args.verify_device != args.search_device and torch.cuda.is_available():
            verified = {}
            for i in elite_ids:
                if time.monotonic() >= deadline:
                    break
                verified[i] = evaluate_genome(vectors[i], args.seed + 100_000,
                                              args.verify_device, args.train_tokens,
                                              args.recall_episodes)
                metrics[i]["verification"] = verified[i]
                metrics[i]["score"] = min(metrics[i]["score"], verified[i]["score"])
            if verified:
                elite_ids = list(verified)

        if args.islands:
            for name, indices in island_elites.items():
                elites = vectors[indices]
                island_means[name] = elites.mean(axis=0)
                island_sigmas[name] = np.maximum(
                    elites.std(axis=0), 0.02 * (bounds[:, 1] - bounds[:, 0]))
            mean = island_means["score"]
            sigma = island_sigmas["score"]
        else:
            elites = vectors[elite_ids]
            mean = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), 0.02 * (bounds[:, 1] - bounds[:, 0]))
        winner = max(elite_ids, key=lambda i: _metric_view(metrics[i])["score"])
        winner_view = _metric_view(metrics[winner])
        record = {"generation": generation, "best_index": winner,
                  "best": metrics[winner], "population": metrics,
                  "vectors": [vector.tolist() for vector in vectors],
                  "elite_indices": elite_ids, "islands": island_labels}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        previous_best_score = (-math.inf if best is None else
                               best.get("validation_score", _metric_view(best["metrics"])["score"]))
        if winner_view["score"] > previous_best_score:
            best = {"vector": vectors[winner].tolist(), "metrics": metrics[winner],
                    "validation_score": winner_view["score"], "generation": generation}
            PlasticityGenome.from_vector(best["vector"]).save(genome_path)
        for i in elite_ids:
            for name, value in _objective_values(metrics[i]).items():
                current = archive.get(name)
                current_key = ((-math.inf, -math.inf) if current is None else
                               (current["objective_value"], current["validation_score"]))
                candidate_key = (value, _metric_view(metrics[i])["score"])
                if candidate_key > current_key:
                    archive[name] = {
                        "vector": vectors[i].tolist(), "metrics": metrics[i],
                        "objective_value": value,
                        "validation_score": _metric_view(metrics[i])["score"],
                        "generation": generation,
                    }
                    archive_path = out.with_name(f"{out.stem}.{name}.genome.json")
                    PlasticityGenome.from_vector(vectors[i]).save(archive_path)
        _atomic_json(state_path, {"generation": generation + 1,
                                  "mean": mean.tolist(), "sigma": sigma.tolist(), "best": best,
                                  "archive": archive,
                                  "island_means": {name: values.tolist()
                                                   for name, values in island_means.items()},
                                  "island_sigmas": {name: values.tolist()
                                                    for name, values in island_sigmas.items()},
                                  "rng_state": rng.bit_generator.state})
        _atomic_json(telemetry_path, {
            "learning_signal": "local_update", "generation": generation,
            "candidate": winner, "fitness": winner_view["score"],
            "loss": winner_view["post_loss"],
            "update_norms": winner_view["update_norms"],
            "fast_update_norms": winner_view["fast_update_norms"],
            "tokens_per_second": winner_view["learning_tokens_per_second"],
            "iter": generation, "depth": 2, "n_scales": 2, "n_embd": 32,
            "n_head": 4, "gammas": [round(math.exp(-1.0 / 8.0), 4), 1.0],
            "taus": [8.0, None], "untie": False, "update_mode": "plasticity_genome",
            "ckpt": "Plasticity Genome search", "arm": "plastic_genome",
        })
        print(f"generation {generation}: score={winner_view['score']:.4f} "
              f"loss={winner_view['post_loss']:.3f} "
              f"recall={winner_view['recall_1fact']:.2f}/{winner_view['recall_3fact']:.2f}",
              flush=True)
        generation += 1

    if best is None:
        raise RuntimeError("the search budget expired before one population completed")
    final = verify_genome(PlasticityGenome.from_vector(best["vector"]), args)
    summary = {"phase": args.phase, "budget_minutes": args.budget_minutes,
               "generations": generation, "best_search": best, "verification": final,
               "archive": archive, "genome": str(genome_path)}
    _atomic_json(out, summary)
    return summary


def _restart_child(args):
    device = args.device
    model = persist.load_model(args.model, device).eval()
    states = persist.load_states(args.restart_child, device)
    payload = torch.load(args.restart_child + ".query.pt", map_location=device, weights_only=True)
    with torch.no_grad():
        logits, _ = model.forward_stream(payload["query"].to(device), states)
    predictions = logits[:, -1].argmax(dim=-1).cpu()
    accuracy = float((predictions == payload["answers"].cpu()).float().mean())
    _atomic_json(args.restart_child + ".result.json", {"accuracy": accuracy})


def process_restart_accuracy(model, device, output_prefix, trials=8):
    rng = random.Random(91_337)
    stores, queries, answers = [], [], []
    for _ in range(trials):
        key, value = rng.randrange(4, 12), rng.randrange(24, 32)
        stores.append([1, key, 2, value, 3] + [0] * 8)
        queries.append([1, key, 2])
        answers.append(value)
    store = torch.tensor(stores, device=device)
    states = model.init_states(trials, device)
    with torch.no_grad():
        _, states = model.forward_stream(store, states)
    state_path = str(output_prefix) + ".restart.pt"
    model_path = str(output_prefix) + ".restart-model.pt"
    persist.save_model(model_path, model)
    persist.save_states(state_path, states)
    persist.atomic_torch_save({"query": torch.tensor(queries), "answers": torch.tensor(answers)},
                              state_path + ".query.pt")
    cmd = [sys.executable, "-m", "fractal.exp_plastic_genome", "--restart-child", state_path,
           "--model", model_path, "--device", device]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        result = {"accuracy": 0.0, "error": (proc.stderr or proc.stdout)[-500:]}
    else:
        result = json.loads(Path(state_path + ".result.json").read_text(encoding="utf-8"))
    for suffix in ("", ".query.pt", ".result.json"):
        Path(state_path + suffix).unlink(missing_ok=True)
    Path(model_path).unlink(missing_ok=True)
    return result


def verify_genome(genome, args):
    device = args.verify_device if args.verify_device != "cuda" or torch.cuda.is_available() else "cpu"
    small = []
    for seed in range(args.seed + 10, args.seed + 13):
        metrics = evaluate_genome(genome.to_vector(), seed, device, args.train_tokens,
                                  args.recall_episodes)
        small.append(metrics)
    scaled, model = evaluate_genome(genome.to_vector(), args.seed + 20, device,
                                    args.train_tokens, args.recall_episodes,
                                    width=64, depth=4, return_model=True)
    restart = process_restart_accuracy(model, device, Path(args.results).with_suffix(""))
    adam = _adam_control(args.seed + 10, device, args.train_tokens)
    mean_improvement = sum(x["relative_loss_improvement"] for x in small) / len(small)
    mean_r1 = sum(x["recall_1fact"] for x in small) / len(small)
    mean_r3 = sum(x["recall_3fact"] for x in small) / len(small)
    scale_retention = (scaled["relative_loss_improvement"] / mean_improvement
                       if mean_improvement > 0.0 else 0.0)
    sequence_gate = all(x["stable"] and x["relative_loss_improvement"] >= 0.20 for x in small)
    recall_gate = mean_r1 >= 0.50 and mean_r3 >= 0.20 and restart["accuracy"] >= 0.50
    scale_gate = scale_retention >= 0.80
    mechanism_pass = sequence_gate and recall_gate and scale_gate
    speed_ratio = (small[0]["learning_tokens_per_second"] /
                   max(adam["learning_tokens_per_second"], 1e-9))
    thousand_x = mechanism_pass and speed_ratio >= 1000.0 and small[0]["post_loss"] <= adam["post_loss"]
    return {"small_seeds": small, "scaled": scaled, "restart": restart, "adam_control": adam,
            "mean_loss_improvement": mean_improvement, "mean_recall_1fact": mean_r1,
            "mean_recall_3fact": mean_r3, "scale_retention": scale_retention,
            "sequence_gate": sequence_gate, "recall_gate": recall_gate,
            "scale_gate": scale_gate, "mechanism_pass": mechanism_pass,
            "assistant_gate": "not_run" if not mechanism_pass else "scale_gated",
            "speed_ratio_vs_adam": speed_ratio, "thousand_x_claim": thousand_x}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "search", "verify"], default="smoke")
    ap.add_argument("--budget_minutes", type=float, default=2.0)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--max_generations", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads_per_worker", type=int, default=1)
    ap.add_argument("--numa", choices=["auto", "off"], default="auto")
    ap.add_argument("--search_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--verify_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--train_tokens", type=int, default=768)
    ap.add_argument("--recall_episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="plastic_genome_results.json")
    ap.add_argument("--genome", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--islands", action="store_true",
                    help="evolve a separate CEM distribution for each fitness objective")
    ap.add_argument("--restart-child", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="", help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    args = get_args()
    if args.restart_child:
        _restart_child(args)
        return
    if args.phase == "verify":
        if not args.genome:
            raise SystemExit("--genome is required for --phase verify")
        result = verify_genome(PlasticityGenome.load(args.genome), args)
        _atomic_json(args.results, result)
    else:
        search(args)


if __name__ == "__main__":
    main()
