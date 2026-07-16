"""Falsification screen for append-only, meta-compiled skill hemispheres.

The model is meta-trained on procedural episodes.  A support specification is compiled into a
low-rank candidate, while the outer loss is measured on unseen executions that do not contain the
procedure.  Held-out procedures are then committed one by one into an append-only expert bank and
queried through sticky task-level routing after the original specification has been removed.

This is a mechanism test, not an assistant-understanding claim.  The procedural language is a
scale-invariant proxy with an exact verifier.  A natural-language and agent-tool gate follows only
if the mechanism beats the declared baselines here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from fractal import persist
from fractal.model import Config, FractalLM


PAD = 0
SPEC = 1
CALL = 2
END = 3


def _atomic_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


class ProcedureWorld:
    """Compositional opaque skill names plus exactly executable modular procedures."""

    def __init__(self, modulus: int = 16, id_parts: int = 8):
        self.modulus = int(modulus)
        self.id_parts = int(id_parts)
        self.id_a = 4
        self.id_b = self.id_a + id_parts
        self.op = self.id_b + id_parts
        self.coef = self.op + 3
        self.value = self.coef + modulus
        self.vocab_size = self.value + modulus

    def tasks(self, seed: int) -> list[tuple[int, int, int, int, int]]:
        rng = random.Random(seed)
        names = [(a, b) for a in range(self.id_parts) for b in range(self.id_parts)]
        procedures = [
            (op, a, b)
            for op in range(3)
            for a in range(1, self.modulus)
            for b in range(self.modulus)
        ]
        rng.shuffle(names)
        rng.shuffle(procedures)
        return [
            (name_a, name_b, op, a, b)
            for (name_a, name_b), (op, a, b) in zip(names, procedures)
        ]

    def sample_tasks(self, rng: random.Random, count: int):
        """Freshly permute opaque names and procedures so name memorization cannot pass."""
        names = rng.sample(
            [(a, b) for a in range(self.id_parts) for b in range(self.id_parts)],
            count,
        )
        procedures = [
            (
                rng.randrange(3),
                rng.randrange(1, self.modulus),
                rng.randrange(self.modulus),
            )
            for _ in range(count)
        ]
        return [
            (name_a, name_b, op, a, b)
            for (name_a, name_b), (op, a, b) in zip(names, procedures)
        ]

    def sample_negative_tasks(self, rng: random.Random, count: int, excluded):
        excluded_names = {(task[0], task[1]) for task in excluded}
        available = [
            (a, b) for a in range(self.id_parts) for b in range(self.id_parts)
            if (a, b) not in excluded_names
        ]
        names = rng.sample(available, count)
        return [
            (
                name_a, name_b, rng.randrange(3),
                rng.randrange(1, self.modulus), rng.randrange(self.modulus),
            )
            for name_a, name_b in names
        ]

    def instruction(self, task, device):
        name_a, name_b, op, a, b = task
        return torch.tensor([[
            SPEC, self.id_a + name_a, self.id_b + name_b,
            self.op + op, self.coef + a, self.coef + b, END,
        ]], dtype=torch.long, device=device)

    def query(self, task, xs, device):
        name_a, name_b, _op, _a, _b = task
        rows = [
            [CALL, self.id_a + name_a, self.id_b + name_b, self.value + int(x)]
            for x in xs
        ]
        return torch.tensor(rows, dtype=torch.long, device=device)

    def address(self, task, device):
        """Stable control-plane address, independent from execution arguments."""
        name_a, name_b, _op, _a, _b = task
        return torch.tensor([[
            CALL, self.id_a + name_a, self.id_b + name_b,
        ]], dtype=torch.long, device=device)

    def context(self, task, xs, device):
        instruction = self.instruction(task, device)[0].tolist()
        query = self.query(task, xs, device).tolist()
        return torch.tensor(
            [instruction + row for row in query],
            dtype=torch.long, device=device,
        )

    def answer(self, task, x):
        _name_a, _name_b, op, a, b = task
        if op == 0:
            y = a * x + b
        elif op == 1:
            y = a * (x * x) + b
        else:
            y = a * (self.modulus - 1 - x) + b
        return y % self.modulus

    def targets(self, task, xs, device):
        rows = []
        for x in xs:
            row = [-1] * 4
            row[-1] = self.value + self.answer(task, int(x))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long, device=device)


def _route_loss(model, compiled, query):
    features = model.tok_emb(query).mean(dim=1)
    route_query = F.normalize(model.skill_cortex.query_proj(features), dim=-1)
    return 1.0 - (route_query * compiled.key[None]).sum(dim=-1).mean()


def _interpreter_step(model, world, tasks, queries_per_task, device):
    contexts = []
    target_rows = []
    for task in tasks:
        query_x = torch.randint(
            0, world.modulus, (queries_per_task,), device=device).tolist()
        context = world.context(task, query_x, device)
        targets = torch.full_like(context, -1)
        targets[:, -1] = world.targets(task, query_x, device)[:, -1]
        contexts.append(context)
        target_rows.append(targets)
    with model.skill_cortex.suspend():
        return model(
            torch.cat(contexts), targets=torch.cat(target_rows))[1]


def _meta_step(model, world, tasks, negative_tasks, queries_per_task, device):
    instructions = []
    queries = []
    targets = []
    contexts = []
    for task in tasks:
        query_x = torch.randint(0, world.modulus, (queries_per_task,), device=device).tolist()
        instructions.append(world.instruction(task, device))
        queries.append(world.query(task, query_x, device))
        targets.append(world.targets(task, query_x, device))
        contexts.append(world.context(task, query_x, device))
    instruction_batch = torch.cat(instructions)
    query_batch = torch.cat(queries)
    target_batch = torch.cat(targets)
    compiled = model.compile_skill(instruction_batch)
    expanded = compiled.repeat_interleave(queries_per_task)
    with torch.no_grad(), model.skill_cortex.suspend():
        teacher = model(torch.cat(contexts))[0][:, -1].float().softmax(dim=-1)
    with model.skill_cortex.use_compiled(expanded):
        logits, execution, _states, _delta = model(query_batch, targets=target_batch)
    distill = F.kl_div(
        logits[:, -1].float().log_softmax(dim=-1), teacher, reduction="batchmean")
    execution = execution + 0.5 * distill

    # Skill identity is a control-plane signal.  Do not mix it with the variable execution payload:
    # the selected hemisphere remains sticky while arguments and tool observations change.
    address_batch = query_batch[:, :3]
    route_queries = F.normalize(
        model.skill_cortex.query_proj(model.tok_emb(address_batch).mean(dim=1)),
        dim=-1,
    )
    keys = compiled.key
    route_similarities = route_queries @ keys.T
    route_logits = route_similarities / 0.1
    route_targets = torch.arange(
        len(tasks), device=device).repeat_interleave(queries_per_task)
    positive_similarity = route_similarities.gather(
        1, route_targets[:, None]).squeeze(1)
    positive_target = min(model.skill_cortex.router_threshold + 0.05, 0.95)
    positive_loss = F.relu(positive_target - positive_similarity).mean()
    negative_query = torch.cat([
        world.address(task, device) for task in negative_tasks
    ])
    negative_features = F.normalize(
        model.skill_cortex.query_proj(model.tok_emb(negative_query).mean(dim=1)),
        dim=-1,
    )
    negative_similarity = negative_features @ keys.T
    negative_loss = F.relu(negative_similarity.max(dim=-1).values - 0.30).mean()
    route_loss = (
        F.cross_entropy(route_logits, route_targets)
        + positive_loss
        + negative_loss
    )
    return execution + 0.25 * route_loss, execution.detach(), route_loss.detach()


@torch.no_grad()
def _accuracy(model, world, task, device, *, compiled=None, forced_id=None, n=32):
    xs = [i % world.modulus for i in range(n)]
    query = world.query(task, xs, device)
    targets = world.targets(task, xs, device)[:, -1]
    if compiled is not None:
        context = model.skill_cortex.use_compiled(compiled)
    elif forced_id is not None:
        context = model.skill_cortex.force(forced_id)
    else:
        expert_id, _similarity = model.route_skill_from_ids(query[:1, :3])
        context = model.skill_cortex.force(expert_id)
    with context:
        predictions = model(query)[0][:, -1].argmax(dim=-1)
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _control_hijack(model, world, tasks, device):
    routed = 0
    rows = []
    for task in tasks:
        query = world.address(task, device)
        expert_id, similarity = model.route_skill_from_ids(query)
        routed += expert_id is not None
        rows.append({"expert_id": expert_id, "similarity": similarity})
    return routed / max(len(tasks), 1), rows


@torch.no_grad()
def _routing_metrics(model, world, tasks, device):
    """Measure task-address selection independently from expert execution quality."""
    correct = 0
    rejected = 0
    wrong = 0
    similarities = []
    rows = []
    total = len(tasks) * world.modulus
    for expected_id, task in enumerate(tasks):
        task_correct = 0
        task_rejected = 0
        task_wrong = 0
        task_similarities = []
        for x in range(world.modulus):
            query = world.query(task, [x], device)
            expert_id, similarity = model.route_skill_from_ids(query[:, :3])
            task_similarities.append(similarity)
            similarities.append(similarity)
            if expert_id == expected_id:
                correct += 1
                task_correct += 1
            elif expert_id is None:
                rejected += 1
                task_rejected += 1
            else:
                wrong += 1
                task_wrong += 1
        rows.append({
            "expert_id": expected_id,
            "correct_rate": task_correct / world.modulus,
            "rejection_rate": task_rejected / world.modulus,
            "wrong_rate": task_wrong / world.modulus,
            "mean_top_similarity": sum(task_similarities) / world.modulus,
            "min_top_similarity": min(task_similarities),
        })
    return {
        "correct_rate": correct / total,
        "rejection_rate": rejected / total,
        "wrong_rate": wrong / total,
        "mean_top_similarity": sum(similarities) / total,
        "rows": rows,
    }


@torch.no_grad()
def _interpreter_accuracy(model, world, tasks, device, n_eval):
    rows = []
    with model.skill_cortex.suspend():
        for task in tasks:
            xs = [i % world.modulus for i in range(n_eval)]
            context = world.context(task, xs, device)
            target = world.targets(task, xs, device)[:, -1]
            prediction = model(context)[0][:, -1].argmax(dim=-1)
            rows.append(float((prediction == target).float().mean()))
    return sum(rows) / len(rows)


def _evaluate_growing(model, world, tasks, controls, device, n_eval):
    baseline = []
    with model.skill_cortex.suspend():
        for task in tasks:
            baseline.append(_accuracy(model, world, task, device, n=n_eval))

    compiled_accuracy = []
    steps = []
    for task in tasks:
        instruction = world.instruction(task, device)
        compiled = model.compile_skill(instruction)
        compiled_accuracy.append(
            _accuracy(model, world, task, device, compiled=compiled, n=n_eval))
        before_logits = None
        probe = world.query(task, [0, 1, 2], device)
        with torch.no_grad():
            before_logits = model(probe)[0].detach().cpu()
        expert_id = model.skill_cortex.birth(compiled=compiled, created_step=len(steps))
        with torch.no_grad():
            after_birth = model(probe)[0].detach().cpu()
        birth_diff = float((before_logits - after_birth).abs().max())
        model.skill_cortex.commit(expert_id, confidence=0.0, mature=True)
        memory_rows = [
            _accuracy(model, world, old_task, device, n=n_eval)
            for old_task in tasks[:len(steps) + 1]
        ]
        steps.append({
            "skills": len(memory_rows),
            "expert_id": expert_id,
            "birth_max_logit_difference": birth_diff,
            "mean_accuracy": sum(memory_rows) / len(memory_rows),
            "oldest_accuracy": memory_rows[0],
            "newest_accuracy": memory_rows[-1],
            "accuracies": memory_rows,
        })
    routing = _routing_metrics(model, world, tasks, device)
    hijack, control_rows = _control_hijack(model, world, controls, device)
    return {
        "baseline_accuracy": sum(baseline) / len(baseline),
        "compiled_accuracy": sum(compiled_accuracy) / len(compiled_accuracy),
        "steps": steps,
        "final_accuracy": steps[-1]["mean_accuracy"],
        "oldest_accuracy": steps[-1]["oldest_accuracy"],
        "birth_max_logit_difference": max(
            step["birth_max_logit_difference"] for step in steps),
        "routing": routing,
        "control_hijack_rate": hijack,
        "control_rows": control_rows,
    }


def _evaluate_global_overwrite(model, world, tasks, device, n_eval):
    first = model.compile_skill(world.instruction(tasks[0], device))
    expert_id = model.skill_cortex.birth(compiled=first)
    model.skill_cortex.commit(expert_id, confidence=0.0, mature=True)
    expert = model.skill_cortex.expert(expert_id)
    steps = []
    for task in tasks:
        compiled = model.compile_skill(world.instruction(task, device))
        expert.load_compiled(compiled)
        accuracies = [
            _accuracy(model, world, old_task, device, forced_id=expert_id, n=n_eval)
            for old_task in tasks[:len(steps) + 1]
        ]
        steps.append({
            "skills": len(accuracies),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "oldest_accuracy": accuracies[0],
            "newest_accuracy": accuracies[-1],
        })
    return {"steps": steps, "final_accuracy": steps[-1]["mean_accuracy"],
            "oldest_accuracy": steps[-1]["oldest_accuracy"]}


def _restart_child(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.restart_child + ".payload.pt", map_location="cpu", weights_only=True)
    world = ProcedureWorld(payload["modulus"], payload["id_parts"])
    model = persist.load_model(args.restart_child, device).eval()
    accuracies = [
        _accuracy(model, world, tuple(task), device, n=payload["n_eval"])
        for task in payload["tasks"]
    ]
    _atomic_json(args.restart_child + ".result.json", {
        "accuracy": sum(accuracies) / len(accuracies),
        "oldest_accuracy": accuracies[0],
        "experts": model.skill_cortex.manifest(),
    })


def _restart_check(model, world, tasks, args):
    checkpoint = str(Path(args.results).with_suffix(".restart.pt"))
    persist.save_model(checkpoint, model)
    persist.atomic_torch_save({
        "modulus": world.modulus, "id_parts": world.id_parts,
        "tasks": tasks, "n_eval": args.eval_inputs,
    }, checkpoint + ".payload.pt")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run([
        sys.executable, "-m", "fractal.exp_growing_cortex",
        "--restart_child", checkpoint,
    ], check=False, capture_output=True, text=True, env=env)
    if completed.returncode:
        return {"complete": False, "error": completed.stderr[-1000:]}
    try:
        with open(checkpoint + ".result.json", encoding="utf-8") as handle:
            return {"complete": True, **json.load(handle)}
    except (OSError, ValueError):
        return {"complete": False, "error": "restart child produced no valid result"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="growing_cortex_results.json")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--interpreter_steps", type=int, default=1200)
    parser.add_argument("--meta_tasks", type=int, default=4)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--eval_tasks", type=int, default=16)
    parser.add_argument("--eval_inputs", type=int, default=32)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--skill_rank", type=int, default=4)
    parser.add_argument("--router_threshold", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--report_every", type=int, default=50)
    parser.add_argument("--telemetry", default="",
                        help="optional live dashboard telemetry JSON")
    parser.add_argument(
        "--resume_checkpoint", default="",
        help="reuse a trained interpreter checkpoint and retrain only the skill compiler",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--restart_child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.restart_child:
        _restart_child(args)
        return
    if args.eval_tasks < 2 or args.meta_tasks < 2:
        parser.error("evaluation and meta batches require at least two tasks")
    if args.smoke:
        args.steps = 8
        args.interpreter_steps = 8
        args.meta_tasks = 2
        args.queries = 2
        args.eval_tasks = 3
        args.eval_inputs = 4
        args.n_embd = 32
        args.n_head = 4
        args.depth = 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    world = ProcedureWorld()
    all_tasks = world.tasks(args.seed)
    eval_tasks = all_tasks[-args.eval_tasks:]
    controls = all_tasks[-2 * args.eval_tasks:-args.eval_tasks]
    if world.id_parts ** 2 < args.meta_tasks:
        parser.error("not enough opaque names for the requested meta batch")

    if args.resume_checkpoint:
        model = persist.load_model(args.resume_checkpoint, device)
        expected = {
            "vocab_size": world.vocab_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "depth": args.depth,
            "n_scales": args.n_scales,
            "skill_rank": args.skill_rank,
        }
        mismatches = {
            name: (getattr(model.cfg, name), value)
            for name, value in expected.items()
            if getattr(model.cfg, name) != value
        }
        if mismatches:
            parser.error(f"resume checkpoint configuration mismatch: {mismatches}")
        model.skill_cortex.experts = torch.nn.ModuleList()
        model.skill_cortex.router_threshold = args.router_threshold
        model.skill_cortex.auto_route = False
        model.cfg.skill_router_threshold = args.router_threshold
        model.cfg.skill_auto_route = False
        model.train()
    else:
        cfg = Config(
            vocab_size=world.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
            depth=args.depth, n_scales=args.n_scales, chunk_size=16,
            growing_cortex=True, skill_rank=args.skill_rank,
            skill_compiler="full", skill_address_dim=args.n_embd,
            skill_router_threshold=args.router_threshold, skill_auto_route=False,
        )
        model = FractalLM(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    started = time.monotonic()
    history = []
    if args.resume_checkpoint:
        history.append({
            "stage": "resume",
            "checkpoint": os.path.basename(args.resume_checkpoint),
            "elapsed_seconds": time.monotonic() - started,
        })
    telemetry_prev = None

    def emit_telemetry(stage, step, loss):
        nonlocal telemetry_prev
        if not args.telemetry:
            return
        from fractal.train import _viz_telemetry
        payload, telemetry_prev = _viz_telemetry(model, telemetry_prev)
        payload.update({
            "schema_version": 1,
            "iter": step,
            "stage": stage,
            "loss": float(loss.detach()),
            "lr": args.lr,
            "batch": args.meta_tasks * args.queries,
            "block": 11 if stage == "interpreter" else 4,
            "params": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
            "active_params": round(model.parameter_counts()[1] / 1e6, 3),
            "n_embd": model.cfg.n_embd,
            "untie": bool(model.cfg.untie),
            "n_experts": model.cfg.n_experts,
            "moe_mode": model.cfg.moe_mode,
            "event_budget": model.cfg.event_budget,
            "event_share": model.event_share(),
            "growing_cortex": model.skill_cortex.snapshot(),
            "update_mode": f"growing-cortex-{stage}",
            "learning_signal": "gradient",
            "ckpt": os.path.basename(args.results),
            "text": "freshly permuted procedural meta-episode",
        })
        _atomic_json(args.telemetry, payload)

    interpreter_steps = 0 if args.resume_checkpoint else args.interpreter_steps
    for step in range(interpreter_steps):
        batch_tasks = world.sample_tasks(random, args.meta_tasks)
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(device == "cuda" and args.bf16))
        with amp:
            loss = _interpreter_step(
                model, world, batch_tasks, args.queries, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step % args.report_every == 0 or step == interpreter_steps - 1:
            emit_telemetry("interpreter", step, loss)
        optimizer.step()
        if step % args.report_every == 0 or step == interpreter_steps - 1:
            row = {
                "stage": "interpreter", "step": step,
                "loss": float(loss.detach()),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            _atomic_json(args.results, {
                "schema_version": 1, "complete": False,
                "seed": args.seed, "parameters": vars(args), "history": history,
            })

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.skill_cortex.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.skill_cortex.parameters(), lr=args.lr, betas=(0.9, 0.95))
    for step in range(args.steps):
        batch_tasks = world.sample_tasks(random, args.meta_tasks)
        negative_tasks = world.sample_negative_tasks(
            random, args.meta_tasks, batch_tasks)
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(device == "cuda" and args.bf16))
        with amp:
            loss, execution, route = _meta_step(
                model, world, batch_tasks, negative_tasks, args.queries, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step % args.report_every == 0 or step == args.steps - 1:
            emit_telemetry("compiler", step, loss)
        optimizer.step()
        if step % args.report_every == 0 or step == args.steps - 1:
            row = {
                "stage": "compiler", "step": step, "loss": float(loss.detach()),
                "execution_loss": float(execution), "route_loss": float(route),
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            _atomic_json(args.results, {
                "schema_version": 1, "complete": False,
                "seed": args.seed, "parameters": vars(args), "history": history,
            })

    model.eval()
    model.skill_cortex.auto_route = True
    interpreter_accuracy = _interpreter_accuracy(
        model, world, eval_tasks, device, args.eval_inputs)
    global_model = copy.deepcopy(model)
    growing = _evaluate_growing(
        model, world, eval_tasks, controls, device, args.eval_inputs)
    global_overwrite = _evaluate_global_overwrite(
        global_model, world, eval_tasks, device, args.eval_inputs)
    restart = _restart_check(model, world, eval_tasks, args)
    promotion = {
        "birth_is_function_preserving": growing["birth_max_logit_difference"] == 0.0,
        "interpreter_accuracy_80pct": interpreter_accuracy >= 0.80,
        "compiled_skill_accuracy_60pct": growing["compiled_accuracy"] >= 0.60,
        "sequential_accuracy_70pct": growing["final_accuracy"] >= 0.70,
        "oldest_skill_50pct": growing["oldest_accuracy"] >= 0.50,
        "control_hijack_below_10pct": growing["control_hijack_rate"] < 0.10,
        "beats_global_overwrite_20pp": (
            growing["final_accuracy"] - global_overwrite["final_accuracy"] >= 0.20),
        "restart_complete": bool(restart.get("complete")),
        "restart_accuracy_70pct": restart.get("accuracy", 0.0) >= 0.70,
    }
    payload = {
        "schema_version": 1,
        "complete": True,
        "seed": args.seed,
        "device": device,
        "elapsed_seconds": time.monotonic() - started,
        "parameters": vars(args),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "active_parameters": model.parameter_counts()[1],
        "history": history,
        "interpreter_accuracy": interpreter_accuracy,
        "growing": growing,
        "global_overwrite": global_overwrite,
        "restart": restart,
        "promotion": promotion,
        "mechanism_pass": all(promotion.values()),
        "final_expert_manifest": model.skill_cortex.manifest(),
        "conclusion": (
            "The append-only meta-compiled skill mechanism passed every declared synthetic gate."
            if all(promotion.values()) else
            "At least one declared synthetic growth or proceduralization gate failed."
        ),
    }
    _atomic_json(args.results, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
