"""Bounded Predictive Event Algebra falsification screen.

The runner uses identical held-out associative episodes for four inference-time arms: ordinary
delta memory, surprise-driven delayed credit, explicit rating-5 credit, and W0 consolidation into
a fresh session. It does not claim natural-language understanding; it cheaply tests the required
mechanism (causal binding, targeted credit, and durable consolidation) before a longer run.

Example:
  uv run python -m fractal.exp_event_algebra --ckpt MODEL.pt --budget_minutes 90 \
      --report_minutes 10 --results event_algebra_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from fractal import feedback, persist, tokenizer as tk
from fractal.recall import PREFIXES, RecallGen


def _atomic_json(path, payload):
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _episode(recall, distance, n_facts):
    template = random.choice(PREFIXES)
    names = random.sample(recall.names, min(n_facts, len(recall.names)))
    values = [random.choice(recall.test_vals) for _ in names]
    facts = []
    for name, value in zip(names, values):
        facts += recall._e(template.format(n=name)) + [value]
    selected = random.randrange(len(names))
    query = recall._filler(distance) + recall._e(template.format(n=names[selected]))
    return facts, query, values[selected]


@torch.no_grad()
def _stream(model, states, ids, device):
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    return model.forward_stream(tensor, states)


def _prediction_metrics(logits, answer):
    scores = logits[0, -1].float()
    target = scores[int(answer)]
    rank = int((scores > target).sum()) + 1
    probability = float(scores.softmax(dim=-1)[int(answer)])
    return int(rank == 1), {
        "answer_probability": probability,
        "answer_rank": float(rank),
        "reciprocal_rank": 1.0 / rank,
    }


@torch.no_grad()
def evaluate_episode(model, facts, query, answer, anchors, controls, device, args):
    outcomes, updates, metrics = {}, {}, {}
    for arm in ("baseline", "autonomous", "feedback"):
        states = model.init_states(1, device)
        logits, states = _stream(model, states, facts, device)
        if arm == "autonomous":
            credit = feedback.observed_surprise(logits, facts)
            updates[arm] = feedback.apply_to_state(
                states, states, credit, lr=args.autonomous_lr,
                max_fraction=args.fast_max_fraction)
        elif arm == "feedback":
            updates[arm] = feedback.apply_to_state(
                states, states, 1.0, lr=args.feedback_lr,
                max_fraction=args.fast_max_fraction)
        logits, states = _stream(model, states, query, device)
        outcomes[arm], metrics[arm] = _prediction_metrics(logits, answer)

    snapshot = feedback.w0_snapshot(model)
    evidence = model.init_states(1, device)
    _, evidence = _stream(model, evidence, facts, device)
    updates["w0"] = feedback.consolidate_w0(
        model, evidence, 1.0, lr=args.w0_lr,
        max_fraction=args.w0_max_fraction)
    fresh = model.init_states(1, device)
    logits, fresh = _stream(model, fresh, query, device)
    outcomes["w0"], metrics["w0"] = _prediction_metrics(logits, answer)
    feedback.restore_w0(model, snapshot)

    teaching = feedback.teach_w0(
        model, query, answer, device, lr=args.teacher_lr,
        steps=args.teacher_steps, max_fraction=args.teacher_max_fraction,
        anchor_prompts=anchors, anchor_weight=args.teacher_anchor_weight,
        anchor_mode=args.teacher_anchor_mode, scope=args.teacher_scope)
    fresh = model.init_states(1, device)
    logits, fresh = _stream(model, fresh, query, device)
    outcomes["w0_teacher"], metrics["w0_teacher"] = _prediction_metrics(logits, answer)
    metrics["w0_teacher"]["initial_loss"] = teaching.initial_loss
    metrics["w0_teacher"]["final_loss"] = teaching.final_loss
    metrics["w0_teacher"]["anchor_kl"] = teaching.anchor_kl
    metrics["w0_teacher"]["preserve_kl"] = teaching.preserve_kl
    control_answer_probability = 0.0
    taught_token_control_probability = 0.0
    control_accuracy = 0.0
    taught_token_control_rate = 0.0
    for control_query, control_answer in controls:
        control_state = model.init_states(1, device)
        control_logits, control_state = _stream(model, control_state, control_query, device)
        scores = control_logits[0, -1].float()
        probabilities = scores.softmax(dim=-1)
        prediction = int(scores.argmax())
        control_answer_probability += float(probabilities[int(control_answer)])
        taught_token_control_probability += float(probabilities[int(answer)])
        control_accuracy += float(prediction == int(control_answer))
        taught_token_control_rate += float(prediction == int(answer))
    denominator = max(len(controls), 1)
    metrics["w0_teacher"].update({
        "control_answer_probability": control_answer_probability / denominator,
        "taught_token_control_probability": taught_token_control_probability / denominator,
        "control_accuracy": control_accuracy / denominator,
        "taught_token_control_rate": taught_token_control_rate / denominator,
    })
    updates["w0_teacher"] = teaching.update_norm
    feedback.restore_w0(model, snapshot)
    return outcomes, updates, metrics


def _restart_child(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, device).eval()
    feedback.enable(model)
    states = persist.load_states(args.restart_child, device)
    payload = torch.load(args.restart_child + ".query.pt", map_location="cpu", weights_only=True)
    logits, _ = _stream(model, states, payload["query"], device)
    _atomic_json(args.restart_child + ".result.json",
                 {"correct": int(logits[0, -1].argmax()) == int(payload["answer"])})


def _w0_restart_child(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, device).eval()
    feedback.enable(model)
    feedback.load_w0(args.restart_w0_child, model)
    payload = torch.load(args.restart_w0_child + ".query.pt", map_location="cpu", weights_only=True)
    states = model.init_states(1, device)
    logits, _states = _stream(model, states, payload["query"], device)
    correct, metrics = _prediction_metrics(logits, int(payload["answer"]))
    _atomic_json(args.restart_w0_child + ".result.json", {"correct": bool(correct), **metrics})


def process_restart(model, recall, args, device):
    facts, query, answer = _episode(recall, args.distance, args.n_facts)
    states = model.init_states(1, device)
    _, states = _stream(model, states, facts, device)
    feedback.apply_to_state(states, states, 1.0)
    state_path = str(Path(args.results).with_suffix(".restart.pt"))
    persist.save_states(state_path, states)
    persist.atomic_torch_save({"query": query, "answer": answer}, state_path + ".query.pt")
    command = [sys.executable, "-m", "fractal.exp_event_algebra", "--ckpt", args.ckpt,
               "--restart_child", state_path]
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=child_env)
    if completed.returncode != 0:
        return {"correct": False, "error": completed.stderr[-1000:]}
    try:
        with open(state_path + ".result.json", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"correct": False, "error": "restart child produced no valid result"}


def w0_process_restart(model, recall, args, device):
    _facts, query, answer = _episode(recall, args.distance, args.n_facts)
    anchors = []
    for _ in range(args.teacher_anchors):
        _anchor_facts, anchor_query, _anchor_answer = _episode(recall, args.distance, args.n_facts)
        anchors.append(anchor_query)
    snapshot = feedback.w0_snapshot(model)
    teaching = feedback.teach_w0(
        model, query, answer, device, lr=args.teacher_lr,
        steps=args.teacher_steps, max_fraction=args.teacher_max_fraction,
        anchor_prompts=anchors, anchor_weight=args.teacher_anchor_weight,
        anchor_mode=args.teacher_anchor_mode, scope=args.teacher_scope)
    overlay_path = str(Path(args.results).with_suffix(".teacher-w0.pt"))
    feedback.save_w0(overlay_path, model)
    feedback.restore_w0(model, snapshot)
    persist.atomic_torch_save({"query": query, "answer": answer}, overlay_path + ".query.pt")
    command = [sys.executable, "-m", "fractal.exp_event_algebra", "--ckpt", args.ckpt,
               "--restart_w0_child", overlay_path]
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=child_env)
    if completed.returncode != 0:
        return {"correct": False, "error": completed.stderr[-1000:],
                "teaching_update_norm": teaching.update_norm}
    try:
        with open(overlay_path + ".result.json", encoding="utf-8") as handle:
            result = json.load(handle)
        result["teaching_update_norm"] = teaching.update_norm
        return result
    except (OSError, ValueError):
        return {"correct": False, "error": "W0 restart child produced no valid result",
                "teaching_update_norm": teaching.update_norm}


def _decision(accuracy, restart, w0_restart):
    return {
        "autonomous_lift_15pp": accuracy["autonomous"] >= accuracy["baseline"] + 0.15,
        "feedback_lift_15pp": accuracy["feedback"] >= accuracy["baseline"] + 0.15,
        "feedback_accuracy_50pct": accuracy["feedback"] >= 0.50,
        "w0_fresh_accuracy_50pct": accuracy["w0"] >= 0.50,
        "w0_teacher_lift_15pp": accuracy["w0_teacher"] >= accuracy["w0"] + 0.15,
        "w0_teacher_accuracy_50pct": accuracy["w0_teacher"] >= 0.50,
        "process_restart": bool(restart.get("correct")),
        "w0_process_restart": bool(w0_restart.get("correct")),
    }


def _parameters(args):
    return {
        "autonomous_lr": args.autonomous_lr,
        "feedback_lr": args.feedback_lr,
        "w0_lr": args.w0_lr,
        "fast_max_fraction": args.fast_max_fraction,
        "w0_max_fraction": args.w0_max_fraction,
        "eligibility_decay": args.eligibility_decay,
        "teacher_lr": args.teacher_lr,
        "teacher_steps": args.teacher_steps,
        "teacher_max_fraction": args.teacher_max_fraction,
        "teacher_controls": args.teacher_controls,
        "teacher_anchors": args.teacher_anchors,
        "teacher_anchor_weight": args.teacher_anchor_weight,
        "teacher_anchor_mode": args.teacher_anchor_mode,
        "teacher_scope": args.teacher_scope,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="fractal_tokenizer.json")
    parser.add_argument("--val_bin", default="fractal_data/val.bin")
    parser.add_argument("--results", default="event_algebra_results.json")
    parser.add_argument("--budget_minutes", type=float, default=90.0)
    parser.add_argument("--report_minutes", type=float, default=10.0)
    parser.add_argument("--max_trials", type=int, default=0,
                        help="optional deterministic trial cap; zero runs until the time budget")
    parser.add_argument("--distance", type=int, default=48)
    parser.add_argument("--n_facts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--autonomous_lr", type=float, default=0.15)
    parser.add_argument("--feedback_lr", type=float, default=0.25)
    parser.add_argument("--w0_lr", type=float, default=0.01)
    parser.add_argument("--fast_max_fraction", type=float, default=0.10)
    parser.add_argument("--w0_max_fraction", type=float, default=0.01)
    parser.add_argument("--eligibility_decay", type=float, default=0.95)
    parser.add_argument("--teacher_lr", type=float, default=0.01)
    parser.add_argument("--teacher_steps", type=int, default=1)
    parser.add_argument("--teacher_max_fraction", type=float, default=0.01)
    parser.add_argument("--teacher_controls", type=int, default=3)
    parser.add_argument("--teacher_anchors", type=int, default=3)
    parser.add_argument("--teacher_anchor_weight", type=float, default=0.0)
    parser.add_argument("--teacher_anchor_mode", choices=("kl", "negative", "margin"), default="kl")
    parser.add_argument("--teacher_scope", choices=("all", "permanent"), default="all")
    parser.add_argument("--restart_child", default="", help=argparse.SUPPRESS)
    parser.add_argument("--restart_w0_child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.restart_child:
        _restart_child(args)
        return
    if args.restart_w0_child:
        _w0_restart_child(args)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, device).eval()
    if not 0.0 <= args.eligibility_decay <= 1.0:
        parser.error("--eligibility_decay must be in [0, 1]")
    for value, name in ((args.autonomous_lr, "--autonomous_lr"),
                        (args.feedback_lr, "--feedback_lr"),
                        (args.w0_lr, "--w0_lr"),
                        (args.fast_max_fraction, "--fast_max_fraction"),
                        (args.w0_max_fraction, "--w0_max_fraction"),
                        (args.teacher_lr, "--teacher_lr"),
                        (args.teacher_max_fraction, "--teacher_max_fraction"),
                        (args.teacher_anchor_weight, "--teacher_anchor_weight")):
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if args.teacher_steps < 1:
        parser.error("--teacher_steps must be positive")
    if args.teacher_controls < 1:
        parser.error("--teacher_controls must be positive")
    if args.teacher_anchors < 1:
        parser.error("--teacher_anchors must be positive")
    model.cfg.eligibility_decay = args.eligibility_decay
    for block in model.blocks:
        block.unit.eligibility_decay = args.eligibility_decay
    feedback.enable(model)
    tokenizer = tk.load(args.tokenizer)
    recall = RecallGen(tokenizer, args.val_bin, seed=args.seed)
    counts = {arm: 0 for arm in ("baseline", "autonomous", "feedback", "w0", "w0_teacher")}
    update_sum = {arm: 0.0 for arm in ("autonomous", "feedback", "w0", "w0_teacher")}
    metric_sum = {
        arm: {name: 0.0 for name in ("answer_probability", "answer_rank", "reciprocal_rank")}
        for arm in counts
    }
    metric_sum["w0_teacher"].update({
        "initial_loss": 0.0, "final_loss": 0.0, "anchor_kl": 0.0, "preserve_kl": 0.0,
    })
    metric_sum["w0_teacher"].update({
        "control_answer_probability": 0.0,
        "taught_token_control_probability": 0.0,
        "control_accuracy": 0.0,
        "taught_token_control_rate": 0.0,
    })
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    next_report = started + args.report_minutes * 60
    trials = 0

    while time.monotonic() < deadline and (not args.max_trials or trials < args.max_trials):
        facts, query, answer = _episode(recall, args.distance, args.n_facts)
        anchors = []
        for _ in range(args.teacher_anchors):
            _anchor_facts, anchor_query, _anchor_answer = _episode(
                recall, args.distance, args.n_facts)
            anchors.append(anchor_query)
        controls = []
        for _ in range(args.teacher_controls):
            _control_facts, control_query, control_answer = _episode(
                recall, args.distance, args.n_facts)
            controls.append((control_query, control_answer))
        outcomes, updates, metrics = evaluate_episode(
            model, facts, query, answer, anchors, controls, device, args)
        trials += 1
        for arm, passed in outcomes.items():
            counts[arm] += int(passed)
        for arm, norm in updates.items():
            update_sum[arm] += float(norm)
        for arm, values in metrics.items():
            for name, value in values.items():
                metric_sum[arm][name] += float(value)
        now = time.monotonic()
        if now >= next_report or (args.max_trials and trials == args.max_trials):
            elapsed = (now - started) / 60
            payload = {
                "schema_version": 1, "complete": False, "elapsed_minutes": elapsed,
                "trials": trials, "distance": args.distance, "n_facts": args.n_facts,
                "accuracy": {arm: counts[arm] / trials for arm in counts},
                "mean_update_norm": {arm: update_sum[arm] / trials for arm in update_sum},
                "mean_metrics": {
                    arm: {name: value / trials for name, value in values.items()}
                    for arm, values in metric_sum.items()
                },
                "checkpoint": os.path.basename(args.ckpt), "seed": args.seed,
                "parameters": _parameters(args),
            }
            _atomic_json(args.results, payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            next_report += args.report_minutes * 60

    restart = process_restart(model, recall, args, device) if trials else {"correct": False}
    w0_restart = w0_process_restart(model, recall, args, device) if trials else {"correct": False}
    accuracy = {arm: counts[arm] / max(trials, 1) for arm in counts}
    decision = _decision(accuracy, restart, w0_restart)
    payload = {
        "schema_version": 1, "complete": True,
        "elapsed_minutes": (time.monotonic() - started) / 60, "trials": trials,
        "distance": args.distance, "n_facts": args.n_facts,
        "accuracy": accuracy,
        "mean_update_norm": {arm: update_sum[arm] / max(trials, 1) for arm in update_sum},
        "mean_metrics": {
            arm: {name: value / max(trials, 1) for name, value in values.items()}
            for arm, values in metric_sum.items()
        },
        "checkpoint": os.path.basename(args.ckpt), "seed": args.seed,
        "parameters": _parameters(args),
        "process_restart": restart, "decision": decision,
        "w0_process_restart": w0_restart,
        "mechanism_pass": all(decision.values()),
    }
    _atomic_json(args.results, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
