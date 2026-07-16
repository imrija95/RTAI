"""Sequential persistent-memory interference screen for vector-taught W0.

The experiment teaches several held-out prompt/answer associations into the same persistent W0
overlay. After every write it evaluates all previously taught prompts from fresh runtime state,
plus unseen control prompts. This distinguishes durable associative memory from one-shot success
that catastrophically overwrites older memories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import torch

from fractal import feedback, persist, tokenizer as tk
from fractal.exp_event_algebra import _atomic_json, _episode, _prediction_metrics, _stream
from fractal.recall import RecallGen


@torch.no_grad()
def _evaluate(model, memories, controls, device):
    memory_rows = []
    learned_answers = {int(answer) for _query, answer in memories}
    for query, answer in memories:
        states = model.init_states(1, device)
        logits, _states = _stream(model, states, query, device)
        correct, metrics = _prediction_metrics(logits, answer)
        prediction = int(logits[0, -1].argmax())
        memory_rows.append({
            "correct": bool(correct),
            "answer": int(answer),
            "prediction": prediction,
            **metrics,
            "wrong_learned_answer": bool(prediction in learned_answers and prediction != int(answer)),
        })

    newest_answer = int(memories[-1][1])
    control_rows = []
    for query, answer in controls:
        states = model.init_states(1, device)
        logits, _states = _stream(model, states, query, device)
        scores = logits[0, -1].float()
        probabilities = scores.softmax(dim=-1)
        prediction = int(scores.argmax())
        control_rows.append({
            "correct": prediction == int(answer),
            "prediction": prediction,
            "newest_answer_probability": float(probabilities[newest_answer]),
            "newest_answer_hijack": prediction == newest_answer,
            "any_learned_answer_hijack": prediction in learned_answers,
        })

    return {
        "memory_accuracy": sum(row["correct"] for row in memory_rows) / len(memory_rows),
        "oldest_memory_correct": memory_rows[0]["correct"],
        "newest_memory_correct": memory_rows[-1]["correct"],
        "mean_memory_probability": sum(row["answer_probability"] for row in memory_rows) / len(memory_rows),
        "mean_memory_rank": sum(row["answer_rank"] for row in memory_rows) / len(memory_rows),
        "wrong_learned_answer_rate": (
            sum(row["wrong_learned_answer"] for row in memory_rows) / len(memory_rows)
        ),
        "control_accuracy": sum(row["correct"] for row in control_rows) / len(control_rows),
        "newest_answer_control_probability": (
            sum(row["newest_answer_probability"] for row in control_rows) / len(control_rows)
        ),
        "newest_answer_hijack_rate": (
            sum(row["newest_answer_hijack"] for row in control_rows) / len(control_rows)
        ),
        "any_learned_answer_hijack_rate": (
            sum(row["any_learned_answer_hijack"] for row in control_rows) / len(control_rows)
        ),
        "memory_rows": memory_rows,
    }


def _restart_child(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, device).eval()
    feedback.enable(model)
    feedback.load_w0(args.restart_child, model)
    payload = torch.load(args.restart_child + ".payload.pt", map_location="cpu", weights_only=True)
    result = _evaluate(model, payload["memories"], payload["controls"], device)
    _atomic_json(args.restart_child + ".result.json", result)


def _restart_check(model, memories, controls, args):
    overlay = str(Path(args.results).with_suffix(".w0.pt"))
    feedback.save_w0(overlay, model)
    persist.atomic_torch_save({"memories": memories, "controls": controls}, overlay + ".payload.pt")
    command = [
        sys.executable, "-m", "fractal.exp_w0_interference",
        "--ckpt", args.ckpt, "--restart_child", overlay,
    ]
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=child_env)
    if completed.returncode != 0:
        return {"complete": False, "error": completed.stderr[-1000:]}
    try:
        with open(overlay + ".result.json", encoding="utf-8") as handle:
            return {"complete": True, **json.load(handle)}
    except (OSError, ValueError):
        return {"complete": False, "error": "restart child produced no valid result"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="fractal_tokenizer.json")
    parser.add_argument("--val_bin", default="fractal_data/val.bin")
    parser.add_argument("--results", default="w0_interference_results.json")
    parser.add_argument("--n_memories", type=int, default=8)
    parser.add_argument("--distance", type=int, default=0)
    parser.add_argument("--n_facts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--teacher_lr", type=float, default=0.3)
    parser.add_argument("--teacher_steps", type=int, default=32)
    parser.add_argument("--teacher_max_fraction", type=float, default=0.5)
    parser.add_argument("--teacher_anchor_weight", type=float, default=5.0)
    parser.add_argument("--teacher_anchors", type=int, default=8)
    parser.add_argument("--teacher_controls", type=int, default=16)
    parser.add_argument("--preserve_limit", type=int, default=0)
    parser.add_argument("--preserve_weight", type=float, default=0.0)
    parser.add_argument("--restart_child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.restart_child:
        _restart_child(args)
        return
    if args.n_memories < 2:
        parser.error("--n_memories must be at least 2")
    if args.teacher_anchors < 1 or args.teacher_controls < 1:
        parser.error("anchor and control counts must be positive")
    if args.preserve_limit < 0 or args.preserve_weight < 0:
        parser.error("preserve limit and weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, device).eval()
    feedback.enable(model)
    tokenizer = tk.load(args.tokenizer)
    recall = RecallGen(tokenizer, args.val_bin, seed=args.seed)

    anchors = []
    for _ in range(args.teacher_anchors):
        _facts, query, _answer = _episode(recall, args.distance, args.n_facts)
        anchors.append(query)
    controls = []
    for _ in range(args.teacher_controls):
        _facts, query, answer = _episode(recall, args.distance, args.n_facts)
        controls.append((query, answer))

    memories = []
    steps = []
    for index in range(args.n_memories):
        _facts, query, answer = _episode(recall, args.distance, args.n_facts)
        preserve = [old_query for old_query, _old_answer in memories]
        forbidden_answers = [old_answer for _old_query, old_answer in memories] + [answer]
        if args.preserve_limit and len(preserve) > args.preserve_limit:
            preserve = random.sample(preserve, args.preserve_limit)
        teaching = feedback.teach_w0(
            model, query, answer, device,
            lr=args.teacher_lr, steps=args.teacher_steps,
            max_fraction=args.teacher_max_fraction,
            anchor_prompts=anchors, anchor_weight=args.teacher_anchor_weight,
            anchor_mode="margin", scope="permanent",
            preserve_prompts=preserve, preserve_weight=args.preserve_weight,
            anchor_forbidden_ids=forbidden_answers,
        )
        memories.append((query, answer))
        evaluation = _evaluate(model, memories, controls, device)
        step = {
            "memory_count": len(memories),
            "teaching": {
                "update_norm": teaching.update_norm,
                "initial_loss": teaching.initial_loss,
                "final_loss": teaching.final_loss,
                "anchor_kl": teaching.anchor_kl,
                "preserve_kl": teaching.preserve_kl,
            },
            **evaluation,
        }
        steps.append(step)
        payload = {
            "schema_version": 1,
            "complete": False,
            "checkpoint": os.path.basename(args.ckpt),
            "seed": args.seed,
            "parameters": {
                "n_memories": args.n_memories,
                "distance": args.distance,
                "n_facts": args.n_facts,
                "teacher_lr": args.teacher_lr,
                "teacher_steps": args.teacher_steps,
                "teacher_max_fraction": args.teacher_max_fraction,
                "teacher_anchor_weight": args.teacher_anchor_weight,
                "teacher_anchors": args.teacher_anchors,
                "teacher_controls": args.teacher_controls,
                "preserve_limit": args.preserve_limit,
                "preserve_weight": args.preserve_weight,
                "protect_all_learned_answers": True,
            },
            "steps": steps,
        }
        _atomic_json(args.results, payload)
        print(json.dumps(step, sort_keys=True), flush=True)

    restart = _restart_check(model, memories, controls, args)
    final = steps[-1]
    payload = {
        "schema_version": 1,
        "complete": True,
        "checkpoint": os.path.basename(args.ckpt),
        "seed": args.seed,
        "parameters": {
            "n_memories": args.n_memories,
            "distance": args.distance,
            "n_facts": args.n_facts,
            "teacher_lr": args.teacher_lr,
            "teacher_steps": args.teacher_steps,
            "teacher_max_fraction": args.teacher_max_fraction,
            "teacher_anchor_weight": args.teacher_anchor_weight,
            "teacher_anchors": args.teacher_anchors,
            "teacher_controls": args.teacher_controls,
            "preserve_limit": args.preserve_limit,
            "preserve_weight": args.preserve_weight,
            "protect_all_learned_answers": True,
        },
        "steps": steps,
        "restart": restart,
        "promotion": {
            "final_memory_accuracy_50pct": final["memory_accuracy"] >= 0.50,
            "oldest_memory_survives": bool(final["oldest_memory_correct"]),
            "any_learned_answer_hijack_below_10pct": (
                final["any_learned_answer_hijack_rate"] < 0.10
            ),
            "restart_complete": bool(restart.get("complete")),
            "restart_memory_accuracy_50pct": restart.get("memory_accuracy", 0.0) >= 0.50,
        },
    }
    payload["mechanism_pass"] = all(payload["promotion"].values())
    _atomic_json(args.results, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
