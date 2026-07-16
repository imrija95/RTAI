"""Final falsification runner for Natural Cortex chat and confirmed local skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch

from fractal import chat_format as cf
from fractal.natural_runtime import load_runtime
from fractal.natural_train import _atomic_json


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _score(text: str, accepted: list[str]) -> bool:
    value = _normalize(text)
    return any(_normalize(candidate) in value for candidate in accepted)


def _paraphrase_score(runtime, expert_id: int | None, rows: list[dict]) -> tuple[float, list[dict]]:
    results = []
    for row in rows:
        output = runtime._greedy_reply(str(row["user"]), expert_id)
        passed = _score(output, [str(value) for value in row.get("accepted") or []])
        results.append({"user": row["user"], "output": output, "passed": passed})
    return sum(row["passed"] for row in results) / max(len(results), 1), results


@torch.no_grad()
def _chat_outputs(runtime, rows: list[dict], max_new: int) -> dict:
    end_id = runtime.tokenizer.token_to_id(cf.END)
    leakage_ids = {
        runtime.tokenizer.token_to_id(marker)
        for marker in (cf.USER, cf.SYSTEM, cf.TOOL_RESULT)
        if runtime.tokenizer.token_to_id(marker) is not None
    }
    results = []
    for row in rows:
        prime = runtime.tokenizer.encode(
            f"{cf.USER}\n{row['prompt']}\n{cf.ASSISTANT}\n").ids
        states = runtime.model.init_states(1, runtime.device)
        with runtime.model.skill_cortex.suspend():
            logits, states = runtime.model.forward_stream(
                torch.tensor([prime], device=runtime.device), states)
            generated = []
            terminated = leaked = False
            for _ in range(max_new):
                token = int(logits[:, -1].argmax(dim=-1))
                if token == end_id:
                    terminated = True
                    break
                if token in leakage_ids:
                    leaked = True
                    break
                generated.append(token)
                logits, states = runtime.model.forward_stream(
                    torch.tensor([[token]], device=runtime.device), states)
        results.append({
            "prompt": row["prompt"],
            "output": runtime.tokenizer.decode(generated).strip(),
            "terminated": terminated,
            "role_leakage": leaked,
            "relevant": row.get("relevant"),
        })
    terminated = sum(row["terminated"] for row in results)
    leaked = sum(row["role_leakage"] for row in results)
    return {
        "prompts": len(results),
        "terminated": terminated,
        "role_leakage": leaked,
        "passed": terminated >= 18 and leaked == 0,
        "outputs": results,
    }


@torch.no_grad()
def _anchor_loss(runtime, expert_id: int | None, anchors: list[dict]) -> float:
    losses = []
    cortex = runtime.model.skill_cortex
    context = cortex.force(expert_id) if expert_id is not None else cortex.suspend()
    with context:
        for row in anchors:
            pieces = cf.render_pieces([
                ("user", row["user"]),
                ("assistant", row["assistant"]),
            ])
            ids, mask = [], []
            for text, trainable in pieces:
                encoded = runtime.tokenizer.encode(text, add_special_tokens=False).ids
                ids.extend(encoded)
                mask.extend([1.0 if trainable else 0.0] * len(encoded))
            sequence = torch.tensor([ids], dtype=torch.long, device=runtime.device)
            weights = torch.tensor([mask[1:]], dtype=torch.float32, device=runtime.device)
            loss = runtime.model(
                sequence[:, :-1], targets=sequence[:, 1:], loss_weight=weights)[1]
            losses.append(float(loss))
    return sum(losses) / max(len(losses), 1)


def validate_spec(spec: dict) -> None:
    if len(spec.get("chat_prompts") or []) != 20:
        raise ValueError("final evaluation requires exactly 20 chat prompts")
    if not spec.get("anchors"):
        raise ValueError("final evaluation requires at least one anchor conversation")
    if len(spec.get("controls") or []) < 10:
        raise ValueError("final evaluation requires at least ten unrelated control prompts")
    skills = spec.get("skills") or []
    if len(skills) != 10:
        raise ValueError("final evaluation requires exactly ten skills")
    for skill in skills:
        if not 1 <= len(skill.get("demonstrations") or []) <= 3:
            raise ValueError("each skill requires one to three demonstrations")
        if len(skill.get("paraphrases") or []) != 5:
            raise ValueError("each skill requires exactly five paraphrase tests")
        if any(not row.get("accepted") for row in skill["paraphrases"]):
            raise ValueError("every paraphrase requires at least one accepted answer string")


def run(args) -> dict:
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_spec(spec)
    runtime_root = Path(args.runtime_dir)
    runtime_root.mkdir(parents=True, exist_ok=True)
    bank_dir = runtime_root / "skill-bank"
    state_path = runtime_root / "fast-weights.pt"
    runtime = load_runtime(
        args.checkpoint, args.tokenizer, str(bank_dir), str(state_path), args.device)
    deadline = time.perf_counter() + args.max_hours * 3600

    termination = _chat_outputs(runtime, spec["chat_prompts"], args.max_new)
    relevance_labels = [row.get("relevant") for row in spec["chat_prompts"]]
    manual_complete = all(isinstance(value, bool) for value in relevance_labels)
    relevant = sum(bool(value) for value in relevance_labels) if manual_complete else None
    anchors = spec.get("anchors") or []
    baseline_anchor = _anchor_loss(runtime, None, anchors)

    skills = []
    oldest_best = 0.0
    for index, skill in enumerate(spec["skills"]):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Natural Cortex final evaluation exceeded its time budget")
        baseline, baseline_rows = _paraphrase_score(runtime, None, skill["paraphrases"])
        started = time.perf_counter()
        teaching = runtime.teach(
            skill["name"], skill["synopsis"], skill["demonstrations"],
            confirmed=True, anchors=anchors, steps=args.teaching_steps, lr=args.teaching_lr)
        runtime.rate(5)
        elapsed = time.perf_counter() - started
        expert_id = teaching["expert_id"]
        learned, learned_rows = _paraphrase_score(runtime, expert_id, skill["paraphrases"])
        anchor = _anchor_loss(runtime, expert_id, anchors)
        row = {
            "index": index,
            "expert_id": expert_id,
            "name": skill["name"],
            "baseline_score": baseline,
            "learned_score": learned,
            "improvement": learned - baseline,
            "teaching_seconds": elapsed,
            "anchor_loss": anchor,
            "anchor_regression": (
                (anchor - baseline_anchor) / max(baseline_anchor, 1e-9)),
            "baseline_outputs": baseline_rows,
            "learned_outputs": learned_rows,
            "teaching": teaching,
        }
        skills.append(row)
        if index == 0:
            oldest_best = learned

    address_examples = []
    for row, skill in zip(skills, spec["skills"]):
        address_examples.append({"text": skill["synopsis"], "expert_id": row["expert_id"]})
        address_examples.extend({
            "text": paraphrase["user"], "expert_id": row["expert_id"]
        } for paraphrase in skill["paraphrases"])
    calibration = runtime.calibrate_addresses(
        address_examples, steps=args.address_steps, lr=args.address_lr)
    if time.perf_counter() >= deadline:
        raise TimeoutError("Natural Cortex final evaluation exceeded its time budget")

    controls = []
    for prompt in spec.get("controls") or []:
        proposal = runtime.propose_skill(str(prompt))
        controls.append({"prompt": prompt, **proposal})
    hijack = sum(row["expert_id"] is not None for row in controls) / max(len(controls), 1)

    oldest_final, oldest_rows = _paraphrase_score(
        runtime, skills[0]["expert_id"], spec["skills"][0]["paraphrases"])
    restart_integrity = runtime.restart_verification(args.checkpoint, str(state_path))
    restarted = load_runtime(
        args.checkpoint, args.tokenizer, str(bank_dir), str(state_path), args.device)
    restart_scores = []
    for row, skill in zip(skills, spec["skills"]):
        score, _outputs = _paraphrase_score(restarted, row["expert_id"], skill["paraphrases"])
        restart_scores.append(score)
    pre_restart_mean = sum(row["learned_score"] for row in skills) / len(skills)
    restart_mean = sum(restart_scores) / len(restart_scores)

    gates = {
        "chat_termination_18_of_20_no_leakage": termination["passed"],
        "manual_relevance_complete": manual_complete,
        "manual_relevance_12_of_20": manual_complete and relevant >= 12,
        "seven_of_ten_improve_30pp": (
            sum(row["improvement"] >= 0.30 for row in skills) >= 7),
        "control_hijack_below_10pct": hijack < 0.10,
        "anchor_loss_within_5pct": all(row["anchor_regression"] <= 0.05 for row in skills),
        "restart_within_2pp": all(
            abs(after - before["learned_score"]) <= 0.02
            for before, after in zip(skills, restart_scores)),
        "teaching_within_60_seconds": all(row["teaching_seconds"] <= 60.0 for row in skills),
        "oldest_retains_70pct": oldest_final >= 0.70 * oldest_best,
        "restart_integrity": restart_integrity["complete"],
    }
    report = {
        "schema_version": 1,
        "checkpoint": Path(args.checkpoint).name,
        "termination": termination,
        "manual_relevance": {
            "complete": manual_complete,
            "relevant": relevant,
            "total": 20,
        },
        "baseline_anchor_loss": baseline_anchor,
        "address_calibration": calibration,
        "skills": skills,
        "controls": controls,
        "control_hijack_rate": hijack,
        "oldest_best_score": oldest_best,
        "oldest_final_score": oldest_final,
        "oldest_final_outputs": oldest_rows,
        "restart": {
            "integrity": restart_integrity,
            "pre_restart_mean": pre_restart_mean,
            "post_restart_mean": restart_mean,
            "scores": restart_scores,
        },
        "gates": gates,
        "accepted": all(gates.values()),
    }
    _atomic_json(Path(args.output), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--teaching-steps", type=int, default=64)
    parser.add_argument("--teaching-lr", type=float, default=1e-2)
    parser.add_argument("--address-steps", type=int, default=200)
    parser.add_argument("--address-lr", type=float, default=1e-3)
    parser.add_argument("--max-hours", type=float, default=3.0)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"accepted": report["accepted"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
