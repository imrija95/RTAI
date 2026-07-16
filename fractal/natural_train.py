"""Production trainer for the Natural Cortex dense and top-1 MoE presets.

This entry point intentionally exposes only the reviewed production surface. Refuted experimental
paths remain available in archived experiment runners but cannot be enabled here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from fractal import chat_format as cf
from fractal import persist
from fractal import tokenizer as tk
from fractal.model import Config, FractalLM
from fractal.natural_data import NATURAL_SOURCES, NaturalCorpus
from fractal.plasticity import PlasticityController


DEFAULT_SEED = 20260716


def natural_config(variant: str) -> Config:
    if variant not in ("dense", "moe"):
        raise ValueError(f"unknown Natural Cortex variant: {variant}")
    return Config(
        vocab_size=24_000,
        n_embd=1792,
        n_head=28,
        depth=8,
        n_scales=2,
        tau0=16.0,
        rho=4.0,
        chunk_size=64,
        mlp_ratio=2,
        dropout=0.0,
        high_pass_keys=False,
        selective=False,
        untie=False,
        n_experts=4 if variant == "moe" else 1,
        moe_mode="top1" if variant == "moe" else "soft",
        event_budget=1.0,
        event_algebra=False,
        growing_cortex=True,
        skill_rank=8,
        skill_compiler="none",
        skill_address_dim=64,
        skill_router_threshold=0.55,
        skill_auto_route=False,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_dense_stem(dense: FractalLM, moe: FractalLM) -> None:
    """Make the A/B pair share all base tensors and clone the dense MLP into every MoE expert."""
    dense_state = dense.state_dict()
    moe_state = moe.state_dict()
    for name, value in dense_state.items():
        if ".mlp.fc." in name or ".mlp.proj." in name:
            continue
        if name in moe_state and moe_state[name].shape == value.shape:
            moe_state[name].copy_(value)
    moe.load_state_dict(moe_state)
    for block_index, block in enumerate(moe.blocks):
        dense_mlp = dense.blocks[block_index].mlp
        for expert in block.mlp.experts:
            expert.load_state_dict(dense_mlp.state_dict())


def _source_weights(anneal_chat: bool) -> tuple[list[int], np.ndarray]:
    if anneal_chat:
        raw = [spec for spec in NATURAL_SOURCES if spec.kind != "chat"]
        raw_total = sum(spec.weight for spec in raw)
        ids = [spec.source_id for spec in raw] + [
            next(spec.source_id for spec in NATURAL_SOURCES if spec.kind == "chat")]
        weights = [0.35 * spec.weight / raw_total for spec in raw] + [0.65]
    else:
        ids = [spec.source_id for spec in NATURAL_SOURCES]
        weights = [spec.weight for spec in NATURAL_SOURCES]
    return ids, np.asarray(weights, dtype=np.float64)


@torch.no_grad()
def validation_loss(model: FractalLM, corpus: NaturalCorpus, device, *,
                    seed: int, batches: int, batch_size: int, seq_len: int,
                    bf16: bool) -> float:
    was_training = model.training
    moe_logging = any(getattr(block.mlp, "_log", False) for block in model.blocks)
    if moe_logging:
        model.pop_moe_balance()
        model.set_moe_log(False)
    model.eval()
    rng = np.random.RandomState(seed)
    total = 0.0
    for _ in range(batches):
        x, y, w, _source = corpus.batch("val", batch_size, seq_len, device, rng)
        with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=bf16 and str(device).startswith("cuda")):
            loss = model(x, targets=y, loss_weight=w)[1]
        total += float(loss)
    model.train(was_training)
    if moe_logging:
        model.set_moe_log(True)
    return total / max(batches, 1)


@torch.no_grad()
def chat_termination_gate(model: FractalLM, tokenizer, device, max_new: int = 80) -> dict:
    prompts = (
        "Hello.",
        "What is water?",
        "Tell me one fact about dogs.",
        "What is two plus three?",
        "Say good morning.",
        "Name a color.",
        "What do plants need?",
        "Explain rain simply.",
        "Write one short sentence.",
        "What is the opposite of hot?",
        "Can birds fly?",
        "Name a fruit.",
        "What is a book?",
        "How many days are in a week?",
        "Say thank you.",
        "What is snow?",
        "Give a simple safety tip.",
        "What does a teacher do?",
        "Name one ocean.",
        "Answer with one word: yes or no.",
    )
    end_id = tokenizer.token_to_id(cf.END)
    leakage_ids = {
        tokenizer.token_to_id(marker)
        for marker in (cf.USER, cf.SYSTEM, cf.TOOL_RESULT)
        if tokenizer.token_to_id(marker) is not None
    }
    terminated = 0
    leaked = 0
    was_training = model.training
    model.eval()
    for prompt in prompts:
        prime = tokenizer.encode(f"{cf.USER}\n{prompt}\n{cf.ASSISTANT}\n").ids
        states = model.init_states(1, device)
        logits, states = model.forward_stream(torch.tensor([prime], device=device), states)
        for _ in range(max_new):
            token = int(logits[:, -1].argmax(dim=-1))
            if token == end_id:
                terminated += 1
                break
            if token in leakage_ids:
                leaked += 1
                break
            logits, states = model.forward_stream(torch.tensor([[token]], device=device), states)
    model.train(was_training)
    return {
        "prompts": len(prompts),
        "terminated": terminated,
        "role_leakage": leaked,
        "passed": terminated >= 18 and leaked == 0,
    }


def _lr_at(tokens: int, max_tokens: int, base_lr: float, warmup_fraction: float) -> float:
    progress = min(max(tokens / max(max_tokens, 1), 0.0), 1.0)
    if progress < warmup_fraction:
        return base_lr * progress / max(warmup_fraction, 1e-9)
    cosine_progress = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-9)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))


def train(args) -> dict:
    _seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(args.bf16 and device.type == "cuda")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = tk.load(args.tokenizer)
    tk.assert_atomic_special_tokens(tokenizer)
    if tokenizer.get_vocab_size() != 24_000:
        raise ValueError("Natural Cortex requires the matching 24k tokenizer")
    corpus = NaturalCorpus(args.data_dir)
    expected_tokenizer = corpus.manifest["tokenizer"]["sha256"]
    if _sha256(Path(args.tokenizer)) != expected_tokenizer:
        raise ValueError("tokenizer checksum does not match the data manifest")

    cfg = natural_config(args.variant)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run-state.pt"
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"resume state does not exist: {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=True)
        model = persist.load_model(str(output / state["checkpoint"]), device)
    elif args.init_checkpoint:
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to initialize in non-empty directory: {output}")
        model = persist.load_model(args.init_checkpoint, device)
        expected = natural_config(args.variant)
        fields = (
            "vocab_size", "n_embd", "n_head", "depth", "n_scales", "tau0", "rho",
            "chunk_size", "mlp_ratio", "untie", "n_experts", "moe_mode",
            "event_budget", "event_algebra", "growing_cortex", "skill_rank",
            "skill_compiler", "skill_address_dim", "skill_auto_route",
        )
        mismatches = {
            field: (getattr(model.cfg, field), getattr(expected, field))
            for field in fields
            if getattr(model.cfg, field) != getattr(expected, field)
        }
        if mismatches:
            raise ValueError(f"initial checkpoint does not match the {args.variant} preset: "
                             f"{mismatches}")
        state = {
            "tokens": int(args.initial_tokens),
            "step": 0,
            "elapsed_seconds": 0.0,
            "checkpoint": "",
        }
        if args.init_state:
            initial_state = torch.load(args.init_state, map_location=device, weights_only=True)
            state.update({
                "tokens": int(initial_state.get("tokens", args.initial_tokens)),
                "step": int(initial_state.get("step", 0)),
                "initial_optimizer": initial_state.get("optimizer"),
                "initial_plasticity": initial_state.get("plasticity"),
            })
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to start a new run in non-empty directory: {output}")
        model = FractalLM(cfg).to(device)
        state = {"tokens": 0, "step": 0, "elapsed_seconds": 0.0, "checkpoint": ""}
    model.grad_ckpt = bool(args.grad_ckpt)

    plasticity = PlasticityController(model)
    optimizer = torch.optim.AdamW(
        plasticity.param_groups(model, args.lr),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        plasticity.load_state_dict(state.get("plasticity") or {})
    elif state.get("initial_optimizer") is not None:
        optimizer.load_state_dict(state["initial_optimizer"])
        plasticity.load_state_dict(state.get("initial_plasticity") or {})

    tokens = int(state.get("tokens", 0))
    step = int(state.get("step", 0))
    elapsed_before = float(state.get("elapsed_seconds", 0.0))
    best_val = float(state.get("best_val", "inf"))
    best_tokens = int(state.get("best_tokens", 0))
    next_eval = ((tokens // args.eval_every_tokens) + 1) * args.eval_every_tokens
    next_save = ((tokens // args.save_every_tokens) + 1) * args.save_every_tokens
    anneal_chat = bool(state.get("anneal_chat", False))
    chat_gate = state.get("chat_gate")
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "config": cfg.__dict__,
        "stored_parameters": model.parameter_counts()[0],
        "active_parameters": model.parameter_counts()[1],
        "data_manifest": str(Path(args.data_dir) / "manifest.json"),
        "tokenizer_sha256": expected_tokenizer,
        "evaluations": list(state.get("evaluations") or []),
        "checkpoints": list(state.get("checkpoints") or []),
    }
    source_ids, source_probabilities = _source_weights(anneal_chat)
    model.train()
    if args.variant == "moe":
        model.set_moe_log(True)

    while tokens < args.max_tokens:
        elapsed = elapsed_before + time.perf_counter() - started
        if elapsed >= args.max_gpu_hours * 3600:
            break
        lr = _lr_at(tokens, args.max_tokens, args.lr, args.warmup_fraction)
        plasticity.apply(model, optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        step_rng = np.random.RandomState(args.seed + step)
        step_tokens = 0
        loss_value = 0.0
        for _ in range(args.accum):
            source_id = int(step_rng.choice(source_ids, p=source_probabilities))
            x, y, w, _labels = corpus.batch(
                "train", args.batch, args.seq_len, device, step_rng, source_id=source_id)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = model(x, targets=y, loss_weight=w)[1] / args.accum
                if args.variant == "moe":
                    balance = model.pop_moe_balance()
                    if balance is not None:
                        loss = loss + args.moe_lambda * balance / args.accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at token {tokens}: {float(loss)}")
            loss.backward()
            loss_value += float(loss.detach())
            step_tokens += x.numel()
        plasticity.update(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        tokens += step_tokens
        step += 1

        if tokens >= next_eval or tokens >= args.max_tokens:
            val = validation_loss(
                model, corpus, device, seed=args.val_seed, batches=args.val_batches,
                batch_size=args.val_batch, seq_len=args.seq_len, bf16=bf16)
            evaluation = {
                "tokens": tokens,
                "step": step,
                "loss": loss_value,
                "validation_loss": val,
                "lr": lr,
                "grad_norm": grad_norm,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            }
            if tokens >= 200_000_000 and chat_gate is None:
                chat_gate = chat_termination_gate(model, tokenizer, device)
                anneal_chat = not chat_gate["passed"]
                source_ids, source_probabilities = _source_weights(anneal_chat)
                evaluation["chat_gate"] = chat_gate
                evaluation["anneal_chat_65pct"] = anneal_chat
            report["evaluations"].append(evaluation)
            if val < best_val - args.min_delta:
                best_val = val
                best_tokens = tokens
            elif tokens - best_tokens >= args.early_stop_tokens:
                report["stop_reason"] = "no_validation_improvement"
                break
            next_eval += args.eval_every_tokens

        if tokens >= next_save or tokens >= args.max_tokens:
            checkpoint_name = f"checkpoint-{tokens:012d}.pt"
            persist.save_model(str(output / checkpoint_name), model)
            report["checkpoints"].append({"tokens": tokens, "file": checkpoint_name})
            durable = {
                "checkpoint": checkpoint_name,
                "optimizer": optimizer.state_dict(),
                "plasticity": plasticity.state_dict(),
                "tokens": tokens,
                "step": step,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "best_val": best_val,
                "best_tokens": best_tokens,
                "anneal_chat": anneal_chat,
                "chat_gate": chat_gate,
                "evaluations": report["evaluations"],
                "checkpoints": report["checkpoints"],
            }
            persist.atomic_torch_save(durable, state_path)
            next_save += args.save_every_tokens

        if args.telemetry and step % args.telemetry_every == 0:
            _atomic_json(Path(args.telemetry), {
                "iter": step,
                "tokens": tokens,
                "loss": loss_value,
                "lr": lr,
                "gnorm": grad_norm,
                "depth": cfg.depth,
                "n_scales": cfg.n_scales,
                "gammas": model.block.unit.gammas,
                "taus": [cfg.tau0, None],
                "n_embd": cfg.n_embd,
                "n_head": cfg.n_head,
                "mlp_ratio": cfg.mlp_ratio,
                "untie": False,
                "n_experts": cfg.n_experts,
                "moe_mode": cfg.moe_mode,
                "event_budget": 1.0,
                "active_params": round(model.parameter_counts()[1] / 1e6, 3),
                "params": round(model.parameter_counts()[0] / 1e6, 3),
                "growing_cortex": model.skill_cortex.snapshot(),
                "learning_signal": "gradient",
            })

    if not report["checkpoints"] or report["checkpoints"][-1]["tokens"] != tokens:
        checkpoint_name = f"checkpoint-{tokens:012d}.pt"
        persist.save_model(str(output / checkpoint_name), model)
        report["checkpoints"].append({"tokens": tokens, "file": checkpoint_name})
        persist.atomic_torch_save({
            "checkpoint": checkpoint_name,
            "optimizer": optimizer.state_dict(),
            "plasticity": plasticity.state_dict(),
            "tokens": tokens,
            "step": step,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "best_val": best_val,
            "best_tokens": best_tokens,
            "anneal_chat": anneal_chat,
            "chat_gate": chat_gate,
            "evaluations": report["evaluations"],
            "checkpoints": report["checkpoints"],
        }, state_path)
    elapsed = elapsed_before + time.perf_counter() - started
    report.update({
        "tokens": tokens,
        "steps": step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "best_validation_loss": best_val,
        "best_tokens": best_tokens,
        "chat_gate": chat_gate,
        "anneal_chat_65pct": anneal_chat,
        "stop_reason": report.get(
            "stop_reason",
            "token_budget" if tokens >= args.max_tokens else "gpu_time_budget",
        ),
    })
    _atomic_json(output / "report.json", report)
    return report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense", "moe"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_SEED + 1)
    parser.add_argument("--max-tokens", type=int, default=240_000_000)
    parser.add_argument("--max-gpu-hours", type=float, default=18.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--moe-lambda", type=float, default=0.01)
    parser.add_argument("--eval-every-tokens", type=int, default=10_000_000)
    parser.add_argument("--save-every-tokens", type=int, default=25_000_000)
    parser.add_argument("--early-stop-tokens", type=int, default=30_000_000)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="",
                        help="selected dense/MoE A/B checkpoint used to start the main run")
    parser.add_argument("--init-state", default="",
                        help="matching A/B optimizer/plasticity state")
    parser.add_argument("--initial-tokens", type=int, default=0,
                        help="processed-token count when --init-state is unavailable")
    parser.add_argument("--telemetry", default="")
    parser.add_argument("--telemetry-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    report = train(get_args())
    print(json.dumps({
        key: report[key]
        for key in ("variant", "tokens", "elapsed_seconds", "tokens_per_second", "stop_reason")
    }, indent=2))


if __name__ == "__main__":
    main()
