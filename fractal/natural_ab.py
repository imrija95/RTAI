"""Deterministic 10M-token dense versus top-1 MoE promotion screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fractal import persist
from fractal import tokenizer as tk
from fractal.natural_data import NATURAL_SOURCES, NaturalCorpus
from fractal.natural_train import (
    DEFAULT_SEED,
    _atomic_json,
    _lr_at,
    _seed_all,
    copy_dense_stem,
    natural_config,
    validation_loss,
)
from fractal.model import FractalLM
from fractal.plasticity import PlasticityController
from fractal.recall import RecallGen


def _initial_models(seed):
    _seed_all(seed)
    dense_model = FractalLM(natural_config("dense")).cpu()
    _seed_all(seed)
    moe_model = FractalLM(natural_config("moe")).cpu()
    copy_dense_stem(dense_model, moe_model)
    return dense_model, moe_model


def _train_arm(model, variant, corpus, args, device, deadline):
    model = model.to(device).train()
    model.grad_ckpt = args.grad_ckpt
    plasticity = PlasticityController(model)
    optimizer = torch.optim.AdamW(
        plasticity.param_groups(model, args.lr), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, fused=device.type == "cuda")
    if variant == "moe":
        model.set_moe_log(True)
    rng = np.random.RandomState(args.seed)
    source_ids = [spec.source_id for spec in NATURAL_SOURCES]
    probabilities = np.asarray([spec.weight for spec in NATURAL_SOURCES], dtype=np.float64)
    tokens = 0
    steps = 0
    started = time.perf_counter()
    losses = []
    while tokens < args.tokens:
        if time.perf_counter() >= deadline:
            raise TimeoutError("dense/MoE A/B exceeded its shared GPU-time budget")
        lr = _lr_at(tokens, args.tokens, args.lr, 0.01)
        plasticity.apply(model, optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            source_id = int(rng.choice(source_ids, p=probabilities))
            x, y, w, _labels = corpus.batch(
                "train", args.batch, args.seq_len, device, rng, source_id=source_id)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.bf16 and device.type == "cuda"):
                loss = model(x, targets=y, loss_weight=w)[1] / args.accum
                if variant == "moe":
                    balance = model.pop_moe_balance()
                    if balance is not None:
                        loss = loss + args.moe_lambda * balance / args.accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} produced non-finite loss")
            loss.backward()
            step_loss += float(loss.detach())
            tokens += x.numel()
        plasticity.update(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(step_loss)
        steps += 1
    elapsed = time.perf_counter() - started
    val = validation_loss(
        model, corpus, device, seed=args.val_seed, batches=args.val_batches,
        batch_size=args.val_batch, seq_len=args.seq_len,
        bf16=args.bf16 and device.type == "cuda")
    return model.eval(), {
        "tokens": tokens,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "validation_loss": val,
        "last_train_loss": losses[-1],
    }, {
        "optimizer": optimizer.state_dict(),
        "plasticity": plasticity.state_dict(),
        "tokens": tokens,
        "step": steps,
    }


@torch.no_grad()
def _moe_routing(model, corpus, args, device):
    source_usage = {}
    overall = []
    for spec in NATURAL_SOURCES:
        rows = []
        rng = np.random.RandomState(args.val_seed + spec.source_id)
        for _ in range(args.affinity_batches):
            x, _y, _w, _labels = corpus.batch(
                "val", args.val_batch, args.seq_len, device, rng, source_id=spec.source_id)
            model(x)
            usage = model.block.mlp._last_usage.float().mean(dim=(0, 1)).cpu().numpy()
            rows.append(usage)
            overall.append(usage)
        source_usage[spec.key] = np.stack(rows).mean(axis=0).tolist()
    total_usage = np.stack(overall).mean(axis=0)
    max_tv = 0.0
    values = list(source_usage.values())
    for left_index, left in enumerate(values):
        for right in values[left_index + 1:]:
            max_tv = max(max_tv, 0.5 * float(np.abs(np.asarray(left) - np.asarray(right)).sum()))
    return {
        "overall_usage": total_usage.tolist(),
        "minimum_expert_usage": float(total_usage.min()),
        "source_usage": source_usage,
        "maximum_pairwise_source_tv": max_tv,
    }


def _recall(model, tokenizer, corpus, seed, device):
    first_val = corpus.manifest["splits"]["val"]["shards"][0]["tokens"]
    generator = RecallGen(tokenizer, str(corpus.root / first_val), seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return generator.accuracy(model, 48, device, n=64, held_out=True, n_facts=1)


def run(args):
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite A/B output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = tk.load(args.tokenizer)
    tk.assert_atomic_special_tokens(tokenizer)
    corpus = NaturalCorpus(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    deadline = time.perf_counter() + args.max_gpu_hours * 3600
    dense_initial, moe_initial = _initial_models(args.seed)
    dense, dense_metrics, dense_state = _train_arm(
        dense_initial, "dense", corpus, args, device, deadline)
    persist.save_model(str(output / "dense-10m.pt"), dense)
    persist.atomic_torch_save(dense_state, output / "dense-10m.resume")
    dense_recall = _recall(dense, tokenizer, corpus, args.seed + 99, device)
    del dense
    if device.type == "cuda":
        torch.cuda.empty_cache()

    moe, moe_metrics, moe_state = _train_arm(
        moe_initial, "moe", corpus, args, device, deadline)
    persist.save_model(str(output / "moe-10m.pt"), moe)
    persist.atomic_torch_save(moe_state, output / "moe-10m.resume")
    moe_recall = _recall(moe, tokenizer, corpus, args.seed + 99, device)
    routing = _moe_routing(moe, corpus, args, device)

    gates = {
        "throughput_at_least_85pct_dense": (
            moe_metrics["tokens_per_second"] >= 0.85 * dense_metrics["tokens_per_second"]),
        "validation_loss_within_2pct": (
            moe_metrics["validation_loss"] <= 1.02 * dense_metrics["validation_loss"]),
        "every_expert_at_least_10pct": routing["minimum_expert_usage"] >= 0.10,
        "source_affinity_tv_at_least_5pct": routing["maximum_pairwise_source_tv"] >= 0.05,
        "fast_weight_recall_no_2pp_regression": moe_recall >= dense_recall - 0.02,
    }
    selected = "moe" if all(gates.values()) else "dense"
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "tokens_per_arm": args.tokens,
        "identical_data_order": True,
        "identical_dense_stem": True,
        "dense": {
            **dense_metrics, "recall": dense_recall,
            "checkpoint": "dense-10m.pt", "resume_state": "dense-10m.resume",
        },
        "moe": {
            **moe_metrics,
            "recall": moe_recall,
            "routing": routing,
            "checkpoint": "moe-10m.pt",
            "resume_state": "moe-10m.resume",
        },
        "gates": gates,
        "selected": selected,
        "continuation_checkpoint": f"{selected}-10m.pt",
        "continuation_state": f"{selected}-10m.resume",
    }
    _atomic_json(output / "report.json", report)
    return report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--max-gpu-hours", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_SEED + 1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--moe-lambda", type=float, default=0.01)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--affinity-batches", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-ckpt", action="store_true")
    return parser.parse_args()


def main():
    report = run(get_args())
    print(json.dumps({"selected": report["selected"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
