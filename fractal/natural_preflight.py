"""Bounded Natural Cortex architecture, checkpointing, and telemetry preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fractal.growing_cortex import teach_expert
from fractal.model import Config, FractalLM
from fractal.natural_train import _atomic_json, _seed_all, natural_config
from fractal.train import _viz_telemetry


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark(cfg: Config, device, *, grad_ckpt: bool, batch: int, seq_len: int,
               steps: int, bf16: bool) -> dict:
    _seed_all(20260716)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = FractalLM(cfg).to(device).train()
    model.grad_ckpt = grad_ckpt
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=device.type == "cuda")
    samples = []
    telemetry_cost = []
    previous = None
    for step in range(steps + 1):
        data = torch.randint(0, cfg.vocab_size, (batch, seq_len + 1), device=device)
        _sync(device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            loss = model(data[:, :-1], targets=data[:, 1:])[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - started
        _sync(device)
        telemetry_started = time.perf_counter()
        _payload, previous = _viz_telemetry(model, previous)
        _sync(device)
        sampled_cost = time.perf_counter() - telemetry_started
        if step:
            samples.append(elapsed)
            telemetry_cost.append(sampled_cost)
    mean_step = sum(samples) / len(samples)
    mean_telemetry = sum(telemetry_cost) / len(telemetry_cost)
    return {
        "gradient_checkpointing": grad_ckpt,
        "seconds_per_step": mean_step,
        "tokens_per_second": batch * seq_len / mean_step,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None),
        "sampled_telemetry_seconds": mean_telemetry,
        "telemetry_overhead_every_25": mean_telemetry / 25.0 / mean_step,
    }


def _skill_birth_check(device) -> dict:
    _seed_all(17)
    model = FractalLM(Config(
        vocab_size=128, n_embd=32, n_head=4, depth=2, n_scales=2, chunk_size=8,
        growing_cortex=True, skill_rank=4, skill_compiler="none", skill_address_dim=8,
        skill_auto_route=False,
    )).to(device)
    address = model.tok_emb(torch.randint(0, 128, (1, 5), device=device))
    expert_id = model.skill_cortex.birth(task_features=address, name="test", synopsis="test skill")
    expert = model.skill_cortex.expert(expert_id)
    initial_up_zero = float(expert.up.detach().norm()) == 0.0
    initial_down_nonzero = float(expert.down.detach().norm()) > 0.0
    ids = torch.randint(0, 128, (1, 8), device=device)
    targets = torch.randint(0, 128, (1, 8), device=device)
    with torch.no_grad(), model.skill_cortex.suspend():
        base = model(ids)[0]
    with torch.no_grad(), model.skill_cortex.force(expert_id):
        candidate = model(ids)[0]
    result = teach_expert(model, expert_id, [(ids, targets)], steps=16, min_steps=16, patience=8)
    return {
        "function_preserving_birth": torch.equal(base, candidate),
        "initial_up_is_zero": initial_up_zero,
        "initial_down_is_nonzero": initial_down_nonzero,
        "zero_output_has_nonzero_gradient": (
            expert.up.grad is not None and float(expert.up.grad.norm()) > 0.0),
        "nonzero_update": result.update_norm > 0.0,
        "steps": result.steps,
    }


def run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    production_cfg = natural_config("dense")
    count_model = FractalLM(production_cfg)
    stored, active = count_model.parameter_counts()
    del count_model
    benchmark_cfg = production_cfg
    scope = "production"
    if device.type != "cuda" or args.smoke:
        benchmark_cfg = Config(
            vocab_size=512, n_embd=64, n_head=4, depth=3, n_scales=2, chunk_size=16,
            growing_cortex=True, skill_rank=4, skill_compiler="none", skill_address_dim=8,
            skill_auto_route=False,
        )
        scope = "smoke"
    without = _benchmark(
        benchmark_cfg, device, grad_ckpt=False, batch=args.batch,
        seq_len=args.seq_len if scope == "production" else 32,
        steps=args.steps, bf16=args.bf16)
    with_checkpointing = _benchmark(
        benchmark_cfg, device, grad_ckpt=True, batch=args.batch,
        seq_len=args.seq_len if scope == "production" else 32,
        steps=args.steps, bf16=args.bf16)
    no_ckpt_promoted = (
        scope == "production"
        and without["tokens_per_second"] >= 1.15 * with_checkpointing["tokens_per_second"]
        and without["peak_vram_gb"] is not None
        and without["peak_vram_gb"] < 10.5
    )
    selected = (
        "pending-production-benchmark" if scope != "production"
        else ("disabled" if no_ckpt_promoted else "enabled")
    )
    selected_speed = (
        without["tokens_per_second"] if selected == "disabled"
        else with_checkpointing["tokens_per_second"])
    report = {
        "schema_version": 1,
        "device": str(device),
        "hardware": (
            {
                "name": torch.cuda.get_device_name(device),
                "total_vram_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
            }
            if device.type == "cuda" else {"name": "CPU", "total_vram_gb": None}
        ),
        "benchmark_scope": scope,
        "production_parameters": {"stored": stored, "active": active},
        "full_hypernetwork_compiler_enabled": False,
        "production_flags": {
            "selective_recurrence": False,
            "event_patches": False,
            "event_algebra": False,
            "global_online_w0": False,
            "scalar_feedback_consolidation": False,
            "timescale_neurogenesis": False,
        },
        "gradient_checkpointing": {
            "without": without,
            "with": with_checkpointing,
            "selection": selected,
            "rule": "disable only when it is at least 15% faster and remains below 10.5 GB",
        },
        "dashboard_overhead_gate": {
            "measured_fraction_every_25": min(
                without["telemetry_overhead_every_25"],
                with_checkpointing["telemetry_overhead_every_25"]),
            "below_2pct": min(
                without["telemetry_overhead_every_25"],
                with_checkpointing["telemetry_overhead_every_25"]) < 0.02,
        },
        "skill_birth": _skill_birth_check(device),
        "projected_240m_gpu_hours": (
            240_000_000 / selected_speed / 3600.0 if scope == "production" else None),
        "long_run_authorized": False,
    }
    _atomic_json(Path(args.output), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/results/natural-cortex-preflight.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
