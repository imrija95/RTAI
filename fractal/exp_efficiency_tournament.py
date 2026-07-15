"""Four-hour, single-GPU falsification tournament for efficient FractalLM training.

The default schedule is intentionally wall-clock bounded. Every arm resets initialization and its
data stream, writes truthful sampled telemetry, saves a checkpoint, and is evaluated on the same
fixed chat/tool/recall suite. `--smoke` runs the complete orchestration quickly on CPU.

Examples:
    python -m fractal.exp_efficiency_tournament --smoke
    python -m fractal.exp_efficiency_tournament --budget_minutes 240 --bf16 --tf32
    python -m fractal.exp_efficiency_tournament --arm genome --minutes 30 --bf16 --tf32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from fractal import persist
from fractal.efficiency import VerifiedToolGen, local_credit_loss
from fractal.model import Config, FractalLM, MoEMLP


SCHEDULE_MINUTES = {
    "baseline": 25,
    "genome": 30,
    "moe_soft": 5,
    "moe_top1": 25,
    "event": 35,
    "local_credit": 30,
    "compiler": 30,
}
TRAIN_MINUTES = sum(SCHEDULE_MINUTES.values())       # 180; 60 min remain for warmup/eval/reserve
ARMS = tuple(SCHEDULE_MINUTES)


class MaskedSource:
    def __init__(self, directory: str, seed: int):
        self.data = np.memmap(os.path.join(directory, "train.bin"), dtype=np.uint16, mode="r")
        self.mask = np.memmap(os.path.join(directory, "train.mask.bin"), dtype=np.uint8, mode="r")
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, seq_len, device, loss_scale=1.0):
        ix = self.rng.integers(0, len(self.data) - seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + seq_len + 1].astype(np.int64) for i in ix])
        w = np.stack([self.mask[i + 1:i + seq_len + 1].astype(np.float32) for i in ix])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                torch.from_numpy(w).to(device) * loss_scale)


class TournamentBatches:
    """A deterministic per-step mixed stream; resetting an arm resets every source."""

    def __init__(self, args, tok, dev, compiler=False):
        self.args, self.dev, self.compiler = args, dev, compiler
        self.rng = random.Random(args.seed + 71)
        self.chat = MaskedSource(args.chat_dir, args.seed + 101)
        self.tools = MaskedSource(args.tools_dir, args.seed + 211)
        self.verified = VerifiedToolGen(tok, args.seed + 307)
        from fractal.recall import RecallGen
        self.recall = RecallGen(tok, val_bin=os.path.join(args.chat_dir, "val.bin"),
                                seed=args.seed + 401)
        self.calls = 0

    def next(self):
        self.calls += 1
        r = self.rng.random()
        B, T = self.args.batch, self.args.seq_len
        if self.compiler:
            if r < 0.50:
                return (*self.verified.batch(B, T, self.dev), "verified_tool")
            if r < 0.85:
                return (*self.chat.batch(B, T, self.dev, loss_scale=0.5), "chat")
        else:
            if r < 0.60:
                return (*self.chat.batch(B, T, self.dev), "chat")
            if r < 0.85:
                return (*self.tools.batch(B, T, self.dev), "tools")
        # RecallGen uses the module RNG; make every call repeatable across arms.
        random.seed(self.args.seed + 100_003 + self.calls)
        return (*self.recall.batch(B, T, self.dev, w_ans=5.0, max_facts=3), "recall")


def arm_config(args, arm):
    return Config(
        vocab_size=args.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
        depth=4, n_scales=2, tau0=16.0, rho=4.0, chunk_size=args.chunk_size,
        untie=False, n_experts=4 if arm.startswith("moe_") else 1,
        moe_mode="top1" if arm == "moe_top1" else "soft",
        event_budget=0.25 if arm == "event" else 1.0,
    )


def _copy_common_initialization(base_state, model):
    """Load every shape-compatible tensor, then transplant the baseline MLP into MoE experts."""
    own = model.state_dict()
    compatible = {k: v for k, v in base_state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    base_prefix = "blocks.0.mlp."
    for block in model.blocks:
        if isinstance(block.mlp, MoEMLP):
            for expert in block.mlp.experts:
                state = expert.state_dict()
                for key in tuple(state):
                    source = base_state.get(base_prefix + key)
                    if source is not None and source.shape == state[key].shape:
                        state[key] = source
                expert.load_state_dict(state)


def _amp(enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def _atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _depth_for_step(step, rng):
    return rng.choices((2, 4, 8), weights=(0.25, 0.625, 0.125), k=1)[0]


def train_arm(args, arm, dev, tok, base_state, duration_seconds, max_steps=None):
    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    cfg = arm_config(args, arm)
    model = FractalLM(cfg).to(dev).train()
    _copy_common_initialization(base_state, model)
    if cfg.n_experts > 1:
        model.set_moe_log(True)

    from fractal.plasticity import PlasticityController
    plast = PlasticityController(model)
    opt = torch.optim.AdamW(plast.param_groups(model, args.lr), betas=(0.9, 0.95),
                            fused=(dev == "cuda"))
    batches = TournamentBatches(args, tok, dev, compiler=(arm == "compiler"))
    depth_rng = random.Random(args.seed + 509)
    telemetry_path = f"{args.out_prefix}_{arm}.tele.json"
    checkpoint_path = f"{args.out_prefix}_{arm}.pt"
    start = time.perf_counter()
    step, tokens, last_loss = 0, 0, math.nan
    depth_counts = {2: 0, 4: 0, 8: 0}
    update_mode = "full"
    viz_prev_g = None
    measured_steps = []

    while True:
        elapsed = time.perf_counter() - start
        if max_steps is not None:
            if step >= max_steps:
                break
        elif elapsed >= duration_seconds:
            break
        measure_this = step >= 10 and len(measured_steps) < 100
        if measure_this and dev == "cuda":
            torch.cuda.synchronize()
        step_started = time.perf_counter()
        progress = min(1.0, elapsed / max(duration_seconds, 1e-6))
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        plast.apply(model, opt, lr)
        opt.zero_grad(set_to_none=True)
        loss_sum, step_depth = 0.0, 4

        batch_kind = "unknown"
        for micro in range(args.accum):
            x, y, w, batch_kind = batches.next()
            with _amp(args.bf16 and dev == "cuda"):
                if arm == "local_credit" and step % 8 != 0:
                    selected = (step * args.accum + micro) % cfg.depth
                    loss = local_credit_loss(model, x, y, w, selected)
                    update_mode, step_depth = f"local-d{selected}", selected + 1
                else:
                    step_depth = _depth_for_step(step * args.accum + micro, depth_rng) \
                        if arm == "genome" else cfg.depth
                    depth_counts[step_depth] = depth_counts.get(step_depth, 0) + 1
                    _, loss, _, _ = model(x, targets=y, loss_weight=w, depth=step_depth)
                    update_mode = "global" if arm == "local_credit" else "full"
                if cfg.n_experts > 1:
                    balance = model.pop_moe_balance()
                    if balance is not None:
                        loss = loss + 0.01 * balance
            (loss / args.accum).backward()
            loss_sum += float(loss.detach()) / args.accum
            tokens += x.numel()

        plast.update(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if measure_this:
            if dev == "cuda":
                torch.cuda.synchronize()
            measured_steps.append(time.perf_counter() - step_started)
        last_loss = loss_sum
        step += 1

        if step == 1 or step % args.telemetry_every == 0:
            from fractal.train import _viz_telemetry
            part, viz_prev_g = _viz_telemetry(model, viz_prev_g)
            stored, active = model.parameter_counts()
            expert_usage, selected_expert = None, None
            if isinstance(model.block.mlp, MoEMLP) and model.block.mlp._last_usage is not None:
                usage = model.block.mlp._last_usage.float()
                expert_usage = usage.sum(dim=(0, 1)).cpu().tolist()
                if model.block.mlp.mode == "top1":
                    selected_expert = int(usage[0, -1].argmax().item())
            payload = {
                "iter": step, "loss": round(last_loss, 5), "lr": lr,
                "gnorm": round(float(grad_norm), 4), "batch": args.batch,
                "block": args.seq_len, "params": round(stored / 1e6, 3),
                "tokens_per_step": args.batch * args.seq_len * args.accum,
                "active_params": round(active / 1e6, 3), "n_embd": cfg.n_embd,
                "depth": cfg.depth, "effective_depth": step_depth, "n_scales": cfg.n_scales,
                "untie": False, "n_experts": cfg.n_experts, "moe_mode": cfg.moe_mode,
                "event_budget": cfg.event_budget, "event_share": round(model.event_share(), 4),
                "update_mode": update_mode, "ckpt": os.path.basename(checkpoint_path),
                "mode": "attach", "arm": arm, "n_head": cfg.n_head,
                "mlp_ratio": cfg.mlp_ratio,
                "batch_kind": batch_kind, "text": tok.decode(x[0, :48].tolist()),
                "expert_usage": expert_usage, "selected_expert": selected_expert,
                "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                 if dev == "cuda" else None),
                "gammas": [float(g) for g in model.block.unit.gammas],
                "taus": [(cfg.tau0 * cfg.rho ** i if g < 1.0 else None)
                         for i, g in enumerate(model.block.unit.gammas)],
            }
            payload.update(part)
            _atomic_json(telemetry_path, payload)

    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    persist.save_model(checkpoint_path, model)
    stored, active = model.parameter_counts()
    train_result = {
        "arm": arm, "seed": args.seed, "seconds": elapsed, "steps": step,
        "tokens": tokens, "tokens_per_second": tokens / max(elapsed, 1e-9),
        "last_train_loss": last_loss,
        "median_step_seconds": (float(np.median(measured_steps)) if measured_steps else None),
        "benchmark_samples": len(measured_steps),
        "median_training_tokens_per_second": (
            args.batch * args.seq_len * args.accum / float(np.median(measured_steps))
            if measured_steps else None),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0),
        "stored_params": stored, "active_params": active,
        "event_share": model.event_share(), "depth_counts": depth_counts,
        "checkpoint": checkpoint_path, "telemetry": telemetry_path,
        "config": asdict(cfg),
    }
    return model.eval(), train_result, batches.recall


@torch.no_grad()
def masked_val_loss(model, args, dev, batches=32):
    source = MaskedSource(args.chat_dir.replace("train", "train"), args.seed + 9_001)
    # Validation uses dedicated files; replace the source arrays while retaining its local RNG.
    source.data = np.memmap(os.path.join(args.chat_dir, "val.bin"), dtype=np.uint16, mode="r")
    source.mask = np.memmap(os.path.join(args.chat_dir, "val.mask.bin"), dtype=np.uint8, mode="r")
    vals = []
    for _ in range(batches):
        x, y, w = source.batch(args.eval_batch, args.seq_len, dev)
        with _amp(args.bf16 and dev == "cuda"):
            _, loss, _, _ = model(x, targets=y, loss_weight=w)
        vals.append(float(loss))
    return sum(vals) / len(vals)


@torch.no_grad()
def tool_eval(model, tok, dev, n, seed):
    from fractal.agent import execute_tool, run_turn
    gen = VerifiedToolGen(tok, seed)
    valid = route = full = executed = 0
    for _ in range(n):
        user, expected_name, expected_args, _ = gen.episode()
        states = model.init_states(1, dev)
        transcript, _ = run_turn(model, tok, states, user, dev, max_new=(8 if n <= 2 else 48),
                                 max_tool_calls=1, temperature=0.1, top_k=1)
        calls = [v for kind, v in transcript if kind == "tool_call"]
        results = [v for kind, v in transcript if kind == "tool_result"]
        if not calls:
            continue
        call = calls[0]
        valid += int(call.get("name") is not None)
        route += int(call.get("name") == expected_name)
        args = call.get("arguments") or {}
        exact = call.get("name") == expected_name and args == expected_args
        full += int(exact)
        if exact and results:
            executed += int(results[0] == execute_tool(expected_name, expected_args))
    return {"valid": valid / n, "route": route / n, "full": full / n, "executed": executed / n}


def _restart_child(args):
    dev = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    model = persist.load_model(args.model, dev).eval()
    states = persist.load_states(args.restart_child, dev)
    payload = torch.load(args.restart_child + ".query.pt", map_location=dev, weights_only=True)
    with torch.no_grad():
        logits, _ = model.forward_stream(payload["query"].to(dev), states)
    pred = int(logits[0, -1].argmax())
    _atomic_json(args.restart_child + ".result.json",
                 {"prediction": pred, "answer": int(payload["answer"]), "correct": pred == payload["answer"]})


def process_restart_eval(model, checkpoint, recall, args, dev):
    random.seed(args.seed + 12_001)
    seq, answer, facts, distance, query_n, answer_pos = recall._episode(args.seq_len, 1, True)
    store = seq[:facts + distance]
    query = seq[facts + distance:answer_pos]
    states = model.init_states(1, dev)
    with torch.no_grad():
        _, states = model.forward_stream(torch.tensor([store], device=dev), states)
    state_path = checkpoint + ".restart_state.pt"
    persist.save_states(state_path, states)
    persist.atomic_torch_save({"query": torch.tensor([query]), "answer": answer},
                              state_path + ".query.pt")
    cmd = [sys.executable, "-m", "fractal.exp_efficiency_tournament",
           "--restart-child", state_path, "--model", checkpoint, "--device", args.device]
    proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    if proc.returncode != 0:
        return {"correct": False, "error": (proc.stderr or proc.stdout)[-500:]}
    with open(state_path + ".result.json", encoding="utf-8") as f:
        result = json.load(f)
    for path in (state_path, state_path + ".query.pt", state_path + ".result.json"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return result


def evaluate(model, train_result, recall, args, tok, dev):
    random.seed(args.seed + 10_001)
    val_batches = 2 if args.smoke else 32
    recall_n = 2 if args.smoke else 64
    tool_n = 2 if args.smoke else 60
    result = dict(train_result)
    result["eval"] = {
        "masked_val_loss": masked_val_loss(model, args, dev, val_batches),
        "recall_1fact_d48": recall.accuracy(model, 48, dev, recall_n, True, 1),
        "recall_3fact_d48": recall.accuracy(model, 48, dev, recall_n, True, 3),
        "tools": tool_eval(model, tok, dev, tool_n, args.seed + 11_001),
        "process_restart_recall": process_restart_eval(
            model, train_result["checkpoint"], recall, args, dev),
    }
    if train_result["arm"] == "genome":
        depth_loss = {}
        depth_caps = {}
        original = model.cfg.depth
        for depth in (2, 4, 8, 16):
            # Fixed validation positions, reset for each depth.
            model.cfg.depth = depth
            depth_loss[str(depth)] = masked_val_loss(model, args, dev, 2 if args.smoke else 8)
            if depth in (4, 8):
                random.seed(args.seed + 20_000 + depth)
                depth_caps[str(depth)] = {
                    "recall_1fact_d48": recall.accuracy(
                        model, 48, dev, 2 if args.smoke else 32, True, 1),
                    "tools": tool_eval(model, tok, dev, 2 if args.smoke else 20,
                                       args.seed + 21_000),
                }
        model.cfg.depth = original
        result["eval"]["depth_val_loss"] = depth_loss
        result["eval"]["depth_capabilities"] = depth_caps
    return result


def _quality_floor(result, baseline, loss_limit):
    ev, base = result["eval"], baseline["eval"]
    return (ev["masked_val_loss"] <= base["masked_val_loss"] * loss_limit
            and ev["recall_1fact_d48"] + 0.10 >= base["recall_1fact_d48"]
            and ev["recall_3fact_d48"] + 0.10 >= base["recall_3fact_d48"]
            and ev["tools"]["full"] + 0.10 >= base["tools"]["full"])


def _speed(result):
    return result.get("median_training_tokens_per_second") or result["tokens_per_second"]


def decide(results):
    """Apply the predeclared falsification gates; never promote a result by narrative judgment."""
    arms = results.get("arms", {})
    if "baseline" not in arms:
        return {"complete": False, "reason": "baseline arm is required for decisions"}
    base = arms["baseline"]
    verdicts = {}
    if "genome" in arms:
        r = arms["genome"]
        caps = r["eval"].get("depth_capabilities", {})
        losses = r["eval"].get("depth_val_loss", {})
        improvements = 0
        if losses.get("8", math.inf) <= 0.95 * losses.get("4", 0.0):
            improvements += 1
        if caps.get("8", {}).get("recall_1fact_d48", 0.0) >= \
                1.05 * caps.get("4", {}).get("recall_1fact_d48", 0.0) + 1e-9:
            improvements += 1
        if caps.get("8", {}).get("tools", {}).get("full", 0.0) >= \
                1.05 * caps.get("4", {}).get("tools", {}).get("full", 0.0) + 1e-9:
            improvements += 1
        verdicts["genome"] = {"pass": improvements >= 2 and _quality_floor(r, base, 1.05),
                              "depth_metrics_improved": improvements}
    if "moe_top1" in arms:
        r, soft = arms["moe_top1"], arms.get("moe_soft")
        speedup = _speed(r) / max(_speed(soft), 1e-9) if soft else 0.0
        verdicts["moe_top1"] = {"pass": speedup >= 1.5 and _quality_floor(r, base, 1.07),
                                "speedup_vs_soft4": speedup}
    if "event" in arms:
        r = arms["event"]
        speedup = _speed(r) / max(_speed(base), 1e-9)
        verdicts["event"] = {"pass": r["event_share"] <= 0.30 and speedup >= 1.5
                             and _quality_floor(r, base, 1.07),
                             "speedup_vs_baseline": speedup}
    if "local_credit" in arms:
        r = arms["local_credit"]
        speedup = _speed(r) / max(_speed(base), 1e-9)
        verdicts["local_credit"] = {"pass": speedup >= 1.4 and _quality_floor(r, base, 1.10),
                                    "speedup_vs_baseline": speedup}
    if "compiler" in arms:
        r = arms["compiler"]
        got, before = r["eval"]["tools"]["full"], base["eval"]["tools"]["full"]
        tool_gain = got >= before + 0.10 or (before > 0 and got >= 2.0 * before)
        verdicts["compiler"] = {"pass": tool_gain and _quality_floor(r, base, 1.10),
                                "absolute_tool_gain": got - before}
    passing = [name for name, verdict in verdicts.items() if verdict["pass"]]
    ranked = sorted(passing, key=lambda name: (
        arms[name]["eval"]["tools"]["executed"]
        + arms[name]["eval"]["recall_1fact_d48"]
        + arms[name]["eval"]["recall_3fact_d48"]
    ) / max(arms[name]["seconds"] / 60.0, 1e-9), reverse=True)
    return {"complete": True, "verdicts": verdicts, "promote_to_12gb": ranked[:2]}


def gpu_preflight(args, tok, dev, base_state):
    """Test the largest activation arm and apply the one predeclared uniform OOM fallback."""
    if dev != "cuda" or args.smoke:
        return
    for batch, accum in ((8, 2), (4, 4)):
        args.batch, args.accum = batch, accum
        try:
            torch.cuda.empty_cache()
            model = FractalLM(arm_config(args, "moe_soft")).to(dev).train()
            _copy_common_initialization(base_state, model)
            source = TournamentBatches(args, tok, dev)
            x, y, w, _ = source.next()
            with _amp(args.bf16):
                _, loss, _, _ = model(x, targets=y, loss_weight=w)
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            del model, loss, x, y, w
            torch.cuda.empty_cache()
            print(f"[preflight] batch={batch} accum={accum}, peak={peak:.2f} GB")
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[preflight] batch={batch} OOM; applying the uniform fallback", flush=True)
    raise SystemExit("the tournament does not fit in 4 GB even at batch=4; no arm was started")


def _base_initialization(args):
    torch.manual_seed(args.seed)
    model = FractalLM(arm_config(args, "baseline")).cpu()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_minutes", type=float, default=240.0)
    ap.add_argument("--minutes", type=float, default=0.0, help="override duration for a single --arm")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--chat_dir", default="fractal_data_chat32")
    ap.add_argument("--tools_dir", default="fractal_data_tools32")
    ap.add_argument("--tokenizer", default="fractal_tokenizer_32k.json")
    ap.add_argument("--out_prefix", default="fractal_ckpt_eff")
    ap.add_argument("--results", default="fractal_efficiency_results.json")
    ap.add_argument("--resume", action="store_true",
                    help="retain completed arms already present in --results")
    ap.add_argument("--telemetry_every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restart-child", default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.restart_child:
        _restart_child(args)
        return
    cuda = torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        raise SystemExit("CUDA requested but unavailable")
    dev = "cuda" if (args.device == "cuda" or (args.device == "auto" and cuda)) else "cpu"
    if args.tf32 and dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.smoke:
        args.batch, args.accum, args.seq_len = 2, 1, 32
        args.n_embd, args.n_head, args.chunk_size = 64, 4, 16
    from fractal import tokenizer as tokenizer_mod
    tok = tokenizer_mod.load(args.tokenizer)
    args.vocab_size = tok.get_vocab_size()
    args.eval_batch = min(args.batch, 4)

    for directory in (args.chat_dir, args.tools_dir):
        for name in ("train.bin", "train.mask.bin", "val.bin", "val.mask.bin"):
            if not os.path.exists(os.path.join(directory, name)):
                raise SystemExit(f"missing tournament data: {directory}/{name}")

    previous = None
    if args.resume and os.path.exists(args.results):
        with open(args.results, encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("seed") != args.seed:
            raise SystemExit("cannot resume results produced with a different seed")
    completed = set((previous or {}).get("arms", {}))
    arms = ((args.arm,) if args.arm else tuple(arm for arm in ARMS if arm not in completed))
    scale = max(0.0, (args.budget_minutes - 60.0) / TRAIN_MINUTES) if not args.arm else 1.0
    base_state = _base_initialization(args)
    gpu_preflight(args, tok, dev, base_state)
    args.eval_batch = min(args.batch, 4)
    all_results = {"schema_version": 1, "seed": args.seed, "device": dev,
                   "budget_minutes": args.budget_minutes, "arms": {}}
    if previous is not None:
        all_results["arms"].update(previous.get("arms", {}))
    for arm in arms:
        minutes = args.minutes if args.minutes > 0 else SCHEDULE_MINUTES[arm] * scale
        print(f"\n=== {arm}: {minutes:.1f} training minutes on {dev} ===", flush=True)
        model, train_result, recall = train_arm(
            args, arm, dev, tok, base_state, minutes * 60.0,
            max_steps=(2 if args.smoke else None))
        result = evaluate(model, train_result, recall, args, tok, dev)
        all_results["arms"][arm] = result
        _atomic_json(args.results, all_results)
        print(json.dumps({"arm": arm, "tok_s": round(result["tokens_per_second"], 1),
                          "val": round(result["eval"]["masked_val_loss"], 4),
                          "tools": result["eval"]["tools"],
                          "recall1": result["eval"]["recall_1fact_d48"]}, indent=2), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    all_results["decisions"] = decide(all_results)
    _atomic_json(args.results, all_results)
    print(f"\nresults -> {args.results}")
    print(json.dumps(all_results["decisions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
