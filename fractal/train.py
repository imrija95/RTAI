"""Fractal model training with truncated-BPTT (train as you deploy).

A long span is split into block_size segments and the (detached) state is CARRIED
between them → the fact→query distance exceeds a single window → the long (γ→1) levels
get signal and must engage. Backward per segment → low peak memory.

To actually engage the fractal, use:
  --recall_ratio 0.5   fraction of recall episodes (fact → distant filler → query)
  --gate_lambda 0.03   gate load-balancing (against collapse onto a single level)
Without them the gate collapses onto the short-term level and the ladder is just decoration.

Run:  uv run python -m fractal.train --iters 3000 --recall_ratio 0.5 --gate_lambda 0.03
Smoke (no data): uv run python -m fractal.train --smoke
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import random

import numpy as np
import torch

from fractal.model import Config, FractalLM
from fractal import persist


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--val_seed", type=int, default=20260717,
                    help="fixed validation sampler seed reused at every evaluation")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--recall_block", type=int, default=0,
                    help="block for recall/agent episodes (0=block_size); episodes are ~<=32 tok, "
                         "so a small value avoids padding most compute to PADs (bit-identical loss)")     # segment length
    ap.add_argument("--segments", type=int, default=2)          # how many segments to carry (TBTT)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_scales", type=int, default=3)
    ap.add_argument("--tau0", type=float, default=16.0)
    ap.add_argument("--rho", type=float, default=4.0)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--plasticity", type=str, default="gate_driven", choices=["none", "gate_driven"],
                    help="usage-driven per-scale plasticity (DEFAULT; see fractal/plasticity.py) vs one "
                         "global cosine LR (none). Validated prove-in-small: better recall, neutral LM.")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--tokenizer", type=str, default="fractal_tokenizer.json",
                    help="tokenizer for the recall/agent curriculum AND dashboard text — MUST match the data")
    ap.add_argument("--out", type=str, default="fractal_ckpt.pt")
    ap.add_argument("--recall_ratio", type=float, default=0.0,
                    help="fraction of recall episodes (fact→filler→query); 0 = stories only")
    ap.add_argument("--gate_lambda", type=float, default=0.0,
                    help="gate regularization strength; 0 = off")
    ap.add_argument("--gate_prior_perm", type=float, default=-1.0,
                    help="routing prior: mass fraction on the PERMANENT scale (e.g. 0.5); <=0 = neg-entropy (uniform)")
    ap.add_argument("--w_ans", type=float, default=5.0,
                    help="weight of the answer token in recall episodes (weighted CE)")
    ap.add_argument("--task", type=str, default="recall", choices=["recall", "agent", "chat"],
                    help="recall (RecallGen) | agent (ToolGen scaffold) | chat (Phase 2: masked "
                         "unified-format trajectories from {train,val}.mask.bin, TBTT carry-state)")
    ap.add_argument("--n_names", type=int, default=0,
                    help="name/key pool size (0 = 6 single-token; >0 = large pool, no memorization)")
    ap.add_argument("--max_facts", type=int, default=4,
                    help="max facts per recall episode (capacity pressure)")
    ap.add_argument("--p_mem", type=float, default=0.35,
                    help="[agent] fraction of memory episodes in the batch (1.0 = pure memory, 0 = routing only)")
    ap.add_argument("--high_pass", action="store_true",
                    help="high-pass keys (novelty; separable keys for multi-fact)")
    ap.add_argument("--selective", action="store_true",
                    help="learned dynamics: data-dependent forget gate instead of fixed γ")
    ap.add_argument("--untie", action="store_true", help="separate block per depth (more params, ~100M)")
    ap.add_argument("--n_experts", type=int, default=1, help="functional MoE experts in the MLP (1=full MLP; >1=router+K MLPs)")
    ap.add_argument("--moe_mode", choices=["soft", "top1"], default="soft",
                    help="soft computes every expert; top1 dispatches each token to one expert")
    ap.add_argument("--moe_lambda", type=float, default=0.0, help="MoE router load-balance strength (usage neg-entropy)")
    ap.add_argument("--event_budget", type=float, default=1.0,
                    help="fraction of causally novel positions sent through the memory unit")
    ap.add_argument("--event_algebra", action="store_true",
                    help="carry O(1) eligibility traces for delayed predictive/user feedback")
    ap.add_argument("--eligibility_decay", type=float, default=0.95,
                    help="per-token retention of Event Algebra eligibility traces")
    ap.add_argument("--feedback_queue", default="",
                    help="optional JSONL queue of live user ratings to consolidate exactly once")
    ap.add_argument("--feedback_state", default="",
                    help="durable W0 + consumed-event state (default: OUT.feedback-state.pt)")
    ap.add_argument("--feedback_max_per_step", type=int, default=4,
                    help="maximum new live ratings consolidated at one safe batch boundary")
    ap.add_argument("--growing_cortex", action="store_true",
                    help="enable append-only, content-addressed low-rank skill hemispheres")
    ap.add_argument("--skill_rank", type=int, default=4,
                    help="rank of one active Growing Cortex skill residual")
    ap.add_argument("--skill_router_threshold", type=float, default=0.25,
                    help="minimum task-key similarity required to activate a stored skill")
    ap.add_argument("--skill_no_auto_route", action="store_true",
                    help="disable automatic skill routing; explicit sticky routing still works")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves VRAM)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the forward (~2x; model stays raw for save/telemetry)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (effective batch = batch×accum)")
    ap.add_argument("--save_every", type=int, default=0, help="periodic checkpoint every N iterations (0=only at the end)")
    ap.add_argument("--resume", action="store_true", help="resume from --out (+ .resume: optimizer, iter)")
    ap.add_argument("--data_dir", type=str, default="fractal_data",
                    help="corpus (.bin) — a different directory for fine-tune (won't overwrite pretraining data)")
    ap.add_argument("--bf16", action="store_true",
                    help="mixed precision (autocast bf16); weights/optimizer stay fp32")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (Ampere+)")
    ap.add_argument("--smoke", action="store_true", help="quick run on synthetic data (no data)")
    # NEUROGENESIS (grow shape at runtime) — see fractal/neurogenesis.py
    ap.add_argument("--neurogenesis", action="store_true",
                    help="allow scales to grow at runtime (new time scale on saturation)")
    ap.add_argument("--max_scales", type=int, default=6, help="cap on the number of scales (neurogenesis)")
    ap.add_argument("--grow_conc", type=float, default=0.6, help="gate concentration threshold for growth")
    ap.add_argument("--grow_plateau", type=float, default=0.02, help="‖W‖ saturation threshold (rel. plateau)")
    ap.add_argument("--grow_cooldown", type=int, default=150, help="min iterations between births")
    ap.add_argument("--grow_warmup", type=int, default=300, help="min iterations before the first birth")
    ap.add_argument("--birth_beta", type=float, default=2.0, help="newborn plasticity (β_gain)")
    ap.add_argument("--mature_steps", type=int, default=400, help="maturation time constant β_gain→1")
    ap.add_argument("--grow_no_demote", action="store_true",
                    help="gentle growth: do NOT demote the permanent scale (more γ=1 slots, preserves memory)")
    ap.add_argument("--grow_until", type=int, default=10 ** 9,
                    help="no births after this step (gentle v3: grow only early → time to recover)")
    ap.add_argument("--viz_telemetry", type=str, default="",
                    help="path to a JSON file with live training telemetry for the dashboard (VIZ_ATTACH)")
    ap.add_argument("--viz_every", type=int, default=25, help="write telemetry for the dashboard every N steps")
    return ap.parse_args()


def _amp(on):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on)


def _write_json_atomic(path, obj):
    """Write telemetry without a race (dashboard reads concurrently): tmp + os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


@torch.no_grad()
def _viz_telemetry(model, prev_g):
    """CHEAP training telemetry for the dashboard (VIBE #2): only gate gradient/weight norms
    per (depth, scale) — no extra forward/backward, grads already exist after .backward().
    Returns (partial payload dict, new prev_g). prev_g = gate gradient vectors for direction."""
    D, L, H = model.cfg.depth, model.cfg.n_scales, model.cfg.n_head
    grad = [[0.0] * L for _ in range(D)]
    wn = [[0.0] * L for _ in range(D)]
    gvec = [[None] * L for _ in range(D)]
    for d in range(D):
        g = model.block_at(d).unit.gate
        if g.weight.grad is None:
            continue
        gg = g.weight.grad.view(H, L, -1)
        gw = g.weight.data.view(H, L, -1)
        for l in range(L):
            grad[d][l] = float(gg[:, l, :].norm())
            wn[d][l] = float(gw[:, l, :].norm())
            gvec[d][l] = gg[:, l, :].reshape(-1)

    def _cos(a, b):
        if a is None or b is None:
            return 0.0
        na, nb = a.norm(), b.norm()
        return 0.0 if na < 1e-12 or nb < 1e-12 else round(float((a * b).sum() / (na * nb)), 3)

    ok = prev_g is not None and len(prev_g) == D and len(prev_g[0]) == L
    res = ([[_cos(gvec[d][l], prev_g[d][l]) for l in range(L)] for d in range(D)]
           if ok else [[0.0] * L for _ in range(D)])
    gammas = [round(float(x), 4) for x in model.block.unit.gammas]
    taus = [(model.cfg.tau0 * model.cfg.rho ** l if x < 1.0 else None) for l, x in enumerate(gammas)]

    # per-depth component signal for the machine cutaway (still cheap: norms of grads that
    # already exist after .backward(); no extra forward/backward)
    def _n(p):
        return float(p.grad.norm()) if (p is not None and p.grad is not None) else 0.0
    def _module_n(module):
        if module is None:
            return 0.0
        return sum(_n(parameter) ** 2 for parameter in module.parameters()) ** 0.5
    parts = {"qk": [], "beta": [], "proj": [], "mlp": [], "skill": []}
    for d in range(D):
        u = model.block_at(d).unit
        mlp = model.block_at(d).mlp
        parts["qk"].append(round(_n(u.to_qk.weight), 5))
        parts["beta"].append(round(_n(u.to_beta.weight), 5))
        parts["proj"].append(round(_n(u.proj.weight), 5))
        if hasattr(mlp, "router"):                  # MoE: router = where the mixture is learning
            parts["mlp"].append(round(_n(mlp.router.weight), 5))
        else:
            parts["mlp"].append(round((_n(mlp.fc.weight) ** 2 + _n(mlp.proj.weight) ** 2) ** 0.5, 5))
        skill = model.skill_cortex
        parts["skill"].append(round(
            0.0 if skill is None else (
                _module_n(skill.compiler) ** 2 + _module_n(skill.query_proj) ** 2
            ) ** 0.5, 5))
    return {"grad": grad, "wn": wn, "res": res, "depth": D, "n_scales": L,
            "gammas": gammas, "taus": taus, "parts": parts,
            "n_head": model.cfg.n_head, "mlp_ratio": getattr(model.cfg, "mlp_ratio", 2)}, gvec


def _tbtt_step(model, x, y, block_size, segments, device, gate_lambda=0.0, bf16=False, w=None):
    """TBTT step: iterate segments, carry a detached state between them, backward per segment
    (low peak memory). Optionally add gate load-balancing. `w` (per-token loss weight, same shape
    as y) enables Phase-2 masked CE — a segment whose weights are all 0 contributes no gradient."""
    B = x.shape[0]
    states = model.init_states(B, device)
    total, n = 0.0, 0
    for s in range(segments):
        xs, ys = x[:, s * block_size:(s + 1) * block_size], y[:, s * block_size:(s + 1) * block_size]
        ws = w[:, s * block_size:(s + 1) * block_size] if w is not None else None
        if xs.shape[1] == 0:
            break
        with _amp(bf16):
            _, loss, states, _ = model(xs, targets=ys, states=states, loss_weight=ws)
        if gate_lambda > 0.0:
            aux = model.block.unit.pop_gate_balance_loss()
            if aux is not None:
                loss = loss + gate_lambda * aux
        (loss / segments).backward()
        states = [st.detach() for st in states]
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _val_ppl(model, device, nb=8, data_dir="fractal_data", seed=20260717):
    from fractal.data import get_batch as gb
    model.eval()
    tot = 0.0
    rng = np.random.RandomState(seed)
    for _ in range(nb):
        x, y = gb("val", 8, 128, device, data_dir=data_dir, rng=rng)
        _, loss, _, _ = model(x, targets=y, states=None)
        tot += loss.item()
    model.train()
    return math.exp(tot / nb)


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        args.bf16 = args.tf32 = False
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.smoke:
        args.iters, args.batch, args.block_size, args.segments = 30, 4, 32, 3
        args.n_embd, args.depth, args.n_scales, args.vocab_size = 64, 3, 3, 64
        args.recall_ratio, args.gate_lambda = 0.0, 0.0
        if args.out == "fractal_ckpt.pt":          # smoke must not overwrite the real checkpoint
            args.out = "fractal_smoke_ckpt.pt"

    cfg = Config(vocab_size=args.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
                 depth=args.depth, n_scales=args.n_scales, tau0=args.tau0, rho=args.rho,
                 chunk_size=args.chunk_size, high_pass_keys=args.high_pass,
                 selective=args.selective, untie=args.untie, n_experts=args.n_experts,
                 moe_mode=args.moe_mode, event_budget=args.event_budget,
                 event_algebra=(args.event_algebra or bool(args.feedback_queue)),
                 eligibility_decay=args.eligibility_decay,
                 growing_cortex=args.growing_cortex,
                 skill_rank=args.skill_rank,
                 skill_router_threshold=args.skill_router_threshold,
                 skill_auto_route=not args.skill_no_auto_route)
    if args.resume and os.path.exists(args.out):
        model = persist.load_model(args.out, dev)      # restore the model (and its cfg)
        cfg = model.cfg
    else:
        model = FractalLM(cfg).to(dev)
    feedback_seen = set()
    feedback_mod = feedback_tok = None
    feedback_state_path = args.feedback_state or (args.out + ".feedback-state.pt")
    if args.feedback_queue:
        from fractal import chat_format as feedback_format
        from fractal import feedback as feedback_mod
        from fractal import tokenizer as feedback_tokenizer
        feedback_mod.enable(model)
        feedback_tok = feedback_tokenizer.load(args.tokenizer)
        if feedback_tok.get_vocab_size() != model.cfg.vocab_size:
            raise ValueError("feedback tokenizer/model vocabulary mismatch")
        feedback_seen = feedback_mod.load_consolidation_state(feedback_state_path, model)
        print(f"[feedback] live queue: {args.feedback_queue} | consumed {len(feedback_seen)} events",
              flush=True)
    model.grad_ckpt = args.grad_ckpt
    if args.gate_lambda > 0.0:
        model.block.unit._log_gate = True
        if args.gate_prior_perm > 0:                # decay-aware routing prior instead of neg-entropy
            model.block.unit.gate_prior_perm = args.gate_prior_perm
    if args.moe_lambda > 0.0:                        # load-balance the MoE router (functional experts)
        model.set_moe_log(True)

    neuro = None
    if args.neurogenesis:
        from fractal import neurogenesis as ng
        for u in ng.units(model):
            u._log_share = True                     # gate-sharing telemetry (growth trigger)
        neuro = ng.NeurogenesisController(
            max_scales=args.max_scales, conc_thresh=args.grow_conc, plateau_eps=args.grow_plateau,
            cooldown=args.grow_cooldown, warmup=args.grow_warmup,
            birth_beta_gain=args.birth_beta, mature_steps=args.mature_steps,
            demote=not args.grow_no_demote, max_grow_step=args.grow_until)
        print(f"[neurogenesis] ON: start {model.cfg.n_scales} scales → max {args.max_scales}, "
              f"conc>{args.grow_conc}, plateau<{args.grow_plateau}, cooldown {args.grow_cooldown}", flush=True)

    plast = None
    if args.plasticity != "none":
        if neuro is not None:
            print("[plasticity] disabled: neurogenesis owns beta_gain and grows the number of scales "
                  "(incompatible with fixed per-scale param groups)", flush=True)
        else:
            from fractal.plasticity import PlasticityController
            plast = PlasticityController(model)
            print(f"[plasticity] ON (default): usage-driven per-scale plasticity ({args.plasticity})", flush=True)

    if plast is not None:
        opt = torch.optim.AdamW(plast.param_groups(model, args.lr), betas=(0.9, 0.95), fused=(dev == "cuda"))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), fused=(dev == "cuda"))
    start_iter = 0
    if args.resume and os.path.exists(args.out + ".resume"):
        r = torch.load(args.out + ".resume", map_location=dev, weights_only=True)
        opt.load_state_dict(r["opt"]); start_iter = r["iter"] + 1
        if plast is not None and r.get("plast") is not None:
            plast.load_state_dict(r["plast"])
        print(f"resume from iteration {start_iter}", flush=True)
    seq_len = args.block_size * args.segments

    rg = None
    if args.recall_ratio > 0.0 and not args.smoke and args.task in ("recall", "agent", "chat"):
        from fractal import tokenizer as tk
        _tok = tk.load(args.tokenizer)
        if args.task == "agent":
            from fractal.data_agent import ToolGen
            rg = ToolGen(_tok)                          # tool curriculum (tooling) + caveman
        else:                                           # recall | chat → RecallGen memory episodes
            from fractal.recall import RecallGen
            rg = RecallGen(_tok, n_names=args.n_names)
            # For chat, RecallGen is a SMALL same-purpose augmentation that keeps the fast-weight
            # recall skill from decaying under masked chat/tool SFT (VIBE #4/#8; ROADMAP Phase 2).

    def story_batch():
        if args.smoke:
            data = torch.randint(0, cfg.vocab_size, (args.batch, seq_len + 1), device=dev)
            return data[:, :-1], data[:, 1:]
        from fractal.data import get_batch as gb
        return gb("train", args.batch, seq_len, dev, data_dir=args.data_dir)

    def masked_batch():                                  # Phase 2: (x, y, w) with the loss mask
        from fractal.data import get_masked_batch as gmb
        return gmb("train", args.batch, seq_len, dev, data_dir=args.data_dir)

    def save_ckpt(it):
        persist.save_model(args.out, model)          # also persists per-scale beta_gain (plasticity)
        blob = {"opt": opt.state_dict(), "iter": it}
        if neuro is not None:
            blob["births"] = neuro.births            # record of births (when/why a scale grew)
        if plast is not None:
            blob["plast"] = plast.state_dict()       # usage EMA (resume-safe)
        persist.atomic_torch_save(blob, args.out + ".resume")

    def consume_feedback():
        if feedback_mod is None:
            return 0
        pending = [event for event in feedback_mod.read_events(args.feedback_queue)
                   if event["event_id"] not in feedback_seen]
        applied = 0
        for event in pending[:max(args.feedback_max_per_step, 0)]:
            role = event.get("role")
            marker = feedback_format.USER if role == "user" else feedback_format.ASSISTANT
            content = str(event.get("content", "")).strip()
            credit = float(event.get("credit_delta", 0.0))
            if content and credit:
                ids = feedback_tok.encode(f"{marker}\n{content}").ids
                evidence = feedback_mod.message_eligibility(model, ids, dev)
                feedback_mod.consolidate_w0(model, evidence, credit)
            feedback_seen.add(event["event_id"])
            applied += 1
        if applied:
            feedback_mod.save_consolidation_state(feedback_state_path, model, feedback_seen)
        return applied

    # --- training telemetry for the dashboard (VIBE #1/#2: truthful but cheap – only every N steps) ---
    viz_params = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    viz_tok, viz_prev_g, viz_gn_t = None, None, None
    if args.viz_telemetry:
        from fractal import tokenizer as tk
        try:
            viz_tok = tk.load(args.tokenizer)
        except Exception:
            viz_tok = None
        print(f"[viz] training telemetry → {args.viz_telemetry} (every {args.viz_every} steps)", flush=True)

    model.train()
    reported = False
    fwd = torch.compile(model) if args.compile else model   # compiled forward only; model stays raw
    if args.compile:
        print("[compile] torch.compile ON (forward)", flush=True)
    for it in range(start_iter, args.iters):
        cur_ids = None                          # input sample for the dashboard (where the data flows)
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * it / args.iters))   # cosine decay
        if plast is not None:
            plast.apply(model, opt, lr)          # per-scale W0 LR + beta_gain from the usage EMA
        else:
            for g in opt.param_groups:
                g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_states = None                     # states for ‖W‖ telemetry (neurogenesis); recall branch only

        if args.task == "chat":
            if rg is not None and random.random() < args.recall_ratio:
                # small MEMORY augmentation: a RecallGen episode (full-BPTT, weighted loss) so the
                # fast-weight recall skill does not decay under masked chat/tool SFT (VIBE #4/#8).
                xr, yr, wr = rg.batch(args.batch, args.recall_block or args.block_size, dev,
                                      w_ans=args.w_ans, max_facts=args.max_facts)
                cur_ids = xr[0]
                with _amp(args.bf16):
                    _, loss_t, step_states, _ = fwd(xr, targets=yr, states=None, loss_weight=wr)
                loss_t.backward()
                loss = loss_t.item()
            else:
                # PHASE 2: masked unified-format trajectory, TBTT carrying a detached state (train-as-
                # deploy). CE only on assistant + <|tool_call|> + <|end|> tokens (w from the mask).
                x, y, w = masked_batch()
                cur_ids = x[0]
                loss = _tbtt_step(fwd, x, y, args.block_size, max(args.segments, 1), dev,
                                  args.gate_lambda, args.bf16, w=w)
        elif rg is not None and random.random() < args.recall_ratio:
            # RECALL: episode ≤ block_size, one full-BPTT pass + weighted loss (see recall.py)
            _extra = {"p_mem": args.p_mem} if args.task == "agent" else {}
            xr, yr, wr = rg.batch(args.batch, args.recall_block or args.block_size, dev, w_ans=args.w_ans, max_facts=args.max_facts, **_extra)
            cur_ids = xr[0]
            with _amp(args.bf16):
                _, loss_t, step_states, _ = fwd(xr, targets=yr, states=None, loss_weight=wr)
            if args.gate_lambda > 0.0:
                aux = model.block.unit.pop_gate_balance_loss()
                if aux is not None:
                    loss_t = loss_t + args.gate_lambda * aux
            if args.moe_lambda > 0.0:
                mb = model.pop_moe_balance()
                if mb is not None:
                    loss_t = loss_t + args.moe_lambda * mb
            loss_t.backward()
            loss = loss_t.item()
        elif args.segments > 1:
            # STORY + TBTT (carries state between segments) — for the recall context of the small model
            x, y = story_batch()
            cur_ids = x[0]
            loss = _tbtt_step(fwd, x, y, args.block_size, args.segments, dev,
                              args.gate_lambda, args.bf16)
        else:
            # PLAIN LM + gradient accumulation (large model; states=None → grad checkpointing)
            loss = 0.0
            for _ in range(args.accum):
                x, y = story_batch()
                cur_ids = x[0]
                with _amp(args.bf16):
                    _, l, _, _ = fwd(x, targets=y, states=None)
                if args.gate_lambda > 0.0:                 # POP here too: otherwise _gate_log holds stale
                    aux = model.block.unit.pop_gate_balance_loss()  # tensors from a freed graph →
                    if aux is not None:                    # next backward fails ("second time")
                        l = l + args.gate_lambda * aux
                (l / args.accum).backward()
                loss += l.item() / args.accum

        if plast is not None:
            plast.update(model)                  # refresh usage EMA from this step's gate reads
        viz_gn_t = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # tensor; float() only in the dump
        opt.step()
        feedback_applied = consume_feedback()
        if feedback_applied:
            print(f"  [feedback @ iter {it}] consolidated {feedback_applied} event(s) into W0",
                  flush=True)

        if args.viz_telemetry and it % args.viz_every == 0:      # cheap dump (VIBE #2) — grads still valid
            part, viz_prev_g = _viz_telemetry(model, viz_prev_g)
            part.update({"iter": it, "loss": round(float(loss), 4), "lr": lr,
                         "gnorm": round(float(viz_gn_t), 3),
                         "batch": args.batch, "block": args.block_size,   # → dashboard tok/s throughput
                         "params": viz_params, "n_embd": model.cfg.n_embd,
                         "untie": bool(model.cfg.untie), "n_experts": getattr(model.cfg, "n_experts", 1),
                         "moe_mode": getattr(model.cfg, "moe_mode", "soft"),
                         "event_budget": getattr(model.cfg, "event_budget", 1.0),
                         "event_share": round(model.event_share(), 4),
                         "growing_cortex": (
                             None if model.skill_cortex is None
                             else model.skill_cortex.snapshot()),
                         "active_params": round(model.parameter_counts()[1] / 1e6, 2),
                         "ckpt": os.path.basename(args.out),
                         "text": (viz_tok.decode(cur_ids[:32].tolist()) if (viz_tok is not None and cur_ids is not None) else "")})
            _write_json_atomic(args.viz_telemetry, part)

        if neuro is not None:
            neuro.mature(model)                                     # β_gain maturation of newborns
            share, wnorm = ng.telemetry(model, step_states)
            neuro.observe(share, wnorm)
            birth = neuro.maybe_grow(model, it)
            if birth is not None:
                # GENTLE: add ONLY the new parameters (gate/to_f/newborn cell) to the optimizer —
                # Adam moments of the OLD weights (embedding, projections, MLP, existing scales) stay.
                # (previously the whole optimizer was reset → destabilization, see A/B v2.)
                opt.add_param_group({"params": birth["new_params"], "lr": lr, "betas": (0.9, 0.95)})
                print(f"  [NEUROGENESIS @ iter {it}] +scale → {birth['new_n_scales']} "
                      f"(dominant L{birth['dominant']} conc {birth['conc']}, "
                      f"‖W‖ plateau slope {birth['wn_slope']}, γ_newborn {birth['birth_gamma']}, "
                      f"demote={neuro.demote})", flush=True)

        if dev == "cuda" and not reported:
            print(f"  peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
            reported = True
        if args.save_every and it > start_iter and it % args.save_every == 0:
            save_ckpt(it)
            print(f"  [checkpoint @ iter {it} → {args.out}]", flush=True)
        if it % max(args.iters // 20, 1) == 0 or it == args.iters - 1:
            extra = "" if args.smoke else (
                f"  val_ppl {_val_ppl(fwd, dev, data_dir=args.data_dir, seed=args.val_seed):.1f}")
            print(f"iter {it:4d}  loss {loss:.4f}  lr {lr:.2e}{extra}", flush=True)

    save_ckpt(args.iters - 1)
    print(f"saved → {args.out}")

    if neuro is not None:
        print(f"[neurogenesis] final number of scales: {model.cfg.n_scales} "
              f"(γ ladder {[round(g, 3) for g in model.block.unit.gammas]})")
        print(f"[neurogenesis] births: {len(neuro.births)} → {neuro.births}")

    # final report
    if not args.smoke:
        print(f"final val_ppl = "
              f"{_val_ppl(model, dev, 16, data_dir=args.data_dir, seed=args.val_seed):.1f}")
    if rg is not None and args.task == "agent":
        from fractal.grammar import guided_call
        for split, ho in [("SEEN", False), ("HELD-OUT", True)]:
            g = rg.accuracy(model, device=dev, n=128, held_out=ho, guided=guided_call)
            mem = rg.mem_accuracy(model, device=dev, n=128, held_out=ho)
            print(f"agent [{split}] guided: route {g['route']*100:.0f}% full {g['full']*100:.0f}% "
                  f"valid {g['valid']*100:.0f}%  ||  MEMORY recall: {mem*100:.0f}%")
    elif rg is not None:
        print("recall 1-fact HELD-OUT:",
              {D: round(rg.accuracy(model, D, dev, n=64, held_out=True, n_facts=1) * 100) for D in (16, 48, 96, 200)})
        # sweep over the number of facts (capacity pressure) — 3 kept for comparability, + harder end
        for nf in sorted({3, 6, args.max_facts}):
            if nf <= len(rg.names):
                print(f"recall {nf}-facts HELD-OUT (D=48):",
                      round(rg.accuracy(model, 48, dev, n=64, held_out=True, n_facts=nf) * 100), "%")
        model.eval()
        gouts = []
        h = model.block.unit.gate.register_forward_hook(lambda mo, i, o: gouts.append(o.detach()))
        xr, _, _ = rg.batch(6, 128, dev)
        with torch.no_grad():
            model(xr)
        h.remove()
        H, L = cfg.n_head, cfg.n_scales
        gate = torch.cat([o.reshape(-1, H * L) for o in gouts], 0).view(-1, H, L).softmax(-1).mean(dim=(0, 1))
        print("gate:", [f"L{l} {gate[l].item()*100:.0f}%" for l in range(L)])


if __name__ == "__main__":
    main()
