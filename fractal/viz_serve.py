"""Live server for the architecture and telemetry dashboard (stdlib http.server).

  /              → fractal3d.html
  /config        → real model config (depth, n_scales, γ ladder, τ, params, mode…)
  /activity      → real quantities per (depth, scale) by mode.

Three modes (VIBE #1: the dashboard truthfully shows shape, data, and REAL learning):
  READ (default)  — model READS the val corpus via persistent state; ‖ΔW‖ write, read gate.
        FRACTAL_CKPT=fractal_ckpt_big.pt PORT=8008 CUDA_VISIBLE_DEVICES= \
            uv run python -m fractal.viz_serve
  LEARN           — server itself trains a fresh small model FROM SCRATCH (learning demo, not a real run):
        VIZ_LEARN=1 CUDA_VISIBLE_DEVICES= uv run python -m fractal.viz_serve
  ATTACH          — MIRRORS a running real training from its telemetry (train.py --viz_telemetry FILE):
        VIZ_ATTACH=path/tele.json CUDA_VISIBLE_DEVICES= uv run python -m fractal.viz_serve
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fractal import persist
from fractal import tokenizer as tk
from fractal import chat_format as cf

DEV = os.environ.get("FRACTAL_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
WEB = os.path.join(os.path.dirname(__file__), "web")
CKPT = os.environ.get("FRACTAL_CKPT", "fractal_ckpt.pt")
WIN = 32                                   # tokens per stream tick
DATA_DIR = os.environ.get("VIZ_DATA_DIR", "fractal_data")
LEARN = os.environ.get("VIZ_LEARN") == "1"  # Train a small model and report sampled gradients.
ATTACH = os.environ.get("VIZ_ATTACH")       # MIRROR a real run: read telemetry from train.py (don't train)
LR = float(os.environ.get("VIZ_LR", 3e-3))
CHAT_ENABLED = os.environ.get("VIZ_CHAT", "0") == "1"
FEEDBACK_ENABLED = os.environ.get("VIZ_FEEDBACK", "0") == "1"
CHAT_STATE = os.environ.get("VIZ_CHAT_STATE", "fractal_ui_state.pt")
CHAT_SESSION = os.environ.get("VIZ_CHAT_SESSION", "fractal_ui_session.json")
FEEDBACK_WEIGHTS = os.environ.get("VIZ_FEEDBACK_WEIGHTS", CKPT + ".feedback-w0.pt")
FEEDBACK_QUEUE = os.environ.get("VIZ_FEEDBACK_QUEUE", "fractal_feedback.jsonl")
AUTH_TOKEN = os.environ.get("VIZ_AUTH_TOKEN", "")
NATURAL_SKILL_BANK = os.environ.get("VIZ_SKILL_BANK", "")
if FEEDBACK_ENABLED and not CHAT_ENABLED:
    raise SystemExit("VIZ_FEEDBACK=1 requires VIZ_CHAT=1")
if FEEDBACK_ENABLED and (ATTACH or LEARN):
    raise SystemExit("VIZ_FEEDBACK=1 is available only in checkpoint read mode")

TOK_PATH = os.environ.get("VIZ_TOKENIZER", "fractal_tokenizer.json")
tok = tk.load(TOK_PATH) if os.path.exists(TOK_PATH) else None

model = None
if ATTACH:
    # We don't load a model — just mirror the live telemetry of a running training (VIBE #1e: REAL learning).
    for _ in range(600):                    # training may not have started yet → wait for the first write (~5 min)
        if os.path.exists(ATTACH):
            break
        print(f"[viz] waiting for training telemetry: {ATTACH}", flush=True)
        time.sleep(0.5)
elif LEARN:
    # A fresh SMALL model learns FROM SCRATCH on CPU — real forward/backward/optimizer step.
    # Sample real gradients and loss from the small training process.
    from fractal.model import Config, FractalLM
    _lcfg = Config(vocab_size=int(os.environ.get("VIZ_VOCAB",
                                                  tok.get_vocab_size() if tok else 256)),
                   n_embd=int(os.environ.get("VIZ_EMBD", 256)),
                   n_head=int(os.environ.get("VIZ_HEADS", 4)),
                   depth=int(os.environ.get("VIZ_DEPTH", 6)),
                   n_scales=int(os.environ.get("VIZ_SCALES", 4)),
                   untie=os.environ.get("VIZ_TIED") != "1")
    model = FractalLM(_lcfg).to(DEV)
    model.grad_ckpt = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    train_path = os.path.join(DATA_DIR, "train.bin")
    train_data = (np.memmap(train_path, dtype=np.uint16, mode="r")
                  if os.path.exists(train_path) else None)
    learn_source = train_path if train_data is not None else "deterministic synthetic token sequences"
    print(f"[viz-learn] source: {learn_source}", flush=True)
    BLK = int(os.environ.get("VIZ_BLK", 128))
    BATCH = int(os.environ.get("VIZ_BATCH", 12))
    _loss_hist: list = []
    _iter = 0
    _prev_g = None                            # gate gradient from the previous step per (d,ℓ) — for direction stability
else:
    if tok is None:
        raise SystemExit(f"read mode requires a tokenizer: {TOK_PATH}")
    model = persist.load_model(CKPT, DEV)
    model.eval()
    if CHAT_ENABLED:
        if tok is None or tok.get_vocab_size() != model.cfg.vocab_size:
            got = None if tok is None else tok.get_vocab_size()
            raise SystemExit(f"chat tokenizer/model vocabulary mismatch: tokenizer={got}, "
                             f"model={model.cfg.vocab_size}; set VIZ_TOKENIZER to the training tokenizer")
    if FEEDBACK_ENABLED:
        from fractal import feedback
        feedback.enable(model)
        feedback.load_w0(FEEDBACK_WEIGHTS, model)
    val_path = os.path.join(DATA_DIR, "val.bin")
    if not os.path.exists(val_path):
        manifest_path = os.path.join(DATA_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            shards = manifest.get("splits", {}).get("val", {}).get("shards", [])
            if shards:
                val_path = os.path.join(DATA_DIR, shards[0]["tokens"])
    if not os.path.exists(val_path):
        raise SystemExit(f"read mode requires validation data: {val_path}")
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
if model is not None:
    unit = model.block.unit


def _atomic_json(path, payload):
    """Write private session metadata atomically beside its destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)


def _load_chat_messages():
    try:
        with open(CHAT_SESSION, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


_chat_messages = _load_chat_messages() if CHAT_ENABLED else []
_chat_states = None
_natural_runtime = None
if CHAT_ENABLED:
    if os.path.exists(CHAT_STATE):
        try:
            _chat_states = persist.load_states(CHAT_STATE, DEV)
        except (OSError, ValueError, RuntimeError):
            _chat_states = None
    if _chat_states is None:
        _chat_states = model.init_states(1, DEV)
    if NATURAL_SKILL_BANK:
        from fractal.natural_runtime import NaturalRuntimeSession, SkillBank
        _natural_bank = SkillBank(NATURAL_SKILL_BANK, model, CKPT)
        _natural_runtime = NaturalRuntimeSession(
            model, tok, _natural_bank, DEV, _chat_states, CHAT_STATE)


def _message(role, content, **extra):
    item = {"id": uuid.uuid4().hex, "role": role, "content": str(content),
            "rating": None, "revision": 0, "created_at": time.time()}
    if role in ("user", "assistant") and "rating_disabled" not in extra:
        item["rating_disabled"] = not FEEDBACK_ENABLED
    item.update(extra)
    _chat_messages.append(item)
    return item


_attach_last = None


def _attach_read():
    """Read the training telemetry (atomic write via os.replace → we read a whole/old file)."""
    global _attach_last
    try:
        with open(ATTACH) as f:
            _attach_last = json.load(f)
    except Exception:
        pass
    return _attach_last


def _taus():
    """τ ladder from the CURRENT gammas (changes during neurogenesis)."""
    return [(model.cfg.tau0 * model.cfg.rho ** l if g < 1.0 else None)
            for l, g in enumerate(unit.gammas)]


def _cfg():
    if ATTACH:
        t = _attach_read() or {}
        L = t.get("n_scales", 4)
        return {"depth": t.get("depth", 6), "n_scales": L,
                "gammas": t.get("gammas", []), "taus": t.get("taus", []),
                "ckpt": t.get("ckpt", "real run"), "n_embd": t.get("n_embd"),
                "untie": (bool(t["untie"]) if "untie" in t else None),
                "n_experts": t.get("n_experts", 1),
                "moe_mode": t.get("moe_mode", "soft"),
                "active_params": t.get("active_params"),
                "event_budget": t.get("event_budget"),
                "event_share": t.get("event_share"),
                "growing_cortex": t.get("growing_cortex"),
                "effective_depth": t.get("effective_depth", t.get("depth", 6)),
                "update_mode": t.get("update_mode"), "arm": t.get("arm"),
                "n_head": t.get("n_head"), "mlp_ratio": t.get("mlp_ratio"),  # None on older telemetry → client default
                "batch": t.get("batch"), "block": t.get("block"),            # → client tok/s (None on older telemetry)
                "tokens_per_step": t.get("tokens_per_step"),
                "learning_signal": t.get("learning_signal", "gradient"),
                "vram_gb": t.get("peak_vram_gb", t.get("vram_gb")),
                "params": t.get("params"), "mode": "attach", "lr": t.get("lr")}
    return {"depth": model.cfg.depth, "n_scales": model.cfg.n_scales,
            "gammas": [float(g) for g in unit.gammas], "taus": _taus(),
            "ckpt": "learning from scratch" if LEARN else os.path.basename(CKPT), "n_embd": model.cfg.n_embd,
            "untie": bool(model.cfg.untie), "n_experts": getattr(model.cfg, "n_experts", 1),
            "moe_mode": getattr(model.cfg, "moe_mode", "soft"),
            "active_params": round(model.parameter_counts()[1] / 1e6, 2),
            "event_budget": getattr(model.cfg, "event_budget", 1.0),
            "n_head": model.cfg.n_head, "mlp_ratio": getattr(model.cfg, "mlp_ratio", 2),
            "params": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "event_algebra": bool(getattr(model.cfg, "event_algebra", False)),
            "growing_cortex": (
                None if model.skill_cortex is None else model.skill_cortex.snapshot()),
            "chat_enabled": CHAT_ENABLED,
            "feedback_enabled": FEEDBACK_ENABLED,
            "batch": (BATCH if LEARN else None), "block": (BLK if LEARN else None),  # → client tok/s (learn mode)
            "vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if DEV.startswith("cuda") else None),
            "mode": "learn" if LEARN else "read", "lr": LR if LEARN else None}


CFG = _cfg()

_lock = threading.Lock()
_pos = 0
_states = None
_runmax = 1e-9
_prev_delta = None                 # ΔW of the previous tick per (d,ℓ) — for resonance
_attach_hist: list = []            # ATTACH: loss history (curve), grows when the iteration changes
_attach_last_iter = -1
_attach_runmax = 1e-9
_attach_mtime = 0.0                 # ATTACH: last-seen telemetry mtime → staleness ("is the run still live?")
_attach_interval = 0.0             # ATTACH: EMA of seconds between telemetry writes → adaptive stale threshold

# ---- neurogenesis LIVE (optional): scales grow directly while reading the stream ----
# VIZ_NEUROGENESIS=1 → controller runs over the inference stream; on saturation a new
# scale is added to every recurrent unit. Paced in TICKS (WIN tokens/tick).
NEURO = None
_tick = 0
_born_pending = False
if os.environ.get("VIZ_NEUROGENESIS") == "1" and model is not None:
    from fractal import neurogenesis as ng
    for _u in ng.units(model):
        _u._log_share = True
    # This demonstration uses an intentionally eager threshold. Rigorous growth evidence comes from
    # the separate A/B training test with predeclared thresholds.
    NEURO = ng.NeurogenesisController(
        max_scales=int(os.environ.get("VIZ_MAX_SCALES", 5)),
        conc_thresh=float(os.environ.get("VIZ_GROW_CONC", 0.18)),
        plateau_eps=float(os.environ.get("VIZ_GROW_PLATEAU", 0.08)),
        cooldown=int(os.environ.get("VIZ_COOLDOWN", 18)), warmup=12,
        birth_beta_gain=2.5, mature_steps=40)
    print(f"[viz] neurogenesis LIVE ON (start {model.cfg.n_scales} → max {NEURO.max_scales} scales)", flush=True)

# read gate: hook on each UNIQUE gate module; calls arrive in depth order,
# so the index in _gcalls = depth (works for tied and untied). After growth the gate module
# is REPLACED → hooks must be re-registered (otherwise _gcalls stays empty).
_gcalls: list = []


def _register_gate_hooks():
    for g in {id(model.block_at(d).unit.gate): model.block_at(d).unit.gate
              for d in range(model.cfg.depth)}.values():
        g.register_forward_hook(lambda m, i, o: _gcalls.append(o.detach()))


if not LEARN and not ATTACH:  # in learn/mirror mode we don't hook the gate (otherwise the list would grow / model missing)
    _register_gate_hooks()


def _cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _grad_parts(m):
    """CHEAP per-depth component signal for the machine cutaway: norms of gradients that
    already exist after .backward() (to_qk, to_beta, proj, MLP/router) — no extra compute."""
    def _n(p):
        return float(p.grad.norm()) if (p is not None and p.grad is not None) else 0.0
    def _module_n(module):
        if module is None:
            return 0.0
        return sum(_n(parameter) ** 2 for parameter in module.parameters()) ** 0.5
    parts = {"qk": [], "beta": [], "proj": [], "mlp": [], "skill": []}
    for d in range(m.cfg.depth):
        u = m.block_at(d).unit
        mlp = m.block_at(d).mlp
        parts["qk"].append(round(_n(u.to_qk.weight), 5))
        parts["beta"].append(round(_n(u.to_beta.weight), 5))
        parts["proj"].append(round(_n(u.proj.weight), 5))
        if hasattr(mlp, "router"):                    # MoE: router = where the mixture is learning
            parts["mlp"].append(round(_n(mlp.router.weight), 5))
        else:
            parts["mlp"].append(round((_n(mlp.fc.weight) ** 2 + _n(mlp.proj.weight) ** 2) ** 0.5, 5))
        skill = m.skill_cortex
        parts["skill"].append(round(
            0.0 if skill is None else (
                _module_n(skill.compiler) ** 2 + _module_n(skill.query_proj) ** 2
            ) ** 0.5, 5))
    return parts


@torch.no_grad()
def activity():
    """One stream tick: the NEXT corpus window passes through the model with CARRIED state
    (persistent memory as in production — no resets, no random jumps).
    All real quantities: ‖ΔW‖ (write), gate (read), ‖W‖ (memory fill),
    resonance = cos(ΔW_t, ΔW_t−1) (directional write stability), window loss (surprise)."""
    global _pos, _states, _runmax, _prev_delta, _tick, _born_pending
    with _lock:
        _tick += 1
        if _pos + WIN + 1 >= len(data):
            _pos = 0
        span = torch.from_numpy(data[_pos:_pos + WIN + 1].astype(np.int64))[None].to(DEV)
        ids, targets = span[:, :-1], span[:, 1:]
        _pos += WIN
        if _states is None:
            _states = model.init_states(1, DEV)
        prevW = [[w.clone() for w in st.W] for st in _states]
        _gcalls.clear()
        t0 = time.perf_counter()
        _, loss, _states, all_dn = model(ids, targets=targets, states=_states,
                                         return_delta=True)
        ms = (time.perf_counter() - t0) * 1000
        D, L, H = model.cfg.depth, model.cfg.n_scales, model.cfg.n_head

        dW = [[float(all_dn[d][l].mean()) for l in range(L)] for d in range(D)]
        _runmax = max(_runmax * 0.995, max(v for row in dW for v in row))

        delta = [[_states[d].W[l] - prevW[d][l] for l in range(L)] for d in range(D)]
        ok_prev = (_prev_delta is not None and len(_prev_delta) == D
                   and len(_prev_delta[0]) == L)                 # after growth the number of scales changes
        res = ([[round(_cos(delta[d][l], _prev_delta[d][l]), 3) for l in range(L)]
                for d in range(D)] if ok_prev else [[0.0] * L for _ in range(D)])
        _prev_delta = delta

        wn = [[float(_states[d].W[l].norm()) for l in range(L)] for d in range(D)]
        wmax = max((v for row in wn for v in row), default=1.0) or 1.0
        en = [[float(_states[d].eligibility[l].norm()) for l in range(L)]
              for d in range(D)] if _states[0].eligibility is not None else None
        emax = max((v for row in (en or []) for v in row), default=1.0) or 1.0

        gate = [[round(float(g), 3) for g in
                 _gcalls[d].view(-1, H, L).softmax(-1).mean(dim=(0, 1))]
                for d in range(D)] if len(_gcalls) == D else None

        born = _born_pending          # birth from the previous tick → this payload already has the new scale count
        _born_pending = False
        if NEURO is not None:         # LIVE growth over the inference stream
            NEURO.mature(model)
            share, wnorm = ng.telemetry(model, _states)
            NEURO.observe(share, wnorm)
            birth = NEURO.maybe_grow(model, _tick)
            if birth is not None:
                for d in range(D):    # migrate carried state: append the empty W of the new scale
                    _states[d].W.append(model.block_at(d).unit.cells[-1].init_state(1, DEV))
                    if _states[d].eligibility is not None:
                        _states[d].eligibility.append(torch.zeros_like(_states[d].W[-1]))
                _register_gate_hooks()
                _prev_delta = None
                _born_pending = True
                print(f"[viz] 🌱 birth @ tick {_tick} → {model.cfg.n_scales} scales "
                      f"(dominant L{birth['dominant']} conc {birth['conc']}, γ {birth['birth_gamma']})", flush=True)

        lv = float(loss)
        return {"dW": [[round(min(v / _runmax, 1.0), 3) for v in row] for row in dW],
                "Wn": [[round(v / wmax, 3) for v in row] for row in wn],
                "En": ([[round(v / emax, 3) for v in row] for row in en]
                       if en is not None else None),
                "res": res, "gate": gate,
                "loss": round(lv, 3), "ppl": round(math.exp(min(lv, 20)), 1),
                "ms": round(ms), "text": tok.decode(ids[0].tolist()),
                "pos": int(_pos), "total": int(len(data)),
                "vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if DEV.startswith("cuda") else None),
                "n_scales": L,          # scale count CONSISTENT with this tick's arrays (after birth it jumps next tick)
                "gammas": [round(float(g), 4) for g in unit.gammas[:L]], "taus": _taus()[:L],
                "born": born,
                "growing_cortex": (
                    None if model.skill_cortex is None else model.skill_cortex.snapshot())}


def learn_step():
    """One real learning step over local corpus data or a deterministic synthetic sequence.

    The payload contains sampled per-depth gradient magnitudes, parameter norms, direction stability,
    and measured loss. Visualization adds no extra forward or backward pass."""
    global _iter, _prev_g, _runmax
    with _lock:
        _iter += 1
        if train_data is not None:
            n = len(train_data)
            ix = np.random.randint(0, n - BLK - 1, size=BATCH)
            xb = np.stack([train_data[i:i + BLK].astype(np.int64) for i in ix])
            yb = np.stack([train_data[i + 1:i + 1 + BLK].astype(np.int64) for i in ix])
            x = torch.from_numpy(xb).to(DEV)
            y = torch.from_numpy(yb).to(DEV)
            text = tok.decode(x[0][:WIN].tolist()) if tok else "local corpus tokens"
            batch_kind = "local corpus"
        else:
            start = torch.arange(BATCH, device=DEV)[:, None] * 17
            offsets = torch.arange(BLK + 1, device=DEV)[None, :]
            sequence = (start + offsets) % model.cfg.vocab_size
            x, y = sequence[:, :-1], sequence[:, 1:]
            text = "deterministic synthetic token sequence"
            batch_kind = "synthetic sequence"
        t0 = time.perf_counter()
        _, loss, _, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        D, L, H = model.cfg.depth, model.cfg.n_scales, model.cfg.n_head

        # per (depth, scale) gradient and norm of gate weight (Linear n_embd → H·L logits)
        grad = [[0.0] * L for _ in range(D)]
        wn = [[0.0] * L for _ in range(D)]
        gvec = [[None] * L for _ in range(D)]
        for d in range(D):
            g = model.block_at(d).unit.gate
            gg = g.weight.grad.view(H, L, -1)             # (H, L, n_embd)
            gw = g.weight.data.view(H, L, -1)
            for l in range(L):
                grad[d][l] = float(gg[:, l, :].norm())
                wn[d][l] = float(gw[:, l, :].norm())
                gvec[d][l] = gg[:, l, :].reshape(-1)

        ok = _prev_g is not None and len(_prev_g) == D and len(_prev_g[0]) == L
        res = ([[round(_cos(gvec[d][l], _prev_g[d][l]), 3) for l in range(L)] for d in range(D)]
               if ok else [[0.0] * L for _ in range(D)])
        _prev_g = gvec

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))  # measure + stabilize
        for pg in opt.param_groups:                       # short warmup → smooth start without spikes
            pg["lr"] = LR * min(1.0, _iter / 40.0)
        opt.step()
        ms = (time.perf_counter() - t0) * 1000

        gmax = _runmax = max(_runmax * 0.98, max(v for row in grad for v in row), 1e-9)
        wmax = max((v for row in wn for v in row), default=1.0) or 1.0
        lv = float(loss.detach())
        _loss_hist.append(round(lv, 3))
        if len(_loss_hist) > 120:
            _loss_hist.pop(0)
        if _iter % 25 == 0:
            print(f"[viz-learn] step {_iter}  loss {lv:.3f}  ppl {math.exp(min(lv,20)):.1f}  ‖∇‖ {gnorm:.2f}", flush=True)
        return {"dW": [[round(min(v / gmax, 1.0), 3) for v in row] for row in grad],  # brightness = ‖∇‖ (learning)
                "Wn": [[round(v / wmax, 3) for v in row] for row in wn],
                "res": res, "gate": None, "parts": _grad_parts(model),
                "loss": round(lv, 3), "ppl": round(math.exp(min(lv, 20)), 1),
                "ms": round(ms), "text": text, "batch_kind": batch_kind,
                "iter": _iter, "gnorm": round(gnorm, 3), "loss_hist": list(_loss_hist),
                "vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if DEV.startswith("cuda") else None),
                "n_scales": L, "gammas": [round(float(g), 4) for g in unit.gammas[:L]],
                "taus": _taus()[:L], "born": False,
                "growing_cortex": (
                    None if model.skill_cortex is None else model.skill_cortex.snapshot())}


def attach_activity():
    """MIRROR a real run: read telemetry from train.py and remap it onto the dashboard payload.
    We neither train nor infer — we just faithfully show the REAL learning of a running training."""
    global _attach_hist, _attach_last_iter, _attach_runmax, _attach_mtime, _attach_interval
    with _lock:
        # FRESHNESS: how long since train.py last wrote telemetry. This is the robust "is the run
        # still live?" signal (independent of the telemetry's own write cadence): a finished/stopped
        # run stops touching the file, so `age` grows without bound → stale → dashboard shows idle.
        try:
            mtime = os.path.getmtime(ATTACH)
        except OSError:
            mtime = 0.0
        age = (time.time() - mtime) if mtime else 1e9
        if mtime and mtime != _attach_mtime:
            if _attach_mtime:                       # learn the typical write interval (adaptive threshold)
                gap = mtime - _attach_mtime
                _attach_interval = gap if _attach_interval <= 0 else _attach_interval * 0.6 + gap * 0.4
            _attach_mtime = mtime
        stale = age > max(12.0, _attach_interval * 4.0)   # no fresh write for several write-intervals
        fresh = {"age": round(age, 1), "stale": bool(stale), "live": (not stale)}

        t = _attach_read()
        if t and t.get("learning_signal") == "local_update":
            D, L = int(t.get("depth", 1)), int(t.get("n_scales", 1))
            fast = t.get("fast_update_norms") or [[0.0] * L for _ in range(D)]
            fast = [(row + [0.0] * L)[:L] for row in (fast + [[0.0] * L] * D)[:D]]
            fmax = max((v for row in fast for v in row), default=0.0) or 1.0
            it, loss = t.get("iter", 0), t.get("loss", 0.0)
            if it != _attach_last_iter:
                _attach_hist.append(round(loss, 3))
                if len(_attach_hist) > 120:
                    _attach_hist.pop(0)
                _attach_last_iter = it
            updates = t.get("update_norms") or {}
            parts = {
                "qk": [float(updates.get("qk", 0.0))],
                "beta": [float(updates.get("routing", 0.0))],
                "proj": [float(updates.get("projection", 0.0))],
                "mlp": [float(updates.get("mlp", 0.0))],
            }
            update_l2 = math.sqrt(sum(float(v) ** 2 for v in updates.values()))
            return {"dW": [[round(v / fmax, 3) for v in row] for row in fast],
                    "Wn": [[0.0] * L for _ in range(D)], "res": [[0.0] * L for _ in range(D)],
                    "gate": None, "parts": parts, "loss": round(loss, 3),
                    "ppl": round(math.exp(min(loss, 20)), 1), "ms": 0, "text": "",
                    "iter": it, "gnorm": round(update_l2, 6), "loss_hist": list(_attach_hist),
                    "n_scales": L, "gammas": (t.get("gammas") or [])[:L],
                    "taus": (t.get("taus") or [])[:L], "born": False,
                    "effective_depth": D, "update_mode": t.get("update_mode"),
                    "learning_signal": "local_update", "fitness": t.get("fitness"), **fresh}
        if not t or "grad" not in t:               # training hasn't written yet / shortly after start
            return {"dW": [[0.0]], "Wn": [[0.0]], "res": [[0.0]], "gate": None, "parts": None,
                    "loss": 0.0, "ppl": 0.0, "ms": 0, "text": "waiting for training…",
                    "iter": 0, "gnorm": 0.0, "loss_hist": [], "n_scales": 1,
                    "gammas": [1.0], "taus": [None], "born": False, "waiting": True, **fresh}
        D, L = t["depth"], t["n_scales"]
        grad, wn = t["grad"], t["wn"]
        res = t.get("res", [[0.0] * L for _ in range(D)])
        _attach_runmax = max(_attach_runmax * 0.98,
                             max((v for row in grad for v in row), default=0.0), 1e-9)
        gmax = _attach_runmax
        wmax = max((v for row in wn for v in row), default=1.0) or 1.0
        it, loss = t.get("iter", 0), t.get("loss", 0.0)
        if it != _attach_last_iter:                # new iteration → add a point to the curve
            _attach_hist.append(round(loss, 3))
            if len(_attach_hist) > 120:
                _attach_hist.pop(0)
            _attach_last_iter = it
        return {"dW": [[round(min(v / gmax, 1.0), 3) for v in row] for row in grad],
                "Wn": [[round(v / wmax, 3) for v in row] for row in wn],
                "res": res, "gate": None,
                "parts": t.get("parts"),           # per-component ‖∇‖ (newer runs); None on old telemetry
                "loss": round(loss, 3), "ppl": round(math.exp(min(loss, 20)), 1),
                "ms": 0, "text": t.get("text", ""), "iter": it,
                "batch_kind": t.get("batch_kind"),
                "gnorm": t.get("gnorm", 0.0), "loss_hist": list(_attach_hist),
                "n_scales": L, "gammas": (t.get("gammas") or [])[:L],
                "taus": (t.get("taus") or [])[:L], "born": False,
                "effective_depth": t.get("effective_depth", D),
                "event_share": t.get("event_share"),
                "update_mode": t.get("update_mode"),
                "expert_usage": t.get("expert_usage"),
                "selected_expert": t.get("selected_expert"),
                "growing_cortex": t.get("growing_cortex"),
                "vram_gb": t.get("peak_vram_gb", t.get("vram_gb")), **fresh}


_last_feedback = None


def _chat_turn(text):
    """Run one serialized persistent agent turn and durably save its private session state."""
    global _chat_states
    if not CHAT_ENABLED:
        return 409, {"error": "chat is disabled in this dashboard mode"}
    clean = str(text).strip()
    if not clean:
        return 400, {"error": "message must not be empty"}
    if len(clean.encode("utf-8")) > 32_768:
        return 413, {"error": "message exceeds the 32 KiB limit"}
    from fractal import agent
    with _lock:
        created = [_message("user", clean)]
        generation = {
            "max_new": int(os.environ.get("VIZ_CHAT_MAX_NEW", 200)),
            "max_tool_calls": int(os.environ.get("VIZ_CHAT_MAX_TOOLS", 6)),
            "temperature": float(os.environ.get("VIZ_CHAT_TEMPERATURE", 0.8)),
            "top_k": int(os.environ.get("VIZ_CHAT_TOP_K", 40)),
            "json_guard": os.environ.get("VIZ_JSON_GUARD") == "1",
        }
        if _natural_runtime is not None:
            result = _natural_runtime.chat(clean, **generation)
            transcript = result["transcript"]
            _chat_states = _natural_runtime.states
        else:
            transcript, _chat_states = agent.run_turn(
                model, tok, _chat_states, clean, DEV, **generation)
        for role, content in transcript:
            if role == "assistant":
                created.append(_message("assistant", content))
            elif role in ("tool_call", "tool_result", "note"):
                rendered = (json.dumps(content, ensure_ascii=False)
                            if not isinstance(content, str) else content)
                created.append(_message(role, rendered, rating_disabled=True))
        persist.save_states(CHAT_STATE, _chat_states)
        _atomic_json(CHAT_SESSION, _chat_messages)
    return 200, {"messages": created}


def _feedback_text(item):
    marker = cf.USER if item["role"] == "user" else cf.ASSISTANT
    return f"{marker}\n{item['content']}"


def _submit_feedback(message_id, rating, requested_revision=None):
    """Apply one user-authority rating revision exactly once."""
    global _last_feedback
    if not FEEDBACK_ENABLED:
        return 409, {"error": "feedback is disabled in this dashboard mode"}
    try:
        new_credit = feedback.credit_for_rating(rating)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    with _lock:
        item = next((m for m in _chat_messages if m.get("id") == message_id), None)
        if item is None:
            return 404, {"error": "message was not found"}
        if item.get("rating_disabled") or item.get("role") not in ("user", "assistant"):
            return 400, {"error": "this message cannot be rated"}
        current_revision = int(item.get("revision", 0))
        if requested_revision is not None and int(requested_revision) != current_revision + 1:
            if int(item.get("rating") or 0) == int(rating) and int(requested_revision) <= current_revision:
                return 200, {"message": item, "duplicate": True, "feedback": _last_feedback}
            return 409, {"error": "stale feedback revision", "message": item}
        old_rating = item.get("rating")
        if old_rating == int(rating):
            return 200, {"message": item, "duplicate": True, "feedback": _last_feedback}
        old_credit = feedback.credit_for_rating(old_rating) if old_rating is not None else 0.0
        credit_delta = new_credit - old_credit
        token_ids = tok.encode(_feedback_text(item)).ids
        evidence = feedback.message_eligibility(model, token_ids, DEV)
        fast_norm = feedback.apply_to_state(_chat_states, evidence, credit_delta)
        w0_norm = 0.0
        if credit_delta:
            feedback.save_w0(FEEDBACK_WEIGHTS + ".rollback", model)
            w0_norm = feedback.consolidate_w0(model, evidence, credit_delta)
            feedback.save_w0(FEEDBACK_WEIGHTS, model)
        item["rating"] = int(rating)
        item["revision"] = current_revision + 1
        item["rated_at"] = time.time()
        item["feedback_source"] = "user"
        persist.save_states(CHAT_STATE, _chat_states)
        _atomic_json(CHAT_SESSION, _chat_messages)
        _last_feedback = {
            "message_id": message_id, "rating": int(rating), "credit_delta": credit_delta,
            "fast_update_norm": fast_norm, "w0_update_norm": w0_norm,
            "eligibility_norm": sum(float(e.norm()) for s in evidence
                                    for e in (s.eligibility or [])),
            "source": "user", "time": time.time(),
        }
        feedback.append_event(FEEDBACK_QUEUE, {
            "event_id": uuid.uuid4().hex, "message_id": message_id,
            "message_revision": item["revision"], "role": item["role"],
            "content": item["content"], "rating": int(rating),
            "credit_delta": credit_delta, "source": "user", "created_at": time.time(),
        })
        return 200, {"message": item, "feedback": _last_feedback}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _authorized(self):
        if not AUTH_TOKEN:
            return True
        supplied = self.headers.get("Authorization", "")
        return supplied == f"Bearer {AUTH_TOKEN}" or self.headers.get("X-RTAI-Token") == AUTH_TOKEN

    def _body(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("invalid content length")
        if size <= 0 or size > 65_536:
            raise ValueError("request body must be between 1 byte and 64 KiB")
        try:
            value = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request JSON must be an object")
        return value

    def do_GET(self):
        p = self.path.split("?")[0]
        if p not in ("/", "/fractal3d.html") and not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if p in ("/", "/fractal3d.html"):
            with open(os.path.join(WEB, "fractal3d.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif p == "/config":
            self._send(200, json.dumps(CFG).encode(), "application/json")
        elif p == "/activity":
            payload = attach_activity() if ATTACH else (learn_step() if LEARN else activity())
            payload["feedback"] = _last_feedback
            payload["autonomous_credit"] = (getattr(model, "_last_autonomous_credit", None)
                                                if model is not None else None)
            payload["skill_route"] = (getattr(model, "_last_skill_route", None)
                                       if model is not None else None)
            payload["natural_runtime"] = (
                None if _natural_runtime is None else _natural_runtime.snapshot())
            if payload.get("growing_cortex") is None and model is not None \
                    and model.skill_cortex is not None:
                payload["growing_cortex"] = model.skill_cortex.snapshot()
            if payload["autonomous_credit"] is not None:
                payload["update_mode"] = "predictive-event-credit"
            self._send(200, json.dumps(payload).encode(), "application/json")
        elif p == "/api/session":
            if not self._authorized():
                self._json(401, {"error": "unauthorized"})
            else:
                self._json(200, {"enabled": CHAT_ENABLED, "messages": _chat_messages,
                                 "checkpoint": os.path.basename(CKPT),
                                 "feedback": _last_feedback,
                                 "growing_cortex": (
                                     None if model is None or model.skill_cortex is None
                                     else model.skill_cortex.snapshot()),
                                 "natural_runtime": (
                                     None if _natural_runtime is None
                                     else _natural_runtime.snapshot())})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        p = self.path.split("?")[0]
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        if p == "/api/chat":
            code, payload = _chat_turn(body.get("message", ""))
            self._json(code, payload)
        elif p == "/api/feedback":
            code, payload = _submit_feedback(body.get("message_id"), body.get("rating"),
                                             body.get("revision"))
            self._json(code, payload)
        elif p.startswith("/api/skill/"):
            if _natural_runtime is None:
                self._json(409, {"error": "Natural Cortex skill runtime is disabled"})
                return
            try:
                if p == "/api/skill/propose":
                    payload = _natural_runtime.propose_skill(body.get("text", ""))
                elif p == "/api/skill/suggest":
                    payload = _natural_runtime.suggest_skill(
                        body.get("user", ""), body.get("assistant", ""))
                elif p == "/api/skill/activate":
                    payload = _natural_runtime.activate(
                        int(body["expert_id"]), confirmed=bool(body.get("confirmed")))
                elif p == "/api/skill/teach":
                    payload = _natural_runtime.teach(
                        body.get("name", ""), body.get("synopsis", ""),
                        body.get("demonstrations") or [],
                        confirmed=bool(body.get("confirmed")),
                        anchors=body.get("anchors") or [],
                        steps=int(body.get("steps", 64)),
                        lr=float(body.get("lr", 1e-2)),
                    )
                elif p == "/api/skill/rate":
                    payload = _natural_runtime.rate(int(body["rating"]))
                elif p == "/api/skill/quarantine":
                    payload = _natural_runtime.quarantine(int(body["expert_id"]))
                elif p == "/api/skill/rollback":
                    payload = _natural_runtime.rollback(int(body["expert_id"]))
                elif p == "/api/skill/restart-verification":
                    payload = _natural_runtime.restart_verification(CKPT, CHAT_STATE)
                elif p == "/api/skill/calibrate-addresses":
                    payload = _natural_runtime.calibrate_addresses(
                        body.get("examples") or [],
                        steps=int(body.get("steps", 200)),
                        lr=float(body.get("lr", 1e-3)),
                    )
                else:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, payload)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                self._json(400, {"error": str(exc)})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not AUTH_TOKEN:
        raise SystemExit("VIZ_AUTH_TOKEN is required when HOST is not loopback")
    src = (f"ATTACH ↔ {ATTACH}" if ATTACH else
           ("LEARN (learning from scratch)" if LEARN else f"model {CKPT}"))
    print(f"FractalLM telemetry: http://{host}:{port}   [{src}, dev: {DEV}]  Ctrl-C to quit",
          flush=True)
    with ThreadingHTTPServer((host, port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
