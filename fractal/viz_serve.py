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
import threading
import time

import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fractal import persist
from fractal import tokenizer as tk

DEV = os.environ.get("FRACTAL_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
WEB = os.path.join(os.path.dirname(__file__), "web")
CKPT = os.environ.get("FRACTAL_CKPT", "fractal_ckpt.pt")
WIN = 32                                   # tokens per stream tick
DATA_DIR = os.environ.get("VIZ_DATA_DIR", "fractal_data")
LEARN = os.environ.get("VIZ_LEARN") == "1"  # Train a small model and report sampled gradients.
ATTACH = os.environ.get("VIZ_ATTACH")       # MIRROR a real run: read telemetry from train.py (don't train)
LR = float(os.environ.get("VIZ_LR", 3e-3))

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
    val_path = os.path.join(DATA_DIR, "val.bin")
    if not os.path.exists(val_path):
        raise SystemExit(f"read mode requires validation data: {val_path}")
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
if model is not None:
    unit = model.block.unit


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
    parts = {"qk": [], "beta": [], "proj": [], "mlp": []}
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
                _register_gate_hooks()
                _prev_delta = None
                _born_pending = True
                print(f"[viz] 🌱 birth @ tick {_tick} → {model.cfg.n_scales} scales "
                      f"(dominant L{birth['dominant']} conc {birth['conc']}, γ {birth['birth_gamma']})", flush=True)

        lv = float(loss)
        return {"dW": [[round(min(v / _runmax, 1.0), 3) for v in row] for row in dW],
                "Wn": [[round(v / wmax, 3) for v in row] for row in wn],
                "res": res, "gate": gate,
                "loss": round(lv, 3), "ppl": round(math.exp(min(lv, 20)), 1),
                "ms": round(ms), "text": tok.decode(ids[0].tolist()),
                "pos": int(_pos), "total": int(len(data)),
                "vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if DEV.startswith("cuda") else None),
                "n_scales": L,          # scale count CONSISTENT with this tick's arrays (after birth it jumps next tick)
                "gammas": [round(float(g), 4) for g in unit.gammas[:L]], "taus": _taus()[:L],
                "born": born}


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
                "taus": _taus()[:L], "born": False}


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
                "vram_gb": t.get("peak_vram_gb", t.get("vram_gb")), **fresh}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/fractal3d.html"):
            with open(os.path.join(WEB, "fractal3d.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif p == "/config":
            self._send(200, json.dumps(CFG).encode(), "application/json")
        elif p == "/activity":
            payload = attach_activity() if ATTACH else (learn_step() if LEARN else activity())
            self._send(200, json.dumps(payload).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    src = (f"ATTACH ↔ {ATTACH}" if ATTACH else
           ("LEARN (learning from scratch)" if LEARN else f"model {CKPT}"))
    print(f"FractalLM telemetry: http://localhost:{port}   [{src}, dev: {DEV}]  Ctrl-C to quit",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
