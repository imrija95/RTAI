"""Live server: keeps a running model + persistent W in memory and exposes an HTTP API,
so a browser frontend can visualize in real time how W self-modifies.

    uv run python -m rtai.serve --ckpt ckpt.pt        # → http://localhost:8000

Endpoints:
  GET  /                → frontend (rtai/web/index.html)
  GET  /api/state       → current W (all layers/heads) + integrity
  POST /api/teach {key,value}  → teach (COMMITs into W), returns trajectory + integrity
  POST /api/ask   {key}        → query (read-only, does NOT commit), returns prediction + distribution
  POST /api/reset              → reset W to W0
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from .data_recall import RecallVocab, encode_query
from .monitor import state_integrity
from . import state as st

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# --- global server state ---
DEVICE = None
MODEL = None
VOCAB = RecallVocab()
STATES = None
LOCK = threading.Lock()
MAX_REQUEST_BYTES = 16 * 1024


def w_payload(states):
    """W of all layers/heads, rounded for JSON. layers[l][h] = an hd×hd matrix."""
    out = []
    for W in states:                       # (1,H,hd,hd)
        Wh = W[0]
        out.append([[[round(float(v), 3) for v in row] for row in Wh[h].cpu().tolist()]
                    for h in range(Wh.shape[0])])
    return out


def integrity_payload(states):
    info = state_integrity(states)
    return [{"layer": i, **{k: (round(m[k], 3) if isinstance(m[k], float) else m[k])
                            for k in ("fro", "spec", "finite")}}
            for i, (_, m) in enumerate(info.items())]


def do_teach(key, value):
    """Process [key, value] from the current state, COMMIT. Return a 2-step trajectory."""
    global STATES
    seq = [VOCAB.key_tok(key), VOCAB.val_tok(value)]
    base = [s.clone() for s in STATES]
    traj, deltas = [], []
    prev = base
    final = None
    for L in range(1, len(seq) + 1):
        x = torch.tensor([seq[:L]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            _, _, out_states, _ = MODEL(x, states=[b.clone() for b in base])
        traj.append(w_payload(out_states))
        deltas.append([round(float((o[0] - p[0]).norm().item()), 4)
                       for o, p in zip(out_states, prev)])
        prev = out_states
        final = out_states
    STATES = final
    return {"traj": traj, "deltas": deltas, "integrity": integrity_payload(STATES)}


def do_ask(key):
    """Query — read-only (on a copy of the state, does NOT commit)."""
    q = encode_query(key, VOCAB, DEVICE)
    with torch.no_grad():
        logits, _, _, _ = MODEL(q, states=[s.clone() for s in STATES])
    last = logits[0, -1]
    vals = last[VOCAB.VAL0:VOCAB.VAL0 + VOCAB.n_vals]
    dist = F.softmax(vals, dim=-1)
    pred = int(dist.argmax().item())
    return {"key": key, "pred": pred,
            "conf": round(float(dist.max().item()), 3),
            "dist": [round(float(v), 4) for v in dist.tolist()]}


def do_reset():
    global STATES
    STATES = MODEL.init_states(1, DEVICE)
    return {"ok": True, "integrity": integrity_payload(STATES)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be at most {MAX_REQUEST_BYTES} bytes")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    @staticmethod
    def _bounded_int(body, name, upper):
        value = int(body[name])
        if not 0 <= value < upper:
            raise ValueError(f"{name} must be in [0, {upper})")
        return value

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with LOCK:
                self._json({"config": {"n_layer": MODEL.cfg.n_layer,
                                       "n_head": MODEL.cfg.n_head,
                                       "hd": MODEL.cfg.head_dim,
                                       "n_keys": VOCAB.n_keys, "n_vals": VOCAB.n_vals},
                            "W": w_payload(STATES),
                            "integrity": integrity_payload(STATES)})
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/teach", "/api/ask", "/api/reset"):
            self._send(404, "not found", "text/plain")
            return
        try:
            body = self._read_json()
            with LOCK:
                if self.path == "/api/teach":
                    key = self._bounded_int(body, "key", VOCAB.n_keys)
                    value = self._bounded_int(body, "value", VOCAB.n_vals)
                    self._json(do_teach(key, value))
                elif self.path == "/api/ask":
                    self._json(do_ask(self._bounded_int(body, "key", VOCAB.n_keys)))
                else:
                    self._json(do_reset())
        except (KeyError, TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, code=400)


def main():
    global DEVICE, MODEL, STATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = st.load_model(args.ckpt, DEVICE)
    STATES = MODEL.init_states(1, DEVICE)
    print(f"[serve] model loaded ({DEVICE}), W reset.")
    print(f"[serve] open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
