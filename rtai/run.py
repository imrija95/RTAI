"""Deploy / acid test across processes.

    uv run python -m rtai.run teach --key 3 --value 7     # teach, save W to disk
    uv run python -m rtai.run ask  --key 3                # NEW process: load W, answer
    uv run python -m rtai.run ask  --key 3 --no-state     # ablation: without persistence
    uv run python -m rtai.run status                      # integrity of the persistent W

The W state lives in a file (--state), so it survives a process restart = persistent learning.
No gating — it just prints integrity; snapshot for manual rollback: --snapshot.
"""

from __future__ import annotations

import argparse
import os

import torch

from .data_recall import RecallVocab, encode_teach, encode_query
from .monitor import state_integrity, format_integrity
from . import state as st


def _load(args, device):
    model = st.load_model(args.ckpt, device)
    vocab = RecallVocab()
    if getattr(args, "no_state", False) or not os.path.exists(args.state):
        states = model.init_states(1, device)
    else:
        states = st.load_state(args.state, device)
    return model, vocab, states


def cmd_teach(args, device):
    model, vocab, states = _load(args, device)
    teach = encode_teach([(args.key, args.value)], vocab, device)
    with torch.no_grad():
        _, _, states, _ = model(teach, states=states)
    st.save_state(args.state, states)
    if args.snapshot:
        path = st.snapshot_state(states, args.snap_dir, tag=f"k{args.key}v{args.value}")
        print(f"snapshot: {path}")
    print(f"taught: key={args.key} → value={args.value}; W saved to {args.state}")
    print(format_integrity(state_integrity(states)))


def cmd_ask(args, device):
    model, vocab, states = _load(args, device)
    q = encode_query(args.key, vocab, device)
    with torch.no_grad():
        logits, _, _, _ = model(q, states=states)
    tok = logits[0, -1].argmax().item()
    if vocab.VAL0 <= tok < vocab.VAL0 + vocab.n_vals:
        print(f"answer for key={args.key}: value={vocab.val_index(tok)}")
    else:
        print(f"answer for key={args.key}: (untokenized/out of range, tok={tok})")
    src = "W0 (no persistence)" if getattr(args, "no_state", False) else args.state
    print(f"[state source: {src}]")


def cmd_chat(args, device):
    """Interactive 'chat' in the recall language. The W memory lives in the session and on disk."""
    model, vocab, states = _load(args, device)
    src = "empty (W0)" if (getattr(args, "no_state", False) or
                           not os.path.exists(args.state)) else args.state
    print(f"RTAI chat — memory: {src}")
    print("commands: teach K V | ask K | status | reset | quit  "
          f"(K=0..{vocab.n_keys-1}, V=0..{vocab.n_vals-1})")
    while True:
        try:
            line = input("rtai> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        p = line.split()
        c = p[0].lower()
        if c in ("quit", "exit", "q"):
            break
        elif c == "teach" and len(p) == 3:
            k, v = int(p[1]), int(p[2])
            teach = encode_teach([(k, v)], vocab, device)
            with torch.no_grad():
                _, _, states, _ = model(teach, states=states)
            st.save_state(args.state, states)
            print(f"  taught: {k} → {v}  (W saved to {args.state})")
        elif c == "ask" and len(p) == 2:
            k = int(p[1])
            q = encode_query(k, vocab, device)
            with torch.no_grad():
                logits, _, _, _ = model(q, states=states)
            tok = logits[0, -1].argmax().item()
            if vocab.VAL0 <= tok < vocab.VAL0 + vocab.n_vals:
                print(f"  {k} → {vocab.val_index(tok)}")
            else:
                print(f"  {k} → (unknown, tok={tok})")
        elif c == "status":
            print("  " + format_integrity(state_integrity(states)))
        elif c == "reset":
            states = model.init_states(1, device)
            if os.path.exists(args.state):
                os.remove(args.state)
            print("  memory cleared (W0)")
        else:
            print("  unknown command — teach K V | ask K | status | reset | quit")


def cmd_status(args, device):
    if not os.path.exists(args.state):
        print(f"state {args.state} does not exist"); return
    states = st.load_state(args.state, device)
    print(format_integrity(state_integrity(states)))


def cmd_reset(args, device):
    if os.path.exists(args.state):
        os.remove(args.state)
    print(f"state {args.state} deleted (next start from W0)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--state", default="persist.pt")
    p.add_argument("--snap_dir", default="snapshots")
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("teach"); t.add_argument("--key", type=int, required=True)
    t.add_argument("--value", type=int, required=True)
    t.add_argument("--snapshot", action="store_true")

    a = sub.add_parser("ask"); a.add_argument("--key", type=int, required=True)
    a.add_argument("--no-state", dest="no_state", action="store_true")

    c = sub.add_parser("chat")
    c.add_argument("--no-state", dest="no_state", action="store_true")

    sub.add_parser("status")
    sub.add_parser("reset")

    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    {"teach": cmd_teach, "ask": cmd_ask, "chat": cmd_chat,
     "status": cmd_status, "reset": cmd_reset}[args.cmd](args, device)


if __name__ == "__main__":
    main()
