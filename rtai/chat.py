"""Persistent chat / storyteller with the language model.

Unlike simple generation, here the model runs in STREAMING mode: the fast weights
`W` (and the convolution context) are **carried across turns and sessions** and saved to disk.
The model thus remembers even what is long past the window of the most recent tokens —
information survives in the weights (verified: the name "Rex" survived ~490 tokens, see README).
It is not perfect (the model is trained on short stories), but it is genuine persistent memory
in the weights.

    uv run python -m rtai.chat                 # continues from the saved memory (chat_state.pt)
    uv run python -m rtai.chat --fresh         # starts with an empty memory
"""

from __future__ import annotations

import argparse
import os

import torch

from .tokenizer import load as load_tok
from . import state as st


def _save(path, states, conv, device):
    st.atomic_torch_save({"W": [s.cpu() for s in states],
                          "conv": [c.cpu() if c is not None else None for c in conv]}, path)


def _load(path, device):
    o = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(o, dict) or not isinstance(o.get("W"), list) \
            or not isinstance(o.get("conv"), list):
        raise ValueError(f"invalid chat runtime-state schema: {path}")
    conv = [c.to(device) if c is not None else None for c in o["conv"]]
    return [w.to(device) for w in o["W"]], conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_lm.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--state", default="chat_state.pt")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--fresh", action="store_true", help="start with an empty memory")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = st.load_model(args.ckpt, device)
    tok = load_tok(args.tokenizer)

    if args.fresh or not os.path.exists(args.state):
        states, conv = model.init_states(1, device), model.init_conv_states()
        mem = "empty memory"
    else:
        states, conv = _load(args.state, device)
        mem = f"continuing from memory in {args.state}"

    print(f"RTAI storyteller (persistent) — {mem}.")
    print(f"(model {sum(p.numel() for p in model.parameters())/1e6:.1f}M, window "
          f"{model.cfg.block_size} tok; W is carried forward and saved). Empty line = end.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            break
        ids = tok.encode(" " + line).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        gen, states, conv = model.generate_stream(
            idx, args.tokens, states, conv, temperature=args.temperature, top_k=args.top_k)
        print("model>" + tok.decode(gen[0].tolist()))
        _save(args.state, states, conv, device)   # persist after every turn


if __name__ == "__main__":
    main()
