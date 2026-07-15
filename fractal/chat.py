"""Persistent chat in the terminal.

The model remembers across turns AND across process restarts: the fast weights W are saved to
disk after every turn and loaded at startup. No external growing table — the memory IS
inside the weights and self-modifies (constant memory footprint).

Run:  uv run python -m fractal.chat
      uv run python -m fractal.chat --fresh     # reset memory
"""

from __future__ import annotations

import argparse
import os

import torch

from fractal import persist, tokenizer as tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="fractal_ckpt.pt")
    ap.add_argument("--tokenizer", type=str, default="fractal_tokenizer.json")
    ap.add_argument("--state", type=str, default="fractal_chat_state.pt")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--fresh", action="store_true", help="start with a cleared memory")
    ap.add_argument("--role", action="store_true",
                    help="User:/Assistant: format (for a model fine-tuned on dialogues, 16k tok)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, dev)
    model.eval()
    tok = tk.load(args.tokenizer)

    if args.fresh or not os.path.exists(args.state):
        states = model.init_states(1, dev)
        print("(fresh memory)")
    else:
        states = persist.load_states(args.state, dev)
        print(f"(memory loaded from {args.state})")

    eot_turn = tok.token_to_id("<|eot|>")             # dialogue tokenizer has a turn-marker; otherwise None
    eot = tok.token_to_id("<|endoftext|>")
    suffix = " <|eot|>" if eot_turn is not None else ""
    print("Write messages (empty line or Ctrl-D quits). Memory survives restart.\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        if args.role:                                 # User:/Assistant: framing (fine-tuned model)
            ids = tok.encode(f"\nUser: {line}\nAssistant:").ids
        else:
            ids = tok.encode(" " + line + suffix).ids  # the message + end of turn → the model continues the reply
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        gen, states = model.generate_stream(
            idx, args.tokens, states, temperature=args.temperature, top_k=args.top_k)
        out = gen[0].tolist()
        for m in (eot_turn, eot):                     # trim to the end of the 1st reply (token)
            if m is not None and m in out:
                out = out[:out.index(m)]
        text = tok.decode(out).strip()
        if args.role:                                 # and to the start of the next reply (text)
            for stop in ("\nUser:", "User:", "\nAssistant:"):
                i = text.find(stop)
                if i != -1:
                    text = text[:i].strip()
        print("model>" + text + "\n")
        persist.save_states(args.state, states)      # persist after every turn


if __name__ == "__main__":
    main()
