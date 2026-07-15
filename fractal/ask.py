"""Pure recall REPL — shows what the model CAN do: single-fact recall from memory.

  teach <sentence>   store a fact into persistent memory (write only, no generation)
  ask <query>        recall the answer (reads memory on a CLONE → the query does not dirty memory)
  reset              clear memory
  empty line         quit

Difference vs `chat`: `chat` generates text after every turn, which also gets written into
memory and overwrites facts. `ask` separates writing (teach) from reading (ask) → clean recall.
"""

from __future__ import annotations

import argparse

import torch

from fractal import persist, tokenizer as tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fractal_ckpt.pt")
    ap.add_argument("--tokenizer", default="fractal_tokenizer.json")
    ap.add_argument("--tokens", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, dev)
    model.eval()
    tok = tk.load(args.tokenizer)
    states = model.init_states(1, dev)

    print("teach <sentence>  |  ask <query>  |  reset  |  empty line = quit\n")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line == "reset":
            states = model.init_states(1, dev)
            print("(memory cleared)")
            continue
        cmd, _, text = line.partition(" ")
        ids = torch.tensor([tok.encode(" " + text).ids], device=dev)
        if cmd == "teach":
            with torch.no_grad():
                _, states = model.forward_stream(ids, states)   # write to memory only
            print("(stored)")
        elif cmd == "ask":
            probe = [s.clone() for s in states]                  # read on a clone → don't dirty memory
            gen, _ = model.generate_stream(ids, args.tokens, probe, temperature=0.1, top_k=1)
            print("→", tok.decode(gen[0].tolist()).strip())
        else:
            print("use 'teach <sentence>' or 'ask <query>'")


if __name__ == "__main__":
    main()
