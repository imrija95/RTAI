"""live — chat that LEARNS ONLINE (no separate train/serve phase).

Two timescales of plasticity (the fractal framework):
  - FAST weights (delta rule, gradient-free) → short-term memory within the conversation,
  - SLOW weights (online gradient + replay) → permanently learn from what is TYPED; replay
    (rehearsal of general data) keeps forgetting in check.

Learns ONLY from the typed text, not from its own replies → no self-contamination of memory.
Learned weights are saved → after a session the model really knows more (learning = serving).

Run:  uv run python -m fractal.live
      uv run python -m fractal.live --fresh     # from scratch, discard what was learned
"""

from __future__ import annotations

import argparse
import os

import torch

from fractal import persist, tokenizer as tk
from fractal.data import get_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fractal_ckpt.pt")            # base (loaded once)
    ap.add_argument("--live_ckpt", default="fractal_ckpt_live.pt")  # LEARNED weights saved here
    ap.add_argument("--tokenizer", default="fractal_tokenizer.json")
    ap.add_argument("--state", default="fractal_live_state.pt")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)                # online steps per turn
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--no_learn", action="store_true", help="reply only, do not learn (ablation)")
    ap.add_argument("--fresh", action="store_true", help="from scratch, discard learned weights and state")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    src = args.live_ckpt if (os.path.exists(args.live_ckpt) and not args.fresh) else args.ckpt
    model = persist.load_model(src, dev)
    tok = tk.load(args.tokenizer)
    eot = tok.token_to_id("<|eot|>")                  # dialogue turn-marker (otherwise None)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.fresh or not os.path.exists(args.state):
        states = model.init_states(1, dev)
    else:
        states = persist.load_states(args.state, dev)

    print(f"live (learns online) — base: {src}"
          + ("  [learning OFF]" if args.no_learn else f"  [lr={args.lr}, {args.steps} steps/turn + replay]"))
    print("write messages; the model replies and learns from them. empty line = quit.\n")

    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        ids = tok.encode(" " + line).ids                          # raw text (for learning)
        idx = torch.tensor([ids], dtype=torch.long, device=dev)

        # 1) REPLY — fast weights, streaming, gradient-free (short-term memory carries)
        model.eval()
        gids = tok.encode(" " + line + (" <|eot|>" if eot is not None else "")).ids
        gen, states = model.generate_stream(torch.tensor([gids], device=dev), args.tokens, states,
                                            temperature=args.temperature, top_k=args.top_k)
        out = gen[0].tolist()
        if eot is not None and eot in out:
            out = out[:out.index(eot)]
        print("model>" + tok.decode(out).strip())

        # 2) LEARNING — slow weights, online gradient from the typed text + replay (rehearsal)
        if not args.no_learn and len(ids) >= 2:
            model.train()
            xu, yu = idx[:, :-1], idx[:, 1:]
            with torch.no_grad():
                _, before, _, _ = model(xu, targets=yu, states=None)
            for _ in range(args.steps):
                _, lu, _, _ = model(xu, targets=yu, states=None)          # the typed sentence
                xb, yb = get_batch("val", 8, 64, dev)
                _, lb, _, _ = model(xb, targets=yb, states=None)          # replay: general data
                (lu + lb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, after, _, _ = model(xu, targets=yu, states=None)
            print(f"   [learned online: loss of the sentence {before.item():.2f} → {after.item():.2f}]")
            persist.save_model(args.live_ckpt, model)                     # learned weights survive the session

        persist.save_states(args.state, states)


if __name__ == "__main__":
    main()
