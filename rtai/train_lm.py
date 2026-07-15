"""Training the language model with long-range recall (#15).

Mixes synthetic recall episodes (fact → long filler → query) with ordinary TinyStories windows.
Because the model has no global attention, information reaches the query only through the fast
weights W → W learns to be a long-term memory. Deploy then streams (rtai/chat.py) and memory persists.

    uv run python -m rtai.train_lm --init_from ckpt_lm.pt --recall_ratio 0.4
"""

from __future__ import annotations

import argparse
import math

import torch

from .data_text import prepare
from .data_recall_lm import RecallGen
from .model import GPTConfig, RTAIModel
from . import state as st


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--recall_ratio", type=float, default=0.4)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=400)
    p.add_argument("--train_mb", type=int, default=400)
    p.add_argument("--init_from", type=str, default=None, help="warm-start from a checkpoint")
    p.add_argument("--out", type=str, default="ckpt_lm.pt")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def lr_at(it, a):
    if it < a.warmup:
        return a.lr * it / max(1, a.warmup)
    t = (it - a.warmup) / max(1, a.iters - a.warmup)
    return 0.1 * a.lr + 0.5 * (0.9 * a.lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def sample_text(model, tok, device, prompt="Once upon a time", n=100):
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate_lm(idx, n, temperature=0.8, top_k=40)
    return tok.decode(out[0].tolist())


def build_model(args, vocab, device):
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        c = ckpt["cfg"]
        cfg = GPTConfig(vocab_size=c["vocab_size"], block_size=args.block_size,
                        n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],
                        srwm_mode="chunk", chunk_size=args.chunk_size)
        model = RTAIModel(cfg).to(device)
        model.load_state_dict(ckpt["model"])       # weights are independent of block_size (no positional embeddings)
        print(f"[train_lm] warm-start from {args.init_from}")
    else:
        cfg = GPTConfig(vocab_size=vocab, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                        srwm_mode="chunk", chunk_size=args.chunk_size)
        model = RTAIModel(cfg).to(device)
    return model


def main():
    args = get_args()
    torch.manual_seed(1337)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_lm] device={device} | block={args.block_size} | recall_ratio={args.recall_ratio}")

    tok = prepare(train_mb=args.train_mb)
    gen = RecallGen(tok, split="train")
    gen_val = RecallGen(tok, split="val")

    model = build_model(args, tok.get_vocab_size(), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    for it in range(1, args.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args)
        model.train()
        x, y = gen.batch(args.block_size, args.batch, args.recall_ratio, device)
        _, loss, _, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if it % args.eval_every == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                xv, yv = gen_val.batch(args.block_size, args.batch, 0.0, device)  # pure fluency
                _, vloss, _, _ = model(xv, targets=yv)
            racc = gen.recall_accuracy(model, args.block_size, device, trials=48)
            print(f"iter {it:5d} | train {loss.item():.3f} | val(story) {vloss.item():.3f} "
                  f"| recall {racc:.3f} | lr {opt.param_groups[0]['lr']:.1e}")
            print("   sample: " + sample_text(model, tok, device).replace('\n', ' ')[:180])
            st.save_model(args.out, model)

    st.save_model(args.out, model)
    print(f"[train_lm] saved to {args.out}")
    print(f"final recall: {gen.recall_accuracy(model, args.block_size, device, trials=128):.3f}")
    print("\nSample:\n" + sample_text(model, tok, device, n=160))


if __name__ == "__main__":
    main()
