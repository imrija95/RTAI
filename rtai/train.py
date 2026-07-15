"""Meta-training: teach the model the ABILITY to usefully modify its own weights.

We train on an associative-recall task. During training we start from a learned W0
(the gradient flows through the recurrence of self-modifications → the slow parameters
and W0 are learned). At deploy time the model then uses this ability PERSISTENTLY
(see rtai/run.py).

Run:  uv run python -m rtai.train
"""

from __future__ import annotations

import argparse
import os

import torch

from .data_recall import RecallVocab, make_recall_batch, encode_teach, encode_query
from .model import GPTConfig, RTAIModel
from .monitor import canary_recall_acc
from . import state as st


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--n_queries", type=int, default=8)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--out", type=str, default="ckpt.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


@torch.no_grad()
def recall_accuracy(model, x, y):
    logits, _, _, _ = model(x, states=None)
    mask = y != -1
    pred = logits.argmax(dim=-1)
    return (pred[mask] == y[mask]).float().mean().item()


@torch.no_grad()
def persistence_check(model, vocab, device, tmp_state="._persist_test.pt", n=6):
    """In-process acid test: teach a few associations, save W to disk, reload, ask
    in an EMPTY context. Compare with the ablation without persistence (start from W0)."""
    model.eval()
    keys = torch.randperm(vocab.n_keys)[:n].tolist()
    vals = torch.randint(vocab.n_vals, (n,)).tolist()
    pairs = list(zip(keys, vals))

    # 1) TEACH: run the teach sequence, the state W self-modifies
    states = model.init_states(1, device)
    teach = encode_teach(pairs, vocab, device)
    _, _, states, _ = model(teach, states=states)
    st.save_state(tmp_state, states)                 # save to disk (simulate a restart)

    # 2) NEW SESSION: load W from disk, empty context, queries
    loaded = st.load_state(tmp_state, device)
    ok_persist = 0
    ok_ablation = 0
    for ki, vi in pairs:
        q = encode_query(ki, vocab, device)
        # with persistent W
        logits, _, _, _ = model(q, states=[s.clone() for s in loaded])
        if logits[0, -1].argmax().item() == vocab.val_tok(vi):
            ok_persist += 1
        # ablation: without persistence (start from W0) — it must not know
        logits_a, _, _, _ = model(q, states=None)
        if logits_a[0, -1].argmax().item() == vocab.val_tok(vi):
            ok_ablation += 1
    if os.path.exists(tmp_state):
        os.remove(tmp_state)
    return ok_persist / n, ok_ablation / n


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    vocab = RecallVocab()
    block = 2 * args.n_pairs + 3 * args.n_queries
    cfg = GPTConfig(vocab_size=vocab.size, block_size=block,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = RTAIModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    for it in range(1, args.iters + 1):
        model.train()
        x, y = make_recall_batch(args.batch, args.n_pairs, args.n_queries, vocab, device)
        _, loss, _, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if it % args.eval_every == 0 or it == 1:
            xe, ye = make_recall_batch(256, args.n_pairs, args.n_queries, vocab, device)
            acc = recall_accuracy(model, xe, ye)
            canary = canary_recall_acc(model, vocab, device,
                                       args.n_pairs, args.n_queries)
            print(f"iter {it:5d} | loss {loss.item():.4f} | recall {acc:.3f} "
                  f"| canary {canary:.3f}")

    st.save_model(args.out, model)
    print(f"[train] saved to {args.out}")

    p_acc, a_acc = persistence_check(model, vocab, device)
    print("\n=== PERSISTENCE ACID TEST ===")
    print(f"recall with persistent W (new session, empty context): {p_acc:.2f}")
    print(f"recall without persistence (ablation, start from W0):  {a_acc:.2f}")
    print("→ Persistence works when the first is clearly > the second.")


if __name__ == "__main__":
    main()
