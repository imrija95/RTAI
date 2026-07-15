"""Export the real trajectory of the self-modifying matrix W to JSON for visualization.

Learn a few associations + queries, and record how W (the fast weights) changes step by
step. Faithfully: for each prefix of length L we run model(x[:, :L]) from a clean start and
take the resulting W — because both the recurrence and the convolution are causal, W after a
prefix of length L is exactly the W the model has at step L in the full sequence.

    uv run python -m rtai.export_viz --ckpt ckpt.pt --out viz_data.json
"""

from __future__ import annotations

import argparse
import json

import torch

from .data_recall import RecallVocab
from . import state as st


def build_sequence(vocab: RecallVocab, pairs, query_keys):
    toks, labels = [], []
    for ki, vi in pairs:
        toks.append(vocab.key_tok(ki)); labels.append({"t": f"K{ki}", "kind": "key"})
        toks.append(vocab.val_tok(vi)); labels.append({"t": f"V{vi}", "kind": "val"})
    query_pos = []
    for ki in query_keys:
        toks.append(vocab.QUERY); labels.append({"t": "?", "kind": "query"})
        query_pos.append(len(toks))
        toks.append(vocab.key_tok(ki)); labels.append({"t": f"K{ki}", "kind": "qkey"})
    return toks, labels, query_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--out", default="viz_data.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = st.load_model(args.ckpt, device)
    model.eval()
    vocab = RecallVocab()
    cfg = model.cfg

    pairs = [(3, 7), (12, 5), (20, 19), (8, 2)]
    truth = dict(pairs)
    query_keys = [3, 12, 20, 8]
    toks, labels, query_pos = build_sequence(vocab, pairs, query_keys)
    x = torch.tensor([toks], dtype=torch.long, device=device)
    T = len(toks)

    def round3(m):
        return [[round(float(v), 3) for v in row] for row in m]

    steps = []
    prev_W = None
    queries = []
    with torch.no_grad():
        for L in range(1, T + 1):
            init = model.init_states(1, device)
            logits, _, states, _ = model(x[:, :L], states=init)
            # W per layer/head after step L
            W_layers = []
            delta = []
            for li, W in enumerate(states):            # W: (1,H,hd,hd)
                Wh = W[0]                              # (H,hd,hd)
                W_layers.append([round3(Wh[h].cpu().tolist()) for h in range(cfg.n_head)])
                if prev_W is not None:
                    delta.append(round(float((Wh - prev_W[li]).norm().item()), 4))
                else:
                    delta.append(round(float(Wh.norm().item()), 4))
            prev_W = [W[0] for W in states]
            steps.append({"W": W_layers, "delta": delta})

            # retrieval at the queried key's position
            if L in query_pos:
                qi = query_pos.index(L)
                key = query_keys[qi]
                tok = int(logits[0, -1].argmax().item())
                pred = vocab.val_index(tok) if vocab.VAL0 <= tok < vocab.VAL0 + vocab.n_vals else None
                queries.append({"step": L - 1, "key": key, "pred": pred,
                                "truth": truth.get(key), "ok": pred == truth.get(key)})

    data = {
        "config": {"n_layer": cfg.n_layer, "n_head": cfg.n_head, "hd": cfg.head_dim},
        "tokens": labels,
        "steps": steps,
        "queries": queries,
        "pairs": [{"k": k, "v": v} for k, v in pairs],
    }
    with open(args.out, "w") as f:
        json.dump(data, f)
    n_ok = sum(q["ok"] for q in queries)
    print(f"[export] {args.out}: T={T} steps, {cfg.n_layer}×{cfg.n_head} matrices {cfg.head_dim}×{cfg.head_dim}")
    print(f"[export] recall on the sample: {n_ok}/{len(queries)}  {queries}")


if __name__ == "__main__":
    main()
