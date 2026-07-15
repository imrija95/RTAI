"""Streaming acid test of language-model persistence.

For each distance D: teach a fact → stream D filler tokens (streamed, W carries over) →
ask → check whether the value was recalled. Compares the model after #15 (recall training)
against the base one (TinyStories only). D above ~512 tests generalization BEYOND the trained distance.

    uv run python tests/exp_persist_lm.py
"""
import numpy as np
import torch
from rtai import state as st
from rtai.tokenizer import load as load_tok
from rtai.data_recall_lm import NAMES, ATTRS, ATTR_KEYS

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = load_tok("tokenizer.json")
data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")


def filler_ids(n):
    i = int(torch.randint(len(data) - n - 1, (1,)).item())
    return [int(t) for t in data[i:i + n]]


@torch.no_grad()
def stream_recall(model, D, trials=40):
    ok = 0
    for _ in range(trials):
        nm = NAMES[torch.randint(len(NAMES), (1,)).item()]
        attr = ATTR_KEYS[torch.randint(len(ATTR_KEYS), (1,)).item()]
        val = ATTRS[attr][torch.randint(len(ATTRS[attr]), (1,)).item()]
        W, C = model.init_states(1, dev), model.init_conv_states()
        fact = torch.tensor([tok.encode(f"{nm}'s favorite {attr} is {val}. ").ids], device=dev)
        _, W, C = model.forward_stream(fact, W, C)
        fill = torch.tensor([filler_ids(D)], device=dev)
        _, W, C = model.forward_stream(fill, W, C)
        q = torch.tensor([tok.encode(f"{nm}'s favorite {attr} is").ids], device=dev)
        gen, _, _ = model.generate_stream(q, 4, W, C, temperature=0.1, top_k=1)
        if val in tok.decode(gen[0].tolist()).lower():
            ok += 1
    return ok / trials


for label, ck in [("after #15 (ckpt_lm.pt)", "ckpt_lm.pt"), ("base (TinyStories only)", "ckpt_lm_base.pt")]:
    m = st.load_model(ck, dev)
    print(f"\n== {label} ==  (training window ~512 tok)")
    for D in [128, 400, 800]:
        print(f"  distance {D:4d} tok: recall = {stream_recall(m, D):.2f}")
