"""Experimental inference-time DIFFUSION-style decoder for the fractal LM.

This is an *inference-only* generation mode — it does not retrain and does not touch the model's
identity (VIBE #4: the fast-weight associative memory `W` is preserved; here it is simply read as
the model runs). Instead of emitting one token at a time left-to-right, it holds a whole-canvas
latent of answer positions, starts them as noise (random tokens), and iteratively RELAXES them
coarse->fine until they "round off" to an answer — the terminal shows the characters churn and
settle live.

Why this is genuinely a diffusion-family sampler (not a gimmick)
---------------------------------------------------------------
The tied head is `logits = (tok_emb) · ln_f(x)`, i.e. an inner product of the relaxed state with
the STORED token embeddings — literally one associative read (the same shape as `v̂ = W q`). One
refinement step resamples the uncommitted positions from `softmax(logits / T)` — an annealed
(Langevin-style) step on the energy the model defines over sequences. The schedule is coarse->fine
on two coupled knobs, mirroring a diffusion noise schedule and the model's own γ ladder:
  * temperature T: high -> low  (global structure first, detail last),
  * frozen fraction: 0 -> 1     (noise level high -> 0; committed positions stop moving).
This is the MaskGIT / discrete-diffusion (LLaDA-style) confidence decoding schedule.

Honest limitations (VIBE #7, #9)
--------------------------------
  * The base model is CAUSAL and AR-trained, so a canvas position's logit is conditioned on the
    prompt and EARLIER canvas positions only (not both sides). The relaxation therefore biases
    toward settling earlier positions first; it is a faithful *approximation* of whole-canvas
    denoising on an AR model, not a natively bidirectional denoiser.
  * At 126M ("caveman" scale) the settled text is not fluent. This mode demonstrates the MECHANISM
    and the live coarse->fine settling — competence is a matter of scale, not of this loop.

The deeper, natively-fractal variant (relaxing token EMBEDDINGS with the score `W·k(x) − v(x)` over
all scales) is described in chat/docs but NOT run here: the current weights were never trained to
denoise embeddings, so it decodes to noise. Kept as a note, not a claim.

CLI:  uv run python -m fractal.diffuse --ckpt fractal_ckpt_chat_phase2b.pt \
          --tokenizer fractal_tokenizer_32k.json
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F

from fractal import persist, tokenizer as tk
from fractal import chat_format as cf


def _schedule(step: int, steps: int, t0: float, t1: float):
    """Coarse->fine schedule for one refinement step.

    Returns (temperature, frozen_fraction). Temperature is geometric t0 -> t1; the frozen fraction
    follows a cosine ramp 0 -> 1 so the canvas commits gently at first (global structure) and
    locks in the tail quickly (detail) — the diffusion noise level going high -> 0."""
    a = step / max(steps - 1, 1)
    temp = t0 * (t1 / t0) ** a
    frozen_frac = 0.5 - 0.5 * math.cos(math.pi * a)     # cosine 0 -> 1
    return temp, frozen_frac


@torch.no_grad()
def diffuse_decode(model, prompt_ids, n_pos: int, steps: int, device,
                   t0: float = 1.6, t1: float = 0.08, depth=None):
    """Relax a canvas of `n_pos` answer positions from noise to an answer, coarse->fine.

    Yields, once per step, a dict {step, temp, frozen_frac, n_frozen, canvas, frozen} so a caller
    can render the live settling. `canvas` is a python list of token ids (len n_pos); `frozen` is a
    bool list marking committed positions. The prompt is fixed context; only the canvas moves."""
    P = prompt_ids.shape[1]
    vocab = model.cfg.vocab_size
    # noise init: uniform random tokens (the "pure noise" the diffusion starts from)
    canvas = torch.randint(0, vocab, (1, n_pos), device=device)
    frozen = torch.zeros(n_pos, dtype=torch.bool, device=device)

    for step in range(steps):
        temp, frozen_frac = _schedule(step, steps, t0, t1)
        full = torch.cat([prompt_ids, canvas], dim=1)              # (1, P + n_pos)
        logits, _, _, _ = model(full, depth=depth)                 # fresh W from W0 (whole-canvas read)
        # logits[:, j-1] predicts absolute position j; canvas pos i is absolute P+i
        pos_logits = logits[0, P - 1 : P - 1 + n_pos, :]           # (n_pos, vocab)
        probs = F.softmax(pos_logits / max(temp, 1e-4), dim=-1)    # annealed read

        # resample every not-yet-frozen position (a Langevin step at temperature T)
        sampled = torch.multinomial(probs, 1).squeeze(-1)          # (n_pos,)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # confidence of the draw
        upd = ~frozen
        canvas[0, upd] = sampled[upd]

        # commit (freeze) the highest-confidence uncommitted positions up to the target fraction
        target_frozen = int(math.ceil(frozen_frac * n_pos))
        n_new = max(0, target_frozen - int(frozen.sum()))
        if n_new > 0:
            cand = conf.clone()
            cand[frozen] = -1.0                                    # only pick from uncommitted
            pick = torch.topk(cand, min(n_new, int((~frozen).sum()))).indices
            # commit each picked position to its most likely token (argmax = "rounded off")
            canvas[0, pick] = pos_logits[pick].argmax(dim=-1)
            frozen[pick] = True

        yield {"step": step, "temp": temp, "frozen_frac": frozen_frac,
               "n_frozen": int(frozen.sum()), "canvas": canvas[0].tolist(),
               "frozen": frozen.tolist()}
        if bool(frozen.all()):
            break

    # final: commit anything still loose to its argmax
    if not bool(frozen.all()):
        full = torch.cat([prompt_ids, canvas], dim=1)
        logits, _, _, _ = model(full, depth=depth)
        pos_logits = logits[0, P - 1 : P - 1 + n_pos, :]
        loose = ~frozen
        canvas[0, loose] = pos_logits[loose].argmax(dim=-1)
        yield {"step": steps, "temp": t1, "frozen_frac": 1.0, "n_frozen": n_pos,
               "canvas": canvas[0].tolist(), "frozen": [True] * n_pos}


DIM, GREEN, RESET, CLR = "\x1b[2m", "\x1b[32m", "\x1b[0m", "\x1b[2K\r"


def _render(tok, frame, delay: float):
    """Redraw the canvas in place: dim while settling, green once fully frozen."""
    text = tok.decode(frame["canvas"]).replace("\n", "⏎")
    done = frame["n_frozen"] == len(frame["canvas"])
    color = GREEN if done else DIM
    bar = f"[{frame['step']:>2}] T={frame['temp']:.2f} frozen {frame['n_frozen']:>3}/{len(frame['canvas'])} "
    sys.stdout.write(CLR + bar + color + text + RESET)
    sys.stdout.flush()
    time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description="Diffusion-style live decoder (experimental).")
    ap.add_argument("--ckpt", default="fractal_ckpt_chat_phase2b.pt")
    ap.add_argument("--tokenizer", default="fractal_tokenizer_32k.json")
    ap.add_argument("--pos", type=int, default=48, help="canvas length (answer positions)")
    ap.add_argument("--steps", type=int, default=24, help="refinement steps (coarse->fine)")
    ap.add_argument("--t0", type=float, default=1.6, help="start temperature (coarse)")
    ap.add_argument("--t1", type=float, default=0.08, help="end temperature (fine)")
    ap.add_argument("--delay", type=float, default=0.12, help="seconds between rendered steps")
    ap.add_argument("--depth", type=int, default=None, help="unroll depth (default = model cfg)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, dev)
    model.eval()
    tok = tk.load(args.tokenizer)

    print(f"diffusion decode | {args.pos} positions × {args.steps} steps | device={dev}")
    print("note: 126M 'caveman' scale — this shows the MECHANISM (live coarse→fine settling),\n"
          "      not fluent text. Empty line = quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except EOFError:
            break
        if not q:
            break
        context = f"{cf.USER}\n{q}\n{cf.ASSISTANT}\n"
        prompt_ids = torch.tensor([tok.encode(context).ids], device=dev)
        for frame in diffuse_decode(model, prompt_ids, args.pos, args.steps, dev,
                                    t0=args.t0, t1=args.t1, depth=args.depth):
            _render(tok, frame, args.delay)
        print()   # newline after the settled answer


if __name__ == "__main__":
    main()
