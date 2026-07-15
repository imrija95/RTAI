"""Proof of persistence across a PROCESS RESTART (run twice, sharing files on disk).

  write: accumulate memory (stream a few tokens), save state to disk, print top-level ‖W‖.
  read : in a NEW process load the state, print ‖W‖; also a fresh state has ‖W‖=0 (W0=zeros).

If read ‖W‖ == write ‖W‖ > 0, the memory really survived the restart and lives inside the weights,
not in an external table. Called from run_persist.sh (two separate interpreter invocations).
"""

from __future__ import annotations

import sys
import torch

from fractal import persist

CKPT = "fractal_ckpt.pt"
STATE = "fractal_roundtrip_state.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _top_norm(states):
    # ‖W‖ of the topmost (permanent, γ=1) level of the first layer
    return states[0].W[-1].norm().item()


def write():
    torch.manual_seed(1)
    model = persist.load_model(CKPT, DEV).eval()
    states = model.init_states(1, DEV)
    print(f"  fresh memory: ‖W_top‖ = {_top_norm(states):.4f}")
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8), device=DEV)
    with torch.no_grad():
        _, states = model.generate_stream(ids, 4, states, temperature=0.8, top_k=10)
    persist.save_states(STATE, states)
    print(f"  after accumulation: ‖W_top‖ = {_top_norm(states):.4f}  → saved to {STATE}")


def read():
    model = persist.load_model(CKPT, DEV).eval()
    fresh = model.init_states(1, DEV)
    loaded = persist.load_states(STATE, DEV)
    nf, nl = _top_norm(fresh), _top_norm(loaded)
    print(f"  fresh process: fresh ‖W_top‖ = {nf:.4f}   loaded ‖W_top‖ = {nl:.4f}")
    ok = nf < 1e-6 and nl > 1e-3
    print("  OK — memory survived the restart and lives inside the weights" if ok
          else "  FAIL — state did not carry over")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    {"write": write, "read": read}[sys.argv[1]]()
