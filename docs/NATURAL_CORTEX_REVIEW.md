# Natural Cortex preflight review

Date: 2026-07-16

This records the code-review and bounded benchmark gate performed before the Natural Cortex GPU
run. Authorization and execution happened later; the post-gate measurements are summarized below
and recorded in `results/natural-cortex-25m.json`.

## Keep

- The tied recurrent block and shared depth rule.
- The two-scale fast-weight ladder with a permanent scale.
- The dense chunk kernel and existing recurrent/chunk numerical equivalence tests.
- Usage-driven plasticity.
- BF16, TF32, fused AdamW, gradient clipping, and atomic tensor-only persistence.
- The unified chat/tool renderer and persistent fast-weight agent loop.
- Throttled dashboard telemetry derived from the real forward/backward pass.
- Append-only low-rank experts and explicit lifecycle states from Growing Cortex.

## Change

- Train a fresh 24k tokenizer with atomic chat, tool, skill, teaching, end, and document tokens.
- Replace first-stream validation with deterministic per-source document-hash splitting.
- Add fixed trainer and validation seeds.
- Sample only windows containing a nonzero target loss.
- Store sharded tokens, masks, source IDs, and document boundaries with a checked manifest.
- Pin every dataset revision and record license, attribution, checksums, and token counts.
- Use compiler-free local expert teaching for production.
- Birth local experts with random `down` and zero `up`, giving zero output and nonzero gradient.
- Store skill tensors, routing threshold, address configuration, rollback revisions, and audit events
  outside the immutable base checkpoint.
- Require explicit confirmation for the first skill activation in a trajectory.
- Select dense versus MoE only through the predeclared 10M-token gate.

## Remove from the production preset

- Timescale neurogenesis.
- Selective recurrent dynamics.
- Event patches.
- Event Algebra and event eligibility traces.
- Global online `W0` teaching.
- Scalar-feedback consolidation.
- Untied depth.
- The full `E² × rank` skill compiler.
- Silent automatic skill routing.

The archived experiment runners retain these mechanisms for reproducibility. They are not reachable
from `fractal.natural_train`.

## Mandatory issue resolution

| Issue | Resolution |
| --- | --- |
| Broken `MIX` import | `train_tokenizer.py` now selects `RECIPES` or the pinned Natural Cortex builder. |
| Missing seed | Legacy and Natural trainers seed Python, NumPy, PyTorch, and CUDA. |
| Non-reproducible validation | Validation uses a fixed seed and stable hash-split documents. |
| First-token-stream validation | Replaced in the Natural Cortex pipeline by per-source document hashing. |
| Multi-token protocol markers | Required markers are tokenizer special tokens and tested as one ID each. |
| `E² × rank` compiler scaling | Production config sets `skill_compiler="none"`; address encoding is factorized. |
| Runtime threshold/address persistence | Stored and validated in the separate skill-bank manifest. |
| Zero-mask chat windows | Samplers reject windows with no loss-bearing target. |
| Refuted flags mixed into training | A dedicated production trainer exposes none of those flags. |
| Zero-output expert had zero gradient | Candidate `down` is random and `up` is zero; the `up` gradient is nonzero. |

## Bounded benchmark

Machine-readable results:
[`results/natural-cortex-preflight.json`](results/natural-cortex-preflight.json).

The production-shape dense model was measured on an RTX 3060 12GB with batch 8 and sequence length
512:

| Mode | Throughput | Peak allocated VRAM |
| --- | ---: | ---: |
| No gradient checkpointing | 5,081 tok/s | 8.97 GB |
| Gradient checkpointing | 3,972 tok/s | 2.30 GB |

Without checkpointing was about 28% faster and remained below the 10.5GB ceiling. The production
preset therefore disables checkpointing. Sampled telemetry amortized over 25 steps measured about
0.05% of step time, below the 2% gate.

The same bounded run verified:

- dense stored parameters: 66,010,708;
- dense active parameters: 65,888,340;
- candidate birth changes logits by exactly zero;
- candidate `down` starts nonzero and `up` starts zero;
- the zero-output candidate has a nonzero gradient;
- 16 local expert steps produce a nonzero update;
- the full hypernetwork compiler is absent;
- every refuted production flag is off.

At the measured 5,081 tok/s, 240M tokens projected to about 13.1 GPU hours. This passed the target
GPU time gate without relying on an extrapolation from different hardware.

## Post-gate execution update

The immutable data build produced 240,001,141 training tokens and 2,406,475 validation tokens.
The 10M-token-per-arm A/B then produced:

| Arm | Throughput | Validation loss | Recall |
| --- | ---: | ---: | ---: |
| Dense | 4,957 tok/s | 5.0678 | 0.0% |
| Top-1 MoE | 4,481 tok/s | 5.0412 | 0.0% |

MoE retained 90.4% of dense throughput, every expert received at least 17.7% of traffic, and maximum
pairwise source-routing total variation was 22.4%. Every predeclared gate passed, so the MoE
checkpoint continued mechanically.

The main run reached a validation loss of 4.6318 at 20,000,768 total tokens and was paused at the
first durable boundary, 25,001,984 tokens. The pause was requested explicitly rather than triggered
by loss, time, or acceptance logic. The checkpoint and optimizer/plasticity run state were written
before termination and remain resumable. The base chat gate and local-teaching acceptance gates
remain unrun.

## Verification performed

- Full repository test suite: 51 passed.
- Focused Natural/Growing Cortex and feedback tests: 21 passed.
- Ruff on all changed Python modules: passed.
- Shell syntax and executable launchers: passed.
- Git whitespace check: passed.
- Production-shape GPU preflight: passed, with the time-budget warning above.

The RTX 3060 preflight and 20M-token A/B are complete. Further training remains subject to the
existing 24-hour GPU budget and acceptance gates.

## Exact implementation surface

New production modules:

- `fractal/natural_data.py`;
- `fractal/natural_train.py`;
- `fractal/natural_ab.py`;
- `fractal/natural_eval.py`;
- `fractal/natural_runtime.py`;
- `fractal/natural_preflight.py`;
- `scripts/run-natural-cortex.sh`.

Operational single-GPU launchers:

- `scripts/run-natural-cortex-gpu.sh`;
- `scripts/start-natural-cortex-gpu.sh`.

Load-bearing changes:

- `fractal/tokenizer.py`;
- `fractal/chat_format.py`;
- `fractal/data.py`;
- `fractal/train_tokenizer.py`;
- `fractal/train.py`;
- `fractal/model.py`;
- `fractal/growing_cortex.py`;
- `fractal/persist.py`;
- `fractal/agent.py`;
- `fractal/viz_serve.py`;
- `fractal/web/fractal3d.html`.

Tests:

- `fractal/tests/test_natural_cortex.py`;
- compatibility updates in `fractal/tests/test_growing_cortex.py`.

No change was made to `VIBE.md`. No commit or push was performed.
