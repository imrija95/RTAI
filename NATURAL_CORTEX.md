# Natural Cortex

Natural Cortex is the first from-scratch RTAI preset aimed at simple English conversation and
confirmed local skill teaching. The target is a useful "smart caveman," not a polished general
assistant. This document is an implementation contract and falsification plan; it does not modify
the invariants in `VIBE.md`.

## Current status

The implementation review, tokenizer, and 240M-token corpus build are complete. The deterministic
10M-token-per-arm screen selected the four-expert top-1 MoE, passing every declared promotion gate.
The selected run was then paused at a durable 25,001,984-token checkpoint by operator request. Its
optimizer and plasticity state are resumable; the 200M chat gate and final acceptance suite have not
run, and no model artifact is approved for release.

The pre-run review is in
[`docs/NATURAL_CORTEX_REVIEW.md`](docs/NATURAL_CORTEX_REVIEW.md). Sanitized measurements through the
pause are in
[`docs/results/natural-cortex-25m.json`](docs/results/natural-cortex-25m.json).

## Production model

The shared configuration is:

- vocabulary: 24,000 byte-level BPE tokens plus atomic protocol tokens;
- embedding width: 1,792;
- heads: 28 with head dimension 64;
- one weight-tied block unrolled to depth 8;
- two fast-weight scales: `tau0=16` and one permanent scale;
- shared MLP ratio 2;
- dense fast-weight path with chunk size 64;
- usage-driven plasticity;
- BF16, TF32, and fused AdamW on CUDA;
- compiler-free rank-8 skill hemispheres;
- factorized 64-dimensional skill address encoder;
- explicit skill activation; silent automatic routing is disabled.

Measured parameter counts:

| Variant | Stored parameters | Active parameters |
| --- | ---: | ---: |
| Dense | 66,010,708 | 65,888,340 |
| Top-1 MoE, four experts | 104,553,048 | 65,895,512 |

One rank-8 skill stores 28,672 trainable down/up parameters. The production skill cortex contains no
`E² × rank` hypernetwork compiler. The archived synthetic Growing Cortex experiment can still load
its old full compiler through an explicit compatibility mode.

The production preset cannot enable timescale neurogenesis, selective recurrence, event patches,
Event Algebra, global online `W0` updates, scalar-feedback consolidation, untied blocks, or silent
skill routing.

## Tokenizer and data

The atomic tokens are:

```text
<|system|> <|user|> <|assistant|> <|tool_call|> <|tool_result|>
<|skill|> <|teach|> <|end|> <|endoftext|>
```

The tokenizer and corpus builders refuse to overwrite existing outputs.

The 240M-token training mix is continuous rather than staged:

| Share | Source | Immutable revision | License recorded in manifest |
| ---: | --- | --- | --- |
| 40% | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-By-1.0 |
| 30% | `HuggingFaceFW/fineweb-edu`, score 4–5 only | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By-1.0; Common Crawl terms also apply |
| 15% | `wikimedia/wikipedia`, `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC-BY-SA-3.0 and GFDL |
| 15% | `HuggingFaceTB/smol-smoltalk` | `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` | Apache-2.0 |

TinyStories remains a smoke/probe dataset and is not part of this mix.

The data pipeline:

- normalizes Unicode and whitespace;
- exact-deduplicates normalized documents across all sources with SHA-256;
- assigns validation independently per source using a stable document hash;
- renders Wikipedia as title plus cleaned article;
- applies full LM loss to raw text;
- applies loss only to assistant/tool-call output and `<|end|>` in chat records;
- writes aligned sharded token, mask, source-ID, and document-boundary files;
- records source revisions, licenses, attribution, checksums, and token counts in `manifest.json`;
- rejects zero-loss sampled windows;
- reproduces validation windows from a fixed seed.

Commands:

```bash
uv run python -m fractal.natural_data tokenizer \
  --out natural_tokenizer_24k.json

uv run python -m fractal.natural_data build \
  --tokenizer natural_tokenizer_24k.json \
  --out-dir natural_data_240m \
  --train-tokens 240000000

uv run python -m fractal.natural_data verify \
  --tokenizer natural_tokenizer_24k.json \
  --data-dir natural_data_240m
```

## Dense/MoE decision

Both arms process 10M tokens from the same deterministic data order. The dense stem is copied into
all four MoE experts before either arm trains. MoE is selected only if every gate passes:

- throughput is at least 85% of dense;
- validation loss is no more than 2% worse;
- every expert receives at least 10% of traffic;
- maximum pairwise source-routing total variation is at least 5%;
- held-out fast-weight recall is no more than two percentage points below dense.

The decision is mechanical:

```bash
uv run python -m fractal.natural_ab \
  --tokenizer natural_tokenizer_24k.json \
  --data-dir natural_data_240m \
  --out-dir natural_ab_10m
```

If any MoE gate fails, `dense-10m.pt` is the continuation checkpoint. The completed screen passed
all gates and selected `moe-10m.pt`; this is a mechanical result, not a subjective architecture
choice.

## Main run and time budget

The selected 10M-token checkpoint continues to 240M total tokens or 18 GPU hours for the main stage.
The dedicated trainer provides:

- 1% linear warmup followed by cosine decay;
- fixed-seed validation every 10M tokens;
- immutable checkpoints every 25M tokens;
- immediate stop on non-finite loss;
- stop after 30M tokens without validation improvement;
- a hard GPU-time limit;
- an automatic termination gate at 200M tokens;
- a final 40M-token mix of 35% raw / 65% chat only when the 200M chat termination gate fails.

```bash
uv run python -m fractal.natural_train \
  --variant moe \
  --tokenizer natural_tokenizer_24k.json \
  --data-dir natural_data_240m \
  --out-dir natural_main \
  --resume
```

For a fresh main run, the variant and both initialization paths must match the A/B report. Gradient
checkpointing is disabled by the target-GPU benchmark; it may be re-enabled only if another target
GPU fails the declared speed/VRAM gate.

The detached single-GPU orchestrator builds missing immutable artifacts, runs the preflight and A/B
gate, resumes the main checkpoint when present, and restores a managed inference service on exit:

```bash
scripts/start-natural-cortex-gpu.sh \
  /srv/rtai /srv/natural-cortex-run /srv/inference-service
```

Set `NATURAL_CORTEX_SERVICE_NAME` when the managed Compose container is not named `llama-swap`.

## Local teaching and persistence

The base checkpoint is immutable. Mutable state is split into:

- a versioned append-only skill bank;
- fast-weight conversation state;
- expert rollback revisions;
- an append-only audit log of demonstrations, ratings, activations, and verification results.

A new candidate has random nonzero `down` weights and zero `up` weights. Its output is exactly zero,
but the zero-output state has a nonzero gradient into `up`. Teaching freezes the base model and
updates only that candidate for 16–64 steps with early stopping and optional anchor replay.

The public runtime operations are:

- `chat`;
- `suggest_skill`;
- `propose_skill`;
- `teach`;
- `rate`;
- `activate`;
- `quarantine`;
- `rollback`;
- `calibrate_addresses`;
- `restart_verification`.

One to three demonstrations are accepted only after the proposed name and synopsis are confirmed.
Rating 4–5 commits the candidate, rating 1–2 rolls it back and quarantines it, and rating 3 discards
it without a durable skill change. A proposed skill requires confirmation on first activation in a
conversation and then remains sticky for that trajectory.

Start the dashboard runtime with one command:

```bash
scripts/run-natural-cortex.sh \
  CHECKPOINT.pt natural_tokenizer_24k.json natural_runtime
```

The dashboard reports the actual active expert, local expert gradient, update norm, before/after
behavior, loss, anchor KL, and routing confidence. Skill operations are exposed under
`/api/skill/*`.

Natural Cortex does not enable Event Algebra message ratings or `W0` consolidation. Those archived
experiments require a separate explicit `VIZ_FEEDBACK=1`; ordinary `VIZ_CHAT=1` and the Natural
Cortex launcher leave them disabled.

## Acceptance gates

The base model is accepted only if:

- at least 18/20 fixed chat prompts terminate without role leakage;
- at least 12/20 simple English prompts are manually judged relevant.

The online skill system is tested on ten naturally described skills, each with one to three
demonstrations and five handwritten paraphrases:

- at least 7/10 skills improve held-out score by at least 30 percentage points;
- control hijack is below 10%;
- anchor chat loss worsens by no more than 5%;
- post-restart skill score remains within two percentage points;
- one skill is learned within 60 seconds;
- the oldest skill retains at least 70% of its best score.

Failure is reported by subsystem. If chat fails, the result is data/scale-gated. If chat passes but
skills fail, teaching objective, address routing, and expert capacity are evaluated separately.

The final runner consumes an explicit JSON specification containing 20 chat prompts, manual
relevance labels, ten skills, five accepted-answer paraphrases per skill, unrelated controls, and
anchor conversations:

```bash
uv run python -m fractal.natural_eval \
  --checkpoint CHECKPOINT.pt \
  --tokenizer natural_tokenizer_24k.json \
  --spec natural_eval_spec.json \
  --runtime-dir natural_eval_runtime \
  --output natural_eval_report.json
```

Missing manual relevance labels fail the corresponding gate rather than being replaced by an
automatic proxy.
