# RTAI Roadmap — from scaffold to a scale-invariant full model

**Status:** proposed direction. The current agent is a deliberate *prove-in-small* scaffold
(FSM-constrained tool syntax, keyword routing, a synthetic "caveman" curriculum) that lets a
tiny model *act* agentic without being a competent LLM. This roadmap replaces the scaffold with
a design that trains on ordinary assistant/agent data and produces agentic behaviour
emergently — while still running on a laptop-class 4GB GPU for testing.

## Guiding principle

**One codebase, size = config.** A small preset runs on a laptop-class 4GB GPU (e.g. an
RTX A2000) for testing; a large preset runs on a cluster. Identical code, data format,
training loop, and agent loop — only the numbers differ. Nothing is throwaway: what is built
small runs large unchanged (VIBE #3, #6).

## North star — what the "full model" is

- **Fractal fast-weight LM**, identity preserved: weight-tied recursion over depth + a γ ladder
  of timescales + the fast weight `W` (untie is an option, not the identity — VIBE #4/#5).
- **Context is the persistent fast-weight state `W`** (O(1), no KV cache), not a growing prompt
  window. This is the differentiator vs. ordinary agents (VIBE #4): conversation history and
  tool results are *absorbed into `W`*, not concatenated into the prompt.
- **One chat/tool template everywhere.** The model **emits tool calls itself**; the harness
  executes them and **streams the result back into `W`**. No FSM, no keyword routing, no
  caveman curriculum.
- Trained on **real assistant/agent data** (smoltalk / tulu / glaive / hh-rlhf / opencode) via
  **train-as-deploy**.
- Output quality scales with size; caveman-level output is acceptable at small scale
  (VIBE #7); fluency emerges at scale — same code.

## Removed (crutches that don't scale) vs. kept

| Removed | Kept (real + scales) |
| --- | --- |
| FSM-forced `<tool>` syntax | fast-weight persistent memory (the identity) |
| `wants_tool` keyword routing | fractal weight-tied recursion |
| copy-constrained tool arguments | a registry of real tool executors |
| synthetic caveman templates as the target | config-driven size knobs |
| the `User:/Bot:` format (divergent from the data) | the dashboard + honest evals |

## Phases (each testable small, each runs large)

### Phase 0 — Unified format (code, CPU; no GPU contention) — ✅ DONE
Define one protocol: `<|system|> <|user|> <|assistant|> <|tool_call|>{json} <|tool_result|>…
<|end|>`, as **literal strings** tokenised by the existing 32k byte-BPE — **no new tokenizer,
so later phases can continue from the current checkpoint** (VIBE #10). One renderer in
`data_mix.py` serialises **every** source into it (glaive tool calls → `<|tool_call|>`, function
responses → `<|tool_result|>`). Retire the divergent `User:/Bot:` format. Deliverable: a unit
test for the renderer.

*Delivered:* the protocol lives in `fractal/chat_format.py` (single source of truth for the
markers + `render()` + per-source normalizers: messages/ShareGPT, glaive, hh); `data_mix._render`
now serialises every chat/instruction/tool source through it (glaive `<functioncall>` →
`<|tool_call|>` with repaired-to-valid-JSON payload, `FUNCTION RESPONSE:` → `<|tool_result|>`,
each conversation terminated by `<|end|>`; pretrain `text` stays raw). Unit test:
`fractal/tests/test_chat_format.py` (20 checks, all green). Retiring `User:/Bot:` in the *runtime*
agent/chat loop is coupled to Phase 1 (the agent-loop rewrite) and is done there.

### Phase 1 — Emergent agent loop (code, CPU) — ✅ DONE
New `agent.py` loop: generate with the persistent-state path (`generate_stream`, carrying `W`)
until a `<|tool_call|>` or `<|end|>`; parse the call (name + JSON); execute via a clean tool
**registry**; stream the `<|tool_result|>…` back through the model (chunk mode) to update `W`;
continue. The model **decides** when and which tool to use. Demote `grammar.py` to an *optional*
inference-time JSON-validity guard (off by default — not a crutch). Deliverable: the loop runs
on the current checkpoint (weak output, but it is the real loop that scales).

*Delivered:* `fractal/agent.py` rewritten to exactly this loop — token-by-token generation over
`forward_stream` (carrying `W`) with marker detection, a tool **registry** (`calc`/`time`/`echo`,
extensible), `<|tool_result|>` streamed back into `W`, bounded by a per-span token cap and a
tool-call budget so it terminates on any checkpoint. `User:/Bot:` retired; `grammar.py` is no longer
imported (an optional JSON-repair guard is inline, `--json-guard`, off by default). Verified: runs
end-to-end on `fractal_ckpt_big.pt`, tool parse/execute/guard unit-checked, `W` survives save/load
(persistence). Output is weak (that checkpoint predates this format) — expected; Phase 2 trains it.
Also delivered here: the **opencode data adapter** (task #43 data side) — `cf.opencode_segments()` +
`data_mix --opencode_file` fold a local opencode export (HF `messages`) into the unified mix.

### Phase 2 — Train-as-deploy on real data, small scale (laptop GPU) — ✅ prove-in-small DONE
Training loop: a multi-turn trajectory streams its context through a **carried, detached
fast-weight state** (the TBTT primitive already exists); cross-entropy is applied **only to the
assistant + tool_call tokens** (user / tool_result masked). This generalises the memory-as-a-
property work (repo task #41) to full agentic turns. The mix is real data in the unified
format; synthetic memory/tool episodes are at most a small, same-format augmentation (not a
scaffold). Honest evals (VIBE #9): held-out instruction loss, tool-call validity + success on
held-out trajectories, recall across turns **and across a restart** (VIBE #8/#10), perplexity.
Baseline at the small size; the dashboard shows it (VIBE #1). **Continues fine-tuning from the
existing checkpoint — not from scratch.**

*Delivered (prove-in-small, 2026-07-06):* the masked objective is wired end-to-end —
`chat_format.render_pieces()` marks the trainable span (assistant + `<|tool_call|>` + `<|end|>`),
`data_mix --emit_mask` writes a token-aligned `*.mask.bin`, `data.get_masked_batch()` yields the
per-token weight, and `train.py --task chat` runs masked TBTT over a carried, detached state.
A 20M-token unified instruct/tool mix (smoltalk/glaive/hh/dolly/tulu + opencode, 4% of it Czech —
structure-signal) was fine-tuned from the 126M v2 checkpoint (lr 5e-5, 3000 iters, batch 2 ×
seg 4, peak 2.44 GB on a 4GB GPU). **Honest result:** held-out masked assistant loss 3.43 → 3.21
(ppl ≈30.7 → 24.9), and — the point — the model **learned to speak the format**: emitting `<|end|>`
went 0/8 → 7/8 and it stopped spilling into other role markers. Tool calls are NOT yet clean
(0/8 valid `<|tool_call|>`; it reaches for tool syntax but mangles the marker/JSON) — needs more
tool-heavy data / iters / scale. Output stays caveman at 126M (VIBE #7), as expected. The mechanism
is proven; competence and tool reliability are for Phase 4 scale.

### Phase 3 — Scale readiness (code + small verification)
Config presets: small (4GB, grad-checkpointing on, compile off) ↔ large (cluster, grad-
checkpointing off, compile on, large batch). Multi-GPU: wrap for DDP/FSDP (the weight-tied
block is FSDP-friendly; make telemetry / MoE-balance / neurogenesis rank-safe); a sharded
`.bin` loader. Verify the **same command** runs the small config locally (VIBE #3) and, if a
second GPU is available, a 2-process DDP smoke.

### Phase 4 — Scale on the cluster
Same artefacts, large config + full data (+ the opencode source, repo task #43). Here the speed
levers that do not fit in 4GB finally pay off (drop grad-checkpointing ~1.3×, compile, large
batch — see `docs/EXPERIMENTS.md` on the 4GB throughput ceiling). Watch for the emergence of
fluency, tool use, and persistent memory; evaluate honestly against the small-scale baseline.

## Research bet (stated honestly)

The differentiator is using O(1) fast-weight state as the context / working memory instead of a
KV cache. Whether it holds enough capacity for long multi-turn agentic use at scale is the open
question. Falsification = the Phase 2 recall-across-many-turns evals. If `W` hits a capacity
ceiling, the mitigations are more scales / a larger head dimension / γ-ladder tuning /
function-preserving neurogenesis (VIBE #5) — all config, still scale-invariant.

## Immediate next step

Phases 0, 1, and 2 (prove-in-small) are **done**: unified format `fractal/chat_format.py`, `data_mix`
wiring + opencode adapter, the emergent `agent.py` loop, and masked train-as-deploy (`--task chat`)
fine-tuned from the 126M checkpoint on a 4GB GPU — the model measurably learned the format
(masked loss 3.43→3.21; `<|end|>` emission 0/8→7/8; tool calls not yet clean). Next, two tracks:
- **Deepen Phase 2** (still small): more tool-heavy data + iters so `<|tool_call|>` becomes clean;
  add the recall-across-restart eval (VIBE #8) to test whether `W` holds multi-turn memory (the
  research bet); truncate giant opencode tool outputs.
- **Phase 3 — scale readiness** (code): small↔large config presets, DDP/FSDP wrap (rank-safe
  telemetry/MoE/neurogenesis), a sharded `.bin` loader; verify the *same command* runs small locally.
Then **Phase 4** — the same artefacts at scale on the cluster, where fluency and tool use emerge.

## Moonshot backlog

These are deliberately not scheduled implementation phases. They stay as falsifiable alternatives
to iterative gradient training and must preserve the architecture and persistent-memory invariants.

- **TODO — Spectral Birth.** Compile additive prefix/suffix moment sketches from one corpus pass,
  recover a low-rank recurrent state with randomized SVD, and algebraically factor its transitions
  into the shared fractal rule. Falsify it if the required spectral rank grows with corpus size rather
  than task complexity, or if a small compiled model cannot approach the loss and held-out recall of
  a compute-matched gradient baseline without optimizer steps.
- **TODO — Neural Decompiler.** Record projected state transitions and logits from a capable teacher,
  solve the linear parts of a constant-state fractal surrogate in bulk, and discard the teacher. The
  resulting model must retain one tied rule, persistent fast weights, and no KV cache. Falsify it if
  the constant state cannot reproduce held-out behavior and restart recall without retaining teacher
  calls or sequence-growing state.
