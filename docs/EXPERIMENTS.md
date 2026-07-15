# FractalLM — lab notebook

A fractal language model from scratch (package `fractal/`). Goal: replace the "matrix" (a static
stack of layers, von-Neumann scaffolding) with a **generating rule** — one shared rule
unrolled across depth × a ladder of timescales. Memory lives in the **fast-weights W** (delta
rule, adjusted at runtime), not in a growing KV-cache.

> This notebook contains chronological results. Branch names and local artifacts from the original
> private research history are archival context, not public release interfaces.

## 2026-07-15 — 4 GB efficiency tournament: no hypothesis passed

The complete four-hour falsification screen finished without a runtime failure. No arm qualified
for the 12 GB follow-up (`promote_to_12gb: []`). A sanitized machine-readable decision summary is in
[`results/efficiency-tournament.json`](results/efficiency-tournament.json); the protocol is in
[`EFFICIENCY_TOURNAMENT.md`](EFFICIENCY_TOURNAMENT.md).

| arm | avg train tok/s | val loss | recall 1/3 facts | exact tool execution | restart recall | gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 31,481 | **4.651** | **40.6% / 20.3%** | 0/60 | yes | reference |
| sampled-depth genome | 25,078 | 4.638 | 28.1% / 17.2% | 0/60 | no | fail |
| soft MoE | 15,182 | 5.943 | 0% / 0% | 0/60 | no | 5-minute speed control |
| top-1 MoE | 18,612 | 4.772 | 10.9% / 4.7% | 0/60 | no | fail |
| event patches | 24,941 | 5.552 | 0% / 0% | 0/60 | no | fail |
| local credit | **47,532** | 4.868 | 1.6% / 3.1% | 0/60 | no | fail |
| typed-action compiler | 26,334 | 5.111 | 18.8% / 14.1% | 0/60 | no | fail |

### Failure analysis

- **Sampled-depth genome:** one shared rule was asked to work at incompatible depths without a
  depth-consistency objective. Depth 4 had the best loss; depth 8/16 regressed. Only one of three
  configured depth metrics improved, and the recall samples were not paired across depths, so this
  run provides no clean scale-invariance evidence.
- **Top-1 MoE:** sparsity removed only about 6.4% of active parameters because the embedding/head
  dominates this small model. Unfused Python indexing and scatter cost more than the skipped expert
  work. The short soft-MoE arm is a speed control, not a quality comparison.
- **Event patches:** four tokens were averaged, memory ran only at the patch boundary, and its
  residual was written only to that position. The other 75% of positions received no memory output;
  token identity/order and the effective decay cadence changed. A Python token loop and scatter also
  erased the expected speed gain.
- **Local credit:** seven of eight steps optimized one detached recurrence rather than the full tied
  recurrence. Weight sharing does not make that local derivative equal to the full unrolled
  derivative. It was 1.51x faster than baseline by average throughput, but recall collapsed despite
  processing 85.6M tokens: this is objective bias, not simple undertraining.
- **Typed-action compiler:** training placed `<|tool_call|>` directly after the user turn, while
  inference primes `<|assistant|>` first. The short synthetic records were mostly padding and were not
  packed. A batch-wide `loss_scale=0.5` cancelled in normalized weighted CE, and the objective was
  imitation rather than execution feedback. It produced one syntactically valid call out of 60 and
  zero correct executions.

### Decision and next small tests

The fixed-depth, full-backward baseline remains the only coherent training path in this screen, but
it is not yet an agent: exact tool execution was zero. The strongest negative results are the current
event/local-credit implementations and the compiler protocol; genome and MoE remain weaker negatives
because the former needs paired depth evaluation and the latter needs a materially sparse, fused
region. Before any scale-up: align and pack compiler records; measure local/full gradient agreement at
shorter full-backward intervals; broadcast or carry vectorized event state with corrected decay; and
rerun genome depth checks on identical episodes. This was one seed (64 recall episodes, 60 tool
prompts, one restart episode), so the result rejects these implementations, not every possible member
of each idea family.

## 2026-07-05 — scaffold → emergent, scale-invariant agent (ROADMAP Phases 0–1)

Migration away from the prove-in-small scaffold (FSM tool syntax, keyword routing, caveman
curriculum) toward the design in `docs/ROADMAP.md`: one codebase, size = config, agentic behaviour
learned from ordinary data. Code-only so far (CPU-testable, no GPU contention), continues from the
existing checkpoint (no new tokenizer).

- **Phase 0 — one unified chat/tool format (done).** `fractal/chat_format.py` is the single source
  of truth for the markers `<|system|> <|user|> <|assistant|> <|tool_call|>{json} <|tool_result|>
  <|end|>` (literal byte-BPE strings; no retrain). `render()` + per-source normalizers
  (messages/ShareGPT, glaive, hh). `data_mix._render` serialises every chat/tool source through it
  (glaive `<functioncall>` → `<|tool_call|>` with repaired-valid-JSON, `FUNCTION RESPONSE:` →
  `<|tool_result|>`). Unit test `fractal/tests/test_chat_format.py`.
- **opencode data adapter (ROADMAP #43, done).** `cf.opencode_segments()` maps the
  opencode-training-exporter HF-`messages` records (inline `<think>`, OpenAI `tool_calls`/`tool`)
  into the unified format; `data_mix --opencode_file` mixes a local export in at a low weight.
  Real-data check (41 sessions): coherent user→`<think>`→tool_call→tool_result→answer loops
  (35/41 full), but ~1/3 are Czech (base is English) so the transferable signal is the tool-call
  *structure* — keep the weight low. Giant tool outputs (≤56 KB) are a future cap candidate.
- **Phase 1 — emergent agent loop (done).** `fractal/agent.py` rewritten: generates in the unified
  format carrying W (persistent, O(1), no KV-cache), emits `<|tool_call|>` itself, a clean tool
  **registry** executes it, the `<|tool_result|>` is streamed back through the model (W absorbs it),
  looping until `<|end|>`. `User:/Bot:` retired; `grammar.py` no longer used (optional inference-time
  JSON guard via `--json-guard`, off by default). Verified: the loop runs end-to-end on
  `fractal_ckpt_big.pt` and W survives save/load — output is weak (that checkpoint predates this
  format, as expected) but it is the real loop that scales.
- **Phase 2 — masked train-as-deploy (prove-in-small, done).** The objective is wired through:
  `chat_format.render_pieces()` marks the trainable span (assistant + `<|tool_call|>` + `<|end|>`),
  `data_mix --emit_mask` writes a token-aligned `*.mask.bin`, `data.get_masked_batch()` yields the
  per-token weight, `train.py --task chat` runs masked TBTT over a carried, detached state (CE on
  the marked tokens only). Fine-tuned the **126M v2 checkpoint** on a **20M-token** unified
  instruct/tool mix (smoltalk/glaive/hh/dolly/tulu + opencode; ~4% Czech = structure-signal, low
  weight) — lr 5e-5, 3000 iters, batch 2 × seg 4, **peak 2.44 GB on a 4GB GPU**.
  *Honest evals (fixed held-out set):* masked assistant loss **3.43 → 3.21** (ppl ≈30.7→24.9;
  most gain by iter 500 then plateau). Format emission over 8 prompts: **`<|end|>` 0/8 → 7/8**
  (learned to stop + no longer spills into other role markers); `<|tool_call|>` 0/8 (reaches for
  tool syntax — once emitted `<tool_call>[…]` — but mangles the exact marker/JSON). *Lesson:* the
  frequent terminator `<|end|>` is learned fast; sparse tool-call spans need more data/iters/scale.
  Output stays caveman at 126M (VIBE #7). **The mechanism is proven small; competence + tool
  reliability are for scale (Phase 4).** LR note: first attempt at lr 2e-4 (no warmup) regressed —
  5e-5 is the SFT sweet spot here. Base v2 preserved as `fractal_ckpt_agent_v2.pt`; Phase 2 result
  as `fractal_ckpt_chat_phase2.pt`.
- **Phase 2 deepening (phase2b) — a MIXED, honest result (multi-objective trade-off).** Added:
  (a) a `TOOL_RESULT_CAP` (truncate giant opencode tool dumps, up to ~56KB), (b) a `tools` recipe
  (glaive-majority + chat + opencode → 18M tokens, high tool-call density), (c) **recall-episode
  interleaving** in `--task chat` via `--recall_ratio` (a small RecallGen memory augmentation, so the
  fast-weight recall skill is not eroded by masked chat/tool SFT — VIBE #4/#8). Continued from
  phase2 (block 192 x seg 4, batch 2, recall_ratio 0.2, lr 5e-5, 4000 iters, peak 2.16 GB).
  Three-way eval (fixed chat32 val for masked loss; 8-prompt format; 20-trial recall-restart):
  masked val loss v2 3.43 / phase2 3.21 / phase2b 3.42; end-emission 0/8 / 7/8 / 1/8; valid
  tool_call 0/8 / 0/8 / **1/8**; recall-across-restart 40% / 15% / 25% (synthetic RecallGen 1-fact
  94/84/84/80%). Honest read: the tool-heavy retrain **won the tool-call format** (first valid
  `<|tool_call|>{json}` — right shape, wrong tool/args) and **partially restored recall** (15%->25%),
  but **over-specialized** — `<|end|>` emission collapsed (7/8->1/8) and general-chat coherence/loss
  regressed (glaive at 58% of the mix pulled it toward its narrow tool-transcript style; chat32 val
  is off-distribution for it). **Lesson:** no single small model wins all three; the mix must be
  **balanced** (tools + chat + recall, not glaive-dominant) and probably needs more iters / gentler
  LR. Persistent memory across restart (beating a 0% no-memory baseline) is robustly confirmed
  throughout. **Next = a balanced mix (glaive ~25-35%, keep chat + recall interleaving), then Phase 3
  scale readiness.** Result checkpoint: `fractal_ckpt_chat_phase2b.pt`.

## Architecture (where things are)

- `fractal/cell.py` — the atom: delta rule `W ← γW + β(v−Wk)kᵀ`, chunk-parallel variant
  (`delta_chunk`) + decay variant for γ<1 (`delta_chunk_decay`, solve always fp32 at `cell.py:94`).
- `fractal/unit.py` — `FractalUnit`: n_scales timescales `τ_ℓ=τ₀·ρ^ℓ`, `γ_ℓ=exp(−1/τ_ℓ)`;
  value v = raw residual state (generalizes copy/recall); learned gate (softmax over scales).
  `FractalState.W` = list per scale (B, n_head, hd, hd).
- `fractal/model.py` — `Config` + `FractalLM`: one shared `Block` unrolled `depth`× (weight-tied),
  or `--untie` = own block/depth (more parameters). Gradient checkpointing toggle.
- `fractal/train.py` — training; `--data_dir` (different corpus), `--bf16 --tf32`, fused AdamW,
  `--save_every/--resume`, `--recall_ratio/--gate_lambda` (recall curriculum), TBTT.
- `fractal/data_web.py` — FineWeb-Edu stream + 16k BPE → `fractal_data/`.
- `fractal/data_chat_ft.py` — DailyDialog through the EXISTING 16k tokenizer → `fractal_data_chat/`
  (safe, does not overwrite pretraining data). Format `User:/Assistant:`.
- `fractal/chat.py` — persistent chat (`--role` for User/Assistant), `fractal/viz_serve.py` +
  `fractal/web/fractal3d.html` — architecture and sampled-telemetry dashboard.

## How it differs from a regular transformer

| | regular transformer | FractalLM |
|---|---|---|
| layers | N different matrices | 1 shared rule × depth (or untied) |
| context | growing KV-cache (O(T) memory) | constant state W (O(1)) |
| position | positional embeddings | keeps a recurrent state |
| attention | softmax O(T²) | delta-rule linear O(T) |
| runtime memory | frozen (only KV) | W is adjusted (self-modifying) |

## Measurements (honest numbers, not estimates)

### Training speed vs Tessera-1B (AIIT-Threshold/Tessera-1B)
Matched config (d_model 512, 6 layers/unrollings, 8 heads, vocab 8k, batch 2, bf16, on a 4GB A2000),
training step fwd+bwd+step, tok/s:

| seq | Tessera (attention) | Fractal (delta) | ratio |
|----:|--------------------:|----------------:|------:|
| 256 | 22 073 | 4 671 | 0.21× |
| 512 | 31 467 | 4 965 | 0.16× |
| 1024 | 41 734 | 4 951 | 0.12× |
| 2048 | 49 076 | 4 711 | 0.10× |
| 4096 | 49 460 | **OOM** | — |

Tessera **5–10× faster**, speeds up with length, less memory. **Why:** flash-attention is a fused,
optimized kernel; our delta-kernel is unfused einsums + fp32 solve ×scales ×depth → low
MFU. The O(T) advantage only kicks in at much longer sequences (16k+) AND with a fused kernel. **Reference:**
Tessera-1B trained on 24.5B tok / ~6 days / 1× H100 ≈ **47.5k tok/s** (full 1B). Our architecture
buys O(1) inference and unbounded context, NOT training speed.

### Speeding up training without losing quality (A/B on 109M, ~10 iter)
| variant | s/iter | peak VRAM |
|---|---:|---:|
| fp32, batch 2×accum 8, +ckpt (baseline) | 8.5 | 2.19 GB |
| bf16, batch 2×accum 8 | **9.7 (SLOWER!)** | 2.19 GB |
| bf16+tf32, **batch 4×accum 4** ✓ chosen | 5.7 | 2.38 GB |
| bf16+tf32, batch 8×accum 2 | 5.4 | 2.93 GB |

Lesson: **bf16 alone with a small batch is slower** (overhead of casts on small kernels) — the speedup
came from bf16 **+ a larger batch**. Effective batch 16 preserved (mathematically identical gradient),
fused AdamW, TF32. Steady-state of the resumed run ~4.7 s/iter ≈ **1.6× vs the fp32 baseline**.

## Runs

### Pretraining 109M (FineWeb-Edu) — PAUSED at iter 1800
Config: n_embd 1024, n_head 16, depth 12 **untied**, n_scales 4, vocab 16k, block 512,
batch 4×accum 4, bf16+tf32, grad_ckpt. **108.73M params, peak 2.38 GB.**

val_ppl: 6957 (0) → 337 (400) → 283 (600) → 174 (1000) → ~150 (1800). Cosine LR over 20000 iter
(done ~9%). Checkpoint `fractal_ckpt_big.pt` (+ `.resume`) current as of iter 1800.

**Resume** (continues from iter 1800, `--iters` must stay 20000 for the LR schedule):
```
nohup uv run python -m fractal.train --iters 20000 --batch 4 --accum 4 --block_size 512 \
  --segments 1 --n_embd 1024 --n_head 16 --depth 12 --n_scales 4 --vocab_size 16000 \
  --untie --grad_ckpt --bf16 --tf32 --save_every 200 --resume --out fractal_ckpt_big.pt \
  >> train_big.log 2>&1 &
```

### Conversational fine-tune (DailyDialog) — DONE as a dry-run
From a COPY `fractal_ckpt_chat.pt` (big checkpoint untouched), `--data_dir fractal_data_chat`
(1.88M tok, 16k tokenizer, User/Assistant), 500 iter, lr 2e-4, bf16+tf32+grad_ckpt, batch 4×accum 2.
Peak 1.76 GB. val_ppl (dialog) 58.8 → 18.4 (min @225) → ~20 (noise).

**Result:** the User/Assistant turn-taking format was **learned to 100%** (`--role` framing + stop works),
grammar valid. BUT **no understanding** (answers don't follow meaning), DailyDialog ESL artifacts
("Mr. Wang"), a self-priming loop ("the best" repeated — fast-weights live). Low ppl =
the dialogs are more predictable + the model narrowed, NOT that it got smarter.
**Conclusion: the only lever for coherence = finish training the base, then rerun the same fine-tune.** More data now
won't help (the ceiling = weak pretraining). Chat: `uv run python -m fractal.chat --ckpt fractal_ckpt_chat.pt
--state fractal_chat_state_ft.pt --role --fresh`.

## Visualization :8008 (faithful + sampled telemetry)
`FRACTAL_CKPT=fractal_ckpt_big.pt PORT=8008 CUDA_VISIBLE_DEVICES= uv run python -m fractal.viz_serve`
Nested self-similar contours show the real repeated D1→Dn computation and its
q/k→β→fast-W→gate/projection→MLP path. Tied-weight fibres and timescale branches describe the
architecture; highlights and measurements come only from fresh sampled telemetry. `VIZ_ATTACH`
mirrors a training run without an extra model pass. Without an attachment, activity is inference
over a validation stream, not training.

## What (doesn't) work
- ✅ Fractal LM from scratch, decay-chunk kernel, single-fact generalized recall (value=raw-state),
  persistent chat, live continual learning, architecture/telemetry dashboard, scaling to 109M on 4GB.
- ✅ **Multi-fact discrimination — SOLVED** (revision 2026-07-03): with enough training (6000 iter,
  block 256, full BPTT recall) even a 2-scale 4.58M model gives **~96–100% held-out recall on 1–6 facts**
  at any distance. The earlier "key collapse (cos ~0.97, fundamental)" was an artifact
  of short/different training, NOT a fundamental. **Caveat:** with a small pool of 6 names it's partly memorization —
  on UNSEEN keys it drops to ~25% (the model generalizes values, keys less so). See the neurogenesis A/B below.

## Key checkpoints (gitignored)
`fractal_ckpt_big.pt` (109M pretraining @1800) · `fractal_ckpt_chat.pt` (fine-tune) ·
`fractal_ckpt.pt` (old small — overwritten by a smoke test) · tokenizer `fractal_tokenizer.json` (16k).

## NEUROGENESIS — implemented + A/B (2026-07-03)
Mechanism and policy done and tested (see `NEUROGENESIS.md`, section "Implementation state"):
`unit.grow_scale` (extending the ladder, invariant `gammas==_make_gammas(n)`, β_gain plasticity),
`fractal/neurogenesis.py` (controller: trigger = gate concentration + ‖W‖ saturation, maturation, cooldown),
`train.py --neurogenesis`, live growth in the dashboard (`viz_serve VIZ_NEUROGENESIS=1`).

### A/B v1 (easy task: 6 names, ≤4 facts) — INCONCLUSIVE (ceiling)
n_embd 256, depth 4, 4.58M, 6000 iter, recall_ratio 0.5, gate_lambda 0.03.

| variant | 3-facts HELD-OUT | scales | births |
|---|---|---|---|
| static-2 | 100% | 2 | — |
| static-4 | 96% | 4 | — |
| grow 2→≤4 | 96% | **3** (stopped on its own) | 1 @ iter 800 |

All at the ceiling → grow did not beat static-4 → hypothesis NOT supported, but the test is uninformative (task
easy, nothing to solve). **Side finding:** grow grew ON DEMAND — after the 1st birth the gate spread out,
concentration dropped below the threshold → growth stopped itself at 3 (it did not grow blindly to the ceiling). A nice property.

### A/B v2 (hard task: 256 names, ≤8 facts) — HYPOTHESIS REFUTED
Large name pool (`--n_names 256`, no memorization) + `--max_facts 8` → real capacity pressure.
The original one-off orchestration script was not retained; the configuration and measured table
below are therefore archival evidence rather than a turnkey reproduction recipe.

| variant | 1-fact(D48) | 3-facts | 6-facts | 8-facts | scales | births |
|---|---|---|---|---|---|---|
| static-2 | 95% | 92% | **67%** | **61%** | 2 | — |
| static-4 | 81% | 78% | **44%** | **47%** | 4 | — |
| grow | 78% | **20%** | **2%** | **0%** | 4 (2→3→4) | @800, @2923 |

**Two clean findings:**
1. **More fixed scales + load-balancing WORSENS recall** (static-4 < static-2). `gate_lambda 0.03` forces
   reading evenly even from the fast (γ<1) scales, which don't hold facts → static-4 takes ~75% of its reads from
   forgetful scales. Load-balancing (previously "needed against collapse") hurts on the recall task.
2. **Neurogenesis BROKE recall** (grow ≪ both static): 1-fact survives (78–92%), but multi-fact collapses
   (6-fact 2%, 8-fact 0%). Pattern = destroyed pattern separation. Mechanism: a birth DEMOTES the
   permanent (γ=1) scale, where the multi-fact binding was held, to γ<1 (it starts to leak) + adds an EMPTY
   permanent one + RESETS the optimizer + changes the ladder that the projections are calibrated to. The second birth
   @2923 (late) → too few iterations to recover under a decaying LR. Exactly the risk that
   `NEUROGENESIS.md` warned about ("changing the shape at runtime destabilizes what was learned"). Recovery did not happen.

### A/B v3 (hard task, gate_lambda 0, GENTLE growth) — rescue FAILED, hypothesis definitively refuted
Gentle growth: `--grow_no_demote` (no demotion of the permanent one), `add_param_group` (no optimizer reset),
births only early (`--grow_until 1500`). All 3 variants `gate_lambda 0` (isolation + the gate may concentrate).

| variant | 1-fact | 3-facts | 6-facts | 8-facts | gate |
|---|---|---|---|---|---|
| static-2 | 52% | 31% | 25% | 14% | 83/17 |
| static-4 | 52% | 33% | 16% | 14% | 76/8/6/10 |
| grow (gentle) | 33% | **2%** | **0%** | **0%** | 82/4/5/9 |

**Two things:**
1. **Correction of the v2 conclusion:** `gate_lambda 0` is the WORST — the gate collapses onto the FAST scale (83% on L0,
   γ=0.94), facts leak out. So λ=0.03 did help (it forced reading from the permanent one). "Load-balancing hurts" from v2
   was hasty: λ helps a few scales, its *forced uniformity* hurts many, turning it off is the worst.
2. **Even GENTLE growth collapses** (grow ≪ static). The unavoidable destruction that can't be bypassed: **a birth reshapes
   the GATE** (new Linear, reset of its Adam state, zero logits for the new scales). And the gate IS the bottleneck
   (routing to the permanent scale = recall). Growth necessarily destroys exactly the component everything rests on.

**DEFINITIVE CONCLUSION:** Neurogenesis (growing timescales at runtime) does NOT improve multi-fact recall and actively
hurts — refuted across 3 rounds, destructive even in the gentle variant. **The real bottleneck is the GATE ROUTING
(reading from the permanent scale), not capacity (the number of scales).** Adding scales doesn't solve the problem, it only worsens it.
The right lever = a better gate/routing (mild λ, a "recall → permanent" prior, or a better gate architecture),
NOT neurogenesis. An honest negative result with a mechanistic explanation.

### Test of the routing hypothesis: decay-aware gate prior (2026-07-03) — routing is NOT a sufficient lever
New option `--gate_prior_perm p`: CE to a prior holding the permanent scale at p (instead of neg-entropy→uniform);
`unit.pop_gate_balance_loss`. Hard task (256 names, ≤8 facts).

| arm | 1-fact(D48) | 3-fact | 6-fact | 8-fact | scales | gate (permanent) |
|---|---|---|---|---|---|---|
| A0 static-2 λ0.03 | 98% | 70% | **55%** | **31%** | 2 | 57% |
| A1 static-4 +prior0.5 | 64% | 56% | **42%** | **27%** | 4 | 48% |
| A2 static-4 +prior0.5 λ0.1 | 53% | 41% | **23%** | **23%** | 4 | 48% |

**The prior controlled routing** (permanent held at 48% regardless of λ) — mechanically it works. **But it didn't rescue
recall:** A1/A2 (4 scales) stayed BELOW A0 (2 scales), stronger λ (A2) even worse (over-regularization).
Tell: even **1-fact** is worse with 4 scales (64/53% vs 98%) — a single fact needs neither capacity nor routing →
the problem isn't WHERE the gate points, but that **the soft mix blends in reads from 3 other scales → it dilutes the clean lift
from the permanent one**, even for 1 fact.

**FINAL CONCLUSION of the recall line:** (1) neurogenesis/capacity doesn't help (it even hurts); (2) routing is
controllable, but "mass on the permanent one" is NOT a sufficient lever; (3) **2 scales (1 fast + 1 permanent)
are the sweet spot** of this recall task — each additional scale dilutes retrieval via the soft mix. Recall with 2 scales
is decent (v2: 6-fact 67%) and would potentially be solvable with a harder gate (top-1 routing), not by adding scales.
Honest note: the recall metric has run-to-run noise of ±~15% (static-2 6-fact: v2 67% vs 55% here).
The new CLI/code (`--gate_prior_perm`, `--n_names`, `--max_facts`, neurogenesis) remain on `feat/neurogenesis`.

### Cheap mechanistic probe (2026-07-04) — EXPLANATION of the whole recall line in 30 s
Instead of another expensive A/B, an inference-time ablation on trained checkpoints used
`unit.force_scale` to force reading from one scale and measured the gate at the answer position.

static-2 (gp_a0): normal 92/72/45 · **permanent only 14/16/5** · fast only 0/0/0 · gate@answer 54/46.
static-4 (gp_a1): normal 59/59/30 · individually max 44/17/20 · gate@answer L0 47 / L1 8 / L2 8 / L3 37.

**Key finding (refutes the coupling premise):** NO scale works on its own; the permanent one ALONE is dismal
(14/16/5 ≪ 45). **Recall is an emergent binding FAST×PERMANENT through the gate** — the hypothesis "the fast one dilutes the clean
permanent read, just subtract/concentrate" is FALSE (no clean permanent read exists).
Mechanism (hypothesis): at the answer "{name} … is __" the fast scale carries the fresh query context (τ=16, the name
fell a few tokens back), the permanent one carries the stored fact; the gate binds them. Even the 4-scale model effectively reads only
2 (L0+L3), the middle ~8% → vestigial → **2 scales = exactly the right structure** (store + fresh context),
which is why more dilutes and why neurogenesis and the routing-prior both failed. **The coupling-for-recall A/B thus falls away** (premise
refuted for free). Inter-scale routing potentially makes sense only for COMPOSITION → test it on the LM, not on recall.
Methodological takeaway: a cheap ablation on a finished ckpt > an expensive noisy end-to-end A/B.

### Per-scale plasticity A/B — a plasticity CURVE per timescale (2026-07-06)
New question (not capacity, not routing): does coupling *how much each part adapts* to the fractal
timescale help recall? A normal LM has one global LR curve; the ladder gives a per-scale knob. Harness:
`fractal/exp_per_scale_plasticity.py` (identical init + identical batch stream + identical optimizer
structure per arm; only the per-scale POLICY differs — a per-scale `W0` LR curve via param groups, since
AdamW is invariant to a constant grad scale, plus a per-scale fast-weight `beta_gain` curve). Four arms:
`none` (global cosine, baseline), `cool_perm` (fast scales stay plastic, permanent consolidates),
`cool_fast` (the opposite), `gate_driven` (plasticity FOLLOWS measured gate usage, self-tuning). Hard
recall (n_scales=3, 30 names), held-out (unseen-value) evals + soft answer log-prob (sensitive at ceiling)
+ a memory-overwrite/update probe.

**Arc (a clean VIBE #9 story):** the first single-seed run had a BUG (weight-tied → `W0` listed once per
depth in its param group → over-stepped) AND a ceiling (6 names → 1-fact at 100%); it looked like
`cool_perm` HURT. Fixing the dedup + a harder task made a single seed look like `cool_perm` WINS. **3 seeds
overturned that too** — the single-seed ranking was seed luck:

| metric (3 seeds, mean±std) | none | cool_perm | cool_fast | gate_driven |
|---|---|---|---|---|
| recall D=48 (mid) | 44±33 | 49±33 | 73±19 | **77±7** |
| recall D=400 (v.long) | 26±27 | 26±22 | 44±26 | **58±15** |
| recall 4-facts | 10±8 | 11±6 | 12±5 | **14±2** |
| logp 4-facts | −8.49±.45 | −8.44±.38 | −8.23±.29 | **−8.18±.18** |
| overwrite (update) | 17±12 | 18±12 | 26±10 | **27±7** |

**`gate_driven` wins every metric — highest mean AND lowest variance.** `cool_fast` second; `cool_perm` ≈
baseline (both high-variance). The robust takeaway: per-scale plasticity helps, but the winning form is
**self-tuning (plasticity follows where the gate routes), not a hand-picked fixed prior** — and its biggest
effect is **variance reduction** (baseline is wildly seed-dependent, ±30%; gate_driven ±7–15%) plus a real
lift at long distance (D=400: 58 vs 26) and on the overwrite/update skill.

**Bigger config (n_embd 512, depth 6, 1 seed, `none` vs `cool_perm`):** per-scale plasticity still crushes
baseline (recall D=200 66 vs 16, D=48 57 vs 12, overwrite 14 vs 3) — the mechanism holds/strengthens at
larger size. Caveat: 1 seed, and this arm tested `cool_perm` (chosen before the multi-seed reversal), NOT
the robust winner `gate_driven`, which remains untested at bigger size.

**Conclusion:** unlike capacity (neurogenesis, refuted) and the routing prior (insufficient), a **per-scale
plasticity schedule is a lever that robustly helps** this recall task, provided it is **usage-driven, not a
fixed timescale prior**. Per VIBE #11 this VALIDATES the mechanism at small scale (and its stabilizing
effect), not its magnitude at scale. Honest caveats: one synthetic task; `beta_gain` is a non-persistent
buffer → in-process eval; the per-scale SLOW surface is only `W0` (projections/MLP are shared by the
weight-tied design), so most of the effect rides on the fast-weight `beta_gain` curve; bigger-config is a
single seed. Next: `gate_driven` at bigger config + more seeds. Code on `experiment/per-scale-plasticity`.

**Confirmation + LM safety (2026-07-07) → adopted as the DEFAULT plasticity.** Re-ran at bigger config
(n_embd 512, depth 6, 2 seeds) and added a language-modeling arm (plain LM on the corpus, val perplexity)
to check the policy does not cost general fluency (the decisive test for a default, since the default
trainer also does LM).

| | none | cool_perm | cool_fast | gate_driven |
|---|---|---|---|---|
| recall D=48 (512×6, 2 seeds) | 54±43% | 70±13% | 86±3% | **89±0%** |
| recall D=200 | 54±39% | 68±3% | 75±2% | **77±1%** |
| overwrite (update) | 18±15% | 19±5% | 27±0% | **34±4%** |
| **val perplexity (LM, ↓)** | 247.1±4.1 | 246.9±4.5 | 247.5±4.0 | **246.5±1.7** |

`gate_driven` holds at bigger config (again best on recall, and again the standout is **variance
collapse** — baseline is a ±40% coin-flip, gate_driven ±0–5%) **and is neutral on LM perplexity**
(246.5 vs 247.1, well inside the ±4 noise). Net: a large gain on the model's identity axis (memory /
unseen-fact recall / memory-update) at **no cost to fluency**.

**Decision:** `gate_driven` (usage-driven per-scale plasticity) becomes the **default** learning policy
(`fractal/plasticity.py`, `train.py --plasticity gate_driven`, on by default; `--plasticity none` restores
the old global cosine). It is a TRAINING-TIME schedule — its effect is baked into the learned weights — and
the final per-scale `beta_gain` is now persisted with the checkpoint (`persist.save_model`) so inference
matches training. Honest scope (VIBE #11): validated as a MECHANISM at small scale on synthetic recall +
corpus LM; the magnitude at large scale and the interaction with the full agentic/chat objective are not
yet proven. Mutually exclusive with neurogenesis (which owns `beta_gain` and grows scales) — if both are
requested, plasticity yields and prints a note.

### Plasticity Genome (2026-07-15) — sequence learning works, general recall does not

Question: can a 32-number, scale-invariant local law replace iterative backpropagation? Each candidate
starts from the same random model, sees every sample once, and changes slow and fast weights under
`torch.no_grad()` with explicit rank-one error/Hebbian/Oja updates. An antithetic CEM search ran on four
NUMA-pinned CPU workers; only objective elites used a 12 GB GPU. AdamW was measured separately.

The test harness changed only when a concrete confound was observed:

1. Exact recall alone was flat at zero despite up to 52% sequence-loss improvement. Continuous recall CE
   exposed a gradient-free selection signal.
2. Answer error was drowned by filler tokens and could not credit an earlier fact. Answer-only masks plus
   an evolved eligibility trace produced the first non-zero recall.
3. Token priors could mimic memory. A paired ablation now compares the same stored state with fast `W`
   present and zeroed, retaining all convolution state; overwrite episodes require the latest value.
4. One scalar CEM center averaged incompatible specialists. Objective niches, full vector logging, and
   independent archives made every candidate reproducible. Seven separate CEM islands then kept score,
   recall, causal-memory, overwrite, and sequence distributions apart at unchanged population cost.
5. A fixed GPU validation seed made generations comparable, but repeated selection eventually overfit it.
   Search stopped after six generations without any archive improvement and all archives were audited on
   three unseen seeds, a larger model, and a process restart.

Best fixed-validation evidence looked promising but did not survive the audit:

| archived objective | fixed validation recall 1/3 | overwrite | sequence improvement | notable signal |
|---|---:|---:|---:|---|
| score / memory | 0 / 0 | 0 | 28.9% | causal CE advantage +0.335 |
| recall-1 | 16.7% / 4.2% | 16.7% | 20.6% | best balanced fixed-seed genome |
| recall-total | 12.5% / 8.3% | 0 | 20.1% | combined recall 20.8% |
| overwrite | 16.7% / 0 | 33.3% | 7.5% | update specialist |

| independent 3-seed audit | mean sequence improvement | recall 1/3 | restart | scale retention | speed vs Adam |
|---|---:|---:|---:|---:|---:|
| score genome | 28.2% | 0 / 0 | 12.5% | 1.40× | 0.51× |
| recall-1 genome | 24.7% | 1.4% / 0 | 12.5% | 1.39× | 0.46× |
| recall-total genome | 24.1% | 1.4% / 1.4% | 12.5% | 1.20× | 0.49× |
| overwrite genome | 4.8% | 0 / 1.4% | 0 | 2.13× | 0.43× |

**Gate result:** scale transfer passed for every archive. Sequence failed its strict all-three-seeds gate
despite good means (the score genome's per-seed improvements were 16.3%, 41.5%, and 26.7%). Recall and
restart failed decisively. No mechanism passed, the assistant test was not run, and the 1000× claim is
false. The current Python local learner is 0.37–0.51× Adam, before any fused-kernel optimization.

**Mechanistic conclusion:** the local law robustly discovers useful one-pass sequence adaptation and
transfers unchanged to a model twice as wide and deep. A single scalar eligibility trace does not robustly
bind arbitrary key/value roles; fixed-seed recall and memory advantage were selection overfit. More search
over the same law is not justified.

**Recommended next experiment — Event Algebra, not more tokens:** use a procedural training format with
typed `STORE(key,value)`, `QUERY(key)`, `OVERWRITE(key,value)`, `DELETE(key)`, and `DISTRACTOR(x)` events.
Keys and values are newly permuted for every episode; collision, order, negative-query, and repeated-update
cases are generated independently. Evolve a structured three-factor rule in which query-key similarity
gates a key⊗value eligibility write and overwrite emits an explicit anti-Hebbian erase. First prove storage
with a direct state decoder, then prove output decoding, then the paired fast-weight ablation. Selection
should use the median and worst case across a rotating procedural seed bank; a sealed 16+ seed audit is
never exposed during search. Only after this event-level law generalizes should a learned text-to-event
compiler connect it to natural language. This cleanly separates memory plasticity from parsing and removes
the fixed-validation loophole found here.
