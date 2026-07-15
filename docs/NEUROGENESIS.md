# Neurogenesis in the fractal — design note (experiment for later)

> Branch `feat/neurogenesis`. Run it **only after the 109M pretraining finishes** and first prove it
> in the small. Goal: find out whether growing the shape at runtime really helps — especially multi-fact
> recall (key collapse), which is our long-standing unsolved problem.

## Hypothesis

Key collapse with multiple facts = a failure of **pattern separation** (facts overwrite each other in memory,
the model doesn't separate them by time/context). In the hippocampus the evolutionary answer to exactly this
problem is **neurogenesis in the dentate gyrus** — a new, hyperplastic neuron separates "now" from
"before". The fractal is a memory system (fast weights W = episodic memory, the γ ladder = a spectrum
of temporal contexts), so the analogy fits where it should. **Testable prediction:** a model
that is allowed to *add a timescale* on saturation will handle more facts than an equally sized model
with a static ladder.

## Four axes of "learned shape" (from discussion)

1. **Unrolling depth** = how many steps the model thinks (adaptive computation / halting). Already today
   a runtime knob (`depth` in `forward`). The cheapest first step, doesn't grow parameters.
2. **Number of scales (neurogenesis)** ← *this experiment*. A new branch = a new timescale.
3. **Ladder ratio ρ** = the fractal dimension of time (how densely we cover the time axis).
4. **Layer differentiation** = stem cell → organ (continuous `untie`: layers start as
   copies of the rule, diverge only where the data pushes).

## Growth primitive (minimal, one mechanism)

The gradient doesn't flow through the shape (discrete) → growth is a **rule outside the gradient**, read from telemetry
that we ALREADY measure (see `viz_serve.activity`):

- **Trigger (when to grow):** the read gate persistently concentrates >X% of the mass into one scale
  **and** its ‖W‖ saturates (overloaded memory). Both signals are in the HUD.
- **Action (how):** a new scale sprouts with an **empty W** (no memories), **high plasticity**
  (higher β / lower γ at the start), τ chosen so as to fill a gap in the ladder (typically
  slower than the slowest saturated one).
- **Maturation:** the newborn scale's β decreases over time, γ settles → from a "greedy cub" to an adult scale.
- **Pruning (the opposite):** a scale that is persistently not read from (gate ~0) and whose ‖W‖ doesn't converge
  withers away (cortical pruning, not hippocampal neurogenesis).

## Regionalization — "centers with different properties" (a question that came up)

The brain is NOT homogeneous — it is regionalized (cortex/hippocampus/cerebellum: different cell types,
plasticities, timescales), but it shares a basic neural mechanism. Our fractal is deliberately
homogeneous today (one rule). The tension is **elegance/sample efficiency (homogeneous) ↔ capacity/division
of labor (specialized centers)**. A disciplined way to have centers without hand design:

- **A center = a band (depth × scale) with its own priors** (τ range, plasticity, growth threshold).
  Regionalization then *emerges* from one growth rule + different per-band priors, instead of
  being hard-designed. (One genome, a phenotype differentiated by position.)
- **Arbitrary generation of connections (synaptogenesis):** today the topology is fixed (scales read/write
  independently, the `gate` connects them). "Free connections" = letting **the gate itself grow** — new paths
  between scales/layers (cross-scale routing), i.e. a graph that grows, not just the ladder. The most radical and
  least stable (à la NEAT); the `gate` is exactly the right place to try this.

**Order of boldness:** (a) neurogenesis of a single scale → (b) regionalization via per-band priors →
(c) a growing gate routing. Each step only after proving the previous one.

## Experiment (small, honest)

- Small model (n_embd ~256, depth ~4), `RecallGen` multi-fact (`fractal/recall.py`, `n_facts=3`).
- **A/B:** static ladder (n_scales=3) vs. neurogenesis (start with 2 scales, grows on saturation) —
  matched compute/params at the end.
- **Metric:** held-out multi-fact recall (`rg.accuracy(..., n_facts=3)`), + track when/how many scales
  grew and whether the reading spread out.
- **Falsification:** if neurogenesis does not improve recall beyond "just more scales from the start",
  the hypothesis falls — and that too is a result.

## An honest warning

- Discrete shape → the gradient doesn't flow; continuous relaxation (learned weights = our gate, passive for now)
  vs. growth rules outside the gradient. Don't mistake "a nice metaphor" for a mechanism — that's why the A/B.
- Changing the shape at runtime **destabilizes** what was learned (a new scale changes the statistics that the
  projections are calibrated to) → growth must be rare and cautious, as in nature.
- Too many degrees of freedom = untrainable. Keep **one** growth primitive until it proves itself.

## Implementation state (DONE — 2026-07-03)

Mechanism and policy written and tested; A/B not yet run (waiting for `run_neuro_ab.sh`).

**Mechanism** (`unit.grow_scale`, `cell.py`): growth = extending the geometric ladder
by a rung, the permanent (γ=1) store stays on top → the newborn IS the new empty permanent
store (γ=1, W0=0) with high plasticity (`beta_gain`, β_eff = 1−(1−β)^gain, bounded in (0,1)),
the old permanent is "demoted" to a long finite one. This keeps the invariant
`gammas == _make_gammas(n_scales)` → the checkpoint reassembles with the right ladder after reload
WITHOUT storing γ. `beta_gain` is persistent=False (doesn't break old checkpoints). The gate (`gate`)
and `to_f` are re-interleaved on growth (gate is scale-inner, to_f scale-major). Verified:
forward/backward after growth, streaming, persistence round-trip bit-identical, numerical equivalence
of kernels (`beta_gain==1` = no change).

**Policy** (`fractal/neurogenesis.py`, `NeurogenesisController`): trigger = gate concentration
> `conc_thresh` AND the dominant scale's ‖W‖ saturating (plateau: |EMA_fast − EMA_slow| small),
with `warmup`/`cooldown` (growth rare). Maturation: `beta_gain` → 1.0 with constant `mature_steps`.
Separation of mechanism/policy → the same controller drives training and the live dashboard.

**Training** (`train.py --neurogenesis`): telemetry after every step (gate share from
`unit._last_share`, ‖W‖ from recall states), at a birth the optimizer is rebuilt (new/re-interleaved
parameters; Adam moments reset — growth is rare, acceptable). Births + the final ladder
are logged and saved to `.resume`.

**A/B run** (an honest control): static-2 vs static-4 (same final capacity FROM THE START = the key
control) vs neurogenesis (2→≤4). The original one-off orchestration script was not retained. Metric:
`recall 3-facts HELD-OUT`. **Falsification holds:** if grow doesn't beat static-4, the hypothesis falls.

**Dashboard live** (`viz_serve.py` with `VIZ_NEUROGENESIS=1`): the controller runs on the inference
stream → on saturation a new scale sprouts right during reading; the fractal gets a new branch on
every node (tree rebuild), a birth bloom pulse + banner. Tuned for a small model (a big one would
grow unbearably dense at L^depth). Verified with an HTTP client: a birth lands, payload consistent,
the ladder invariant holds live.
```
FRACTAL_CKPT=<small_2-scale_ckpt> VIZ_NEUROGENESIS=1 VIZ_MAX_SCALES=5 PORT=8008 \
    CUDA_VISIBLE_DEVICES= uv run python -m fractal.viz_serve
```

## Lessons from the A/B (2026-07-03) — hypothesis REFUTED (in this implementation)

Two rounds of A/B (details + tables in `EXPERIMENTS.md`):
- **v1 (easy task):** everything at the ceiling → inconclusive. Grow stopped itself at 3 (on-demand growth — nice).
- **v2 (hard: 256 names, ≤8 facts):** real headroom appeared (static-2: 6/8-fact 67/61%). And then:
  - **static-4 < static-2** — more fixed scales + `gate_lambda` load-balancing WORSENS recall
    (forces reading from the fast γ<1 scales that don't hold facts). An independent finding about the base architecture.
  - **grow ≪ both** — neurogenesis BROKE multi-fact recall (6-fact 2%, 8-fact 0%; 1-fact survived).

**Why growth hurts (mechanism):** a birth (a) DEMOTES the permanent γ=1 scale — where the
multi-fact binding was held — to γ<1 (it starts to leak), (b) adds an EMPTY permanent one, (c) RESETS the optimizer,
(d) changes the ladder that the projections are calibrated to. The second birth @ iter 2923 (late) → too few iterations
to recover under a decaying cosine LR. Exactly the destabilization that this note warned about.

**Possible rescue of the concept (untested) — if we want to separate the CONCEPT from this implementation:**
1. **Don't demote the permanent scale** — add a new scale WITHOUT changing existing γ (breaks the invariant
   `gammas==_make_gammas(n)` → γ must then be persisted per cell, but it preserves the learned memory).
2. **Don't reset the optimizer** — surgically add a param-group only for the new parameters, keep the Adam state
   of the old ones.
3. **Grow only EARLY** (all births in the first ~15% of training) → enough iterations to recover.
4. **Turn off/lower `gate_lambda`** on the recall task — let the gate be able to IGNORE the surplus scales
   (concentrate on the permanent one), instead of being forced to read from them.
5. Gentler birth: briefly freeze the old scales / a local LR-warmup for the new scale.

But falsification holds: **as growth is built now, it actively hurts.** A clean negative result.

### v3 — the GENTLE-growth rescue also FAILED (definitive verdict)
Implemented: `grow_scale(demote=False)` (don't demote the permanent one), `opt.add_param_group` instead of a reset
(`--grow_no_demote`), births only early (`--grow_until`). All 3 variants `gate_lambda 0`.
**Grow collapsed again** (3-fact 2%, 6/8-fact 0%, worse than static). Moreover `gate_lambda 0` revealed that
the gate collapses onto the FAST scale without balancing (→ correcting the hasty v2 conclusion "λ hurts").

**Why even gentle growth fails:** one piece of destruction can't be bypassed — **a birth necessarily reshapes the GATE** (new Linear,
reset of its Adam state, zero logits for the new scales). And **the gate = the bottleneck** (routing to the permanent
scale decides recall). So growth necessarily destroys exactly the component everything rests on.

**CONCLUSION OF THE WHOLE EXPERIMENT:** Multi-fact recall is NOT limited by capacity (the number of scales) → neurogenesis
has nothing to solve and hurts via a routing-breaking birth. The real lever = **a better gate/routing** (mild λ;
a "recall → permanent scale" prior; or a better gate architecture), NOT adding scales. Neurogenesis as a
direction for THIS model is closed (an honest negative result with a mechanism). The code stays (functional, tested)
for possible future use on a task where capacity WOULD really be the bottleneck.

## Critical files (when we get to it)

- `fractal/unit.py` — `FractalState.W` (list per scale), `gate`, `gammas`; add a dynamic
  number of scales + per-band priors here.
- `fractal/cell.py` — `delta_chunk_decay` per scale (β, γ) → per-scale plasticity/maturation.
- `fractal/recall.py` — already does multi-fact eval (`accuracy(..., n_facts=3)`).
- `fractal/viz_serve.py` — trigger telemetry (gate, ‖W‖) already done; growth visualization a bonus.
