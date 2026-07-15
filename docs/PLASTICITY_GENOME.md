# Plasticity Genome experiment

This is a bounded attempt to replace iterative backpropagation with a small, scale-invariant local
learning law. Every candidate starts from identical random slow weights, observes every training
sample once, and changes both the persistent fast state and selected slow weights during an ordinary
streaming forward pass. AdamW is run only as a separately reported control.

## Mechanism

After token `x_t` predicts the observed `x_{t+1}`, the learner constructs the exact output-space
error from the tied embedding. Identity feedback is used where dimensions agree; otherwise a
deterministic hash-seeded Rademacher projection supplies a local error. One rank-one update is
aggregated per block for the embedding/head, q/k, beta and scale routing, memory output projection,
and shared MLP. Layer normalization and the causal convolution remain fixed random structure.

The genome contains bounded, named coefficients for error alignment, Hebbian correlation, Oja
stabilization, decay, clipping, surprise selection, random feedback, and per-scale fast-write
modulation. There is no optimizer and the learning path runs under `torch.no_grad()`. Each sample ID
is accepted once; replay is an error.

## Run

Fast CPU correctness and bounded smoke screen:

```bash
uv run python -m fractal.tests.test_plastic_genome
uv run python -m fractal.exp_plastic_genome --phase smoke \
  --budget_minutes 15 --population 8 --elites 4 \
  --search_device cpu --verify_device cuda \
  --results plastic_genome_results_smoke.json
```

Two-hour heterogeneous search on a host with a 12 GB GPU and four CPU/NUMA workers:

```bash
uv run python -m fractal.exp_plastic_genome --phase search \
  --budget_minutes 120 --population 64 --elites 8 \
  --workers 4 --threads_per_worker 20 \
  --numa auto \
  --search_device cpu --verify_device cuda \
  --results plastic_genome_results.json
```

The JSONL history is append-only. `--resume` restores the search distribution and random-generator
state. The attached dashboard accepts `plastic_genome_results.tele.json`; it labels the measured
signals as explicit local updates rather than gradients.

Fitness is staged: held-out sequence improvement is capped at the 20% gate, after which continuous
held-out recall loss supplies selection pressure. Exact recall accuracy remains the declared gate;
the continuous value only prevents the evolutionary landscape from being completely flat while all
candidates still predict the wrong token. Recall is evaluated causally against a paired ablation:
the same stored prompt and query run with the learned fast `W` and with only `W` zeroed while causal
convolution state is retained. `CE(no-W) - CE(W)` measures memory contribution rather than token
priors. Answer-only learning masks and an evolved eligibility trace carry delayed credit from the
answer back to earlier fact activations without BPTT. One quarter of episodes overwrite a key with a
new value and require the latest value, preventing a static key prior from passing.

The search keeps separate elite niches for scalar fitness, one- and three-fact exact recall,
causal memory advantage, overwrite accuracy, and sequence improvement. CPU candidates share a seed
within each generation, while niche elites are re-evaluated on one fixed GPU validation seed. This
makes global-best comparisons independent of generation difficulty and prevents an early recall
mechanism from being averaged away merely because another candidate has smoother auxiliary loss.
Every candidate vector is written to generation history. The best fixed-validation genome for each
elite objective is also checkpointed independently, so a specialist remains reproducible even when
later recombination lowers its score.

Island search goes one step further: each of the seven objectives owns a separate antithetic CEM
distribution. Only the best candidate from each island is sent to the shared fixed-seed GPU
validation. This avoids repeatedly averaging incompatible recall, overwrite, and sequence
specialists into one mediocre center while keeping total population and verification cost fixed.

## Predeclared gates

1. **Sequence:** at least 20% held-out loss reduction versus the random delta-only model on every one
   of three seeds.
2. **Recall:** at least 50% one-fact, 20% three-fact, and 50% process-restart recall on held-out values.
3. **Scale transfer:** the unchanged genome at twice the width and depth retains at least 80% of the
   small model's loss improvement.

Passing all three proves only the local-learning mechanism in small. Assistant learning remains
scale-gated until ordinary-language loss, exact tool execution, and restart recall pass together.
A 1000x claim additionally requires Adam-level quality at at least 1000x measured learning
throughput. A failed bounded search is recorded as a negative result and is not extended silently.

## Measured outcome

The bounded search found robust one-pass sequence adaptation and scale transfer, but no general
associative recall. Fixed-validation specialists reached 16.7% one-fact recall, 8.3% three-fact recall,
and 33.3% overwrite accuracy; a three-seed audit reduced these to noise-level recall and zero overwrite.
All predeclared mechanism gates therefore did not pass, and no 1000x claim is made. The complete result,
including the adaptive test corrections and the proposed Event Algebra successor, is recorded in
`docs/EXPERIMENTS.md`. A sanitized machine-readable audit summary is available in
[`results/plasticity-genome.json`](results/plasticity-genome.json).
