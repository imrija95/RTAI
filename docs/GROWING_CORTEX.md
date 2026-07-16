# Growing Cortex

Growing Cortex is an append-only continual-skill experiment for the fractal model. It preserves the
existing recurrent rule, fast-weight memory, and fixed active compute while allowing persistent
low-rank skill hemispheres to be added during training or operation.

This is an experimental mechanism. It does not yet establish natural-language understanding or
general agent skill acquisition.

## Architecture

One `GrowingCortex` module is shared by every recurrent unrolling. The ordinary block remains:

```text
memory read + shared MLP + optional selected skill residual
```

Each stored skill owns:

- a stable content address,
- a low-rank down/up residual,
- a lifecycle state,
- confidence and usage telemetry,
- an optional parent identifier.

The compiler maps an ordered task specification to a differentiable candidate address and residual.
Read routing compares a stable task address against the stored addresses. Skill identity is a
control-plane signal and is deliberately separated from variable execution arguments and tool
observations. At most one stored expert is active for a sample or sticky agent trajectory.

## Function-preserving growth

New experts are appended in the `candidate` state and are excluded from routing. Appending a
candidate therefore changes active logits by exactly zero. A candidate can be tested in a shadow
context, committed as `juvenile` or `mature`, quarantined, made dormant, or pruned without reshaping
the router or changing existing expert tensors.

This avoids the destructive mechanism observed in the earlier timescale-neurogenesis experiment:

- no fixed-output gate is resized,
- no existing routing row moves,
- no optimizer state is reset,
- no existing memory timescale changes meaning.

Checkpoint persistence stores a small structural manifest before loading the tensor state. Runtime
fast-weight state remains separate and O(1) in sequence length.

## Two-stage proceduralization curriculum

The falsification harness deliberately separates two learning problems:

1. **Interpreter:** learn to execute freshly permuted specifications while the specification is
   present in the input.
2. **Compiler:** freeze the interpreter and learn to turn a specification into a low-rank residual
   that executes after the specification is removed.

Opaque skill names are reassigned to new procedures every episode. This prevents the compiler from
passing by memorizing a name-to-answer shortcut. The compiler is trained with exact execution loss,
teacher distillation from the interpreter, and a contrastive task-address loss.

The sealed evaluation then:

- compiles held-out procedures,
- appends and commits them sequentially,
- tests all earlier skills after every birth,
- measures unseen-task routing hijack,
- compares against one globally overwritten expert,
- reloads the grown expert tree in a fresh process.

## Results

The initial 128-wide, depth-4 pilot produced:

- interpreter exact accuracy: 25.5%,
- compiled held-out accuracy: 22.4%,
- no-skill baseline: 6.8%,
- one globally overwritten expert after twelve skills: 7.8%,
- append-only sequential accuracy after twelve skills: 22.4%,
- maximum logit change from candidate birth: exactly 0,
- fresh-process accuracy: 22.4%.

That pilot showed a real procedural-transfer signal and safe structural persistence, but failed the
declared capability gates. Routing specificity was also poor: 66.7% of unseen control tasks selected
an existing expert.

A scaled 256-wide, depth-6 run first reproduced a misleading failure: compiled skills reached 75.8%,
but the sequential bank fell to chance. The experts themselves were unchanged. The router had been
trained on an average over many calls while deployment used one call, and variable arguments were
pooled into the skill identity.

Separating the stable address from the execution payload, then applying 2,000 address-repair steps,
produced:

- interpreter exact accuracy: 76.17%,
- compiled held-out accuracy: 76.17%,
- no-skill baseline: 2.3%,
- one globally overwritten expert after sixteen skills: 8.6%,
- append-only sequential accuracy after sixteen skills: 76.17%,
- oldest-skill accuracy: 56.2%,
- known-address routing accuracy: 100%,
- unseen-address hijack: 0%,
- maximum logit change from candidate birth: exactly 0,
- fresh-process accuracy: 76.17%.

All mechanism-specific gates passed. The only declared miss was interpreter accuracy at 76.2% versus
the 80% target. The result validates addressed append-only procedural memory on the synthetic task.
It does not establish natural-language understanding, autonomous curriculum discovery, or arbitrary
agent-tool skill acquisition. Sanitized measurements are in
[`results/growing-cortex-addressed.json`](results/growing-cortex-addressed.json).

The final shorthand result is therefore **76.17% exact / 0% unseen-address hijack / 76.17% after a
fresh-process restart**. Before the address/payload separation, stored expert tensors were already
intact; the failure was entirely in the mismatched routing control plane.

## Run

Quick smoke:

```bash
uv run python -m fractal.exp_growing_cortex \
  --smoke --results /tmp/growing-cortex-smoke.json
```

Single-GPU screen with live dashboard telemetry:

```bash
uv run python -m fractal.exp_growing_cortex \
  --interpreter_steps 8000 --steps 8000 \
  --meta_tasks 8 --queries 16 \
  --n_embd 256 --n_head 8 --depth 6 --n_scales 2 --skill_rank 16 \
  --bf16 --tf32 \
  --telemetry growing-cortex.tele.json \
  --results growing-cortex.json

VIZ_ATTACH=growing-cortex.tele.json uv run python -m fractal.viz_serve
```

The heterogeneous server launcher assigns one GPU experiment and three independent CPU controls to
separate NUMA nodes. It may stop a managed inference service for the experiment and restores it from
an exit trap:

```bash
scripts/start-growing-cortex-dl580.sh \
  /srv/rtai /srv/rtai/.venv/bin/python /srv/inference growing-cortex-v1
```

## Promotion gates

The original experiment contract passes only if all gates hold:

- disabled birth changes logits by exactly zero,
- interpreter exact accuracy is at least 80%,
- one compiled held-out skill reaches at least 60%,
- sixteen sequential skills retain at least 70% mean accuracy,
- the oldest skill remains at least 50% correct,
- unseen-task routing hijack stays below 10%,
- append-only retention beats global overwrite by at least 20 percentage points,
- the grown model retains at least 70% accuracy after a fresh-process reload.

Natural-language paraphrase, exact agent-tool execution, negative tool routing, and slow shared-weight
consolidation remain later gates.
