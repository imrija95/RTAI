# Efficiency tournament on a 4 GB GPU

This is a wall-clock-bounded falsification screen for five ways to reduce training cost. It does
not claim that a small run is an 8B model. The same architecture and deployment protocol are used;
only width, runtime, and batch size are reduced.

## Result (2026-07-15)

The full run completed and no arm passed its promotion gate. The measured table, concise failure
analysis, limitations, and next tests are recorded in [`EXPERIMENTS.md`](EXPERIMENTS.md); a sanitized
decision summary is in [`results/efficiency-tournament.json`](results/efficiency-tournament.json).

## Run

First verify orchestration without consuming GPU time:

```bash
uv run python -m fractal.tests.test_efficiency
uv run python -m fractal.exp_efficiency_tournament --smoke --device cpu \
  --out_prefix /tmp/fractal_ckpt_eff --results /tmp/fractal_efficiency_results.json
```

Then run the four-hour tournament:

```bash
uv run python -m fractal.exp_efficiency_tournament \
  --budget_minutes 240 --bf16 --tf32
```

If a driver, power, or experimental-code failure interrupts the run, retain completed arms and
continue only the missing schedule entries:

```bash
uv run python -m fractal.exp_efficiency_tournament --resume --bf16 --tf32
```

The preflight starts with batch 8 and effective batch 16. If the largest soft-MoE control does not
fit, every arm is switched to batch 4 with accumulation 4. If that also fails, no arm starts.

To watch real sampled gradients without adding a training pass:

```bash
VIZ_ATTACH=fractal_ckpt_eff_event.tele.json uv run python -m fractal.viz_serve
```

Run one arm for diagnosis with an explicit duration:

```bash
uv run python -m fractal.exp_efficiency_tournament \
  --arm event --minutes 10 --bf16 --tf32
```

## Arms

| arm | mechanism | scheduled training time |
| --- | --- | ---: |
| `baseline` | fixed depth, full backward | 25 min |
| `genome` | sampled recurrent depth 2/4/8 | 30 min |
| `moe_soft` | four-expert all-active speed control | 5 min |
| `moe_top1` | true top-1 token dispatch | 25 min |
| `event` | one causal global event per completed four-token patch | 35 min |
| `local_credit` | one local recurrence; full backward every eighth step | 30 min |
| `compiler` | verified typed actions with a fourfold tool-call loss | 30 min |

The remaining hour is reserved for preflight, fixed evaluations, process-restart recall, and
runtime margin. The runner does not label an unmeasured compiler or kernel experiment as evidence.

## Decision contract

`fractal_efficiency_results.json` records stored and active parameters, wall time, throughput, peak
VRAM, masked validation loss, held-out one/three-fact recall, free-generated tool validity and
execution, and recall after loading state in a new process. The runner applies the thresholds in
code and promotes at most two arms. With one seed, a pass is only permission for the 12 GB demo; a
failure is an honest small-scale negative result.

Measured results in `docs/EXPERIMENTS.md` must come from the real GPU run, never a smoke test.
