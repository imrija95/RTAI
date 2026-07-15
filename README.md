# RTAI

**Naturally fractal language models with persistent memory in their own weights.**

RTAI is a from-scratch PyTorch research project exploring models that keep learning during
operation. Its distinctive mechanism is self-modifying fast-weight memory,
`W <- gamma W + beta (v - Wk) k^T`: constant-size state, no growing KV cache, and memory that can
survive a process restart. The work is experimental, measured on modest hardware first, and reports
negative results alongside successes.

> **Hardware and compute sponsorships**
>
> RTAI deliberately proves ideas on modest hardware before scaling them. Donations and time-limited
> access to neuromorphic hardware, AI accelerators, CUDA-capable GPUs, workstations, servers, ECC
> memory, or training-cluster credits can expand the experiments that can be tested honestly.
> Sponsorship does not buy favorable results, endorsements, or roadmap control; failed experiments
> remain part of the record. Organizations interested in supporting the research can open a GitHub
> issue titled **Hardware sponsorship** with the available resource class and access constraints.

## Current evidence

| Model or study | Size | Evidence | Status |
| --- | ---: | --- | --- |
| SRWM symbolic PoC (`rtai/`) | 0.82M | key-value recall persists across restart; ablation falls to zero | Validated in small |
| SRWM TinyStories LM (`rtai/`) | 16.9M | storytelling plus persistent in-weight facts | Validated, limited |
| Fractal LM (`fractal/`) | 108.7M active | weight-tied rule, timescale ladder, FineWeb-Edu pretraining | Partial run, not a finished model |
| Fractal recall study | about 4.6M | two-scale routing was strongest; empty-add neurogenesis failed | Study complete |
| Plasticity genome | small | one-pass sequence adaptation transferred scale, recall did not | Refuted for declared gate |
| Efficiency tournament | small | no candidate passed the promotion contract | Negative result |

The detailed measurements and falsification criteria are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). The non-negotiable design contract is
[`VIBE.md`](VIBE.md).

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Train the small symbolic model, then open its local memory visualization:

```bash
uv run python -m rtai.train
uv run python -m rtai.serve --ckpt ckpt.pt
# http://localhost:8000
```

Teach and query the same model from a terminal; its runtime memory is saved locally:

```bash
uv run python -m rtai.run --ckpt ckpt.pt chat
```

Run the FractalLM dashboard in truthful learn-from-scratch mode:

```bash
VIZ_LEARN=1 uv run python -m fractal.viz_serve
# http://localhost:8000
```

No checkpoint is currently published or stored in Git. Future model releases will use
checksum-verified Safetensors on Hugging Face; see
[`docs/MODEL_RELEASES.md`](docs/MODEL_RELEASES.md).

## Architecture

The repository contains two related lines:

- `rtai/` is the compact SRWM proof of concept and language-model baseline.
- `fractal/` applies one recurrent rule over depth and a ladder of memory timescales, with training,
  agent grammar, evaluation, persistence, and live architecture telemetry.
- `docs/` is the experiment record, including failed approaches and their measured mechanisms.
- `tests/` and `fractal/tests/` cover kernel equivalence, streaming causality, persistence, safe
  checkpoint handling, and learning in small.

The dashboard is part of the architecture contract: it shows actual model geometry, data flow, and
sampled signals from a real run. Telemetry is throttled and never adds a second training pass.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
./scripts/audit-public.sh
```

Contribution requirements are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Security and runtime-memory
privacy are documented in [`SECURITY.md`](SECURITY.md).

## Responsible status

RTAI is research software, not a production assistant or safety boundary. Small checkpoints can be
wrong, brittle, and easy to destabilize. Persistent state may encode information supplied during
operation and must be handled as sensitive local data. Capability claims are limited to the
published test and experiment conditions.

## License and governance

Source code is available under the [Apache License 2.0](LICENSE), including commercial use and an
explicit patent grant. Attribution notices are recorded in [`NOTICE`](NOTICE).
Project stewardship is maintainer-led as described in [`GOVERNANCE.md`](GOVERNANCE.md). Dataset and
future model licenses are reviewed separately from the source-code license.
