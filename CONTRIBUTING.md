# Contributing

RTAI is an experiment-led research project. Contributions are welcome when they preserve the
invariants in [`VIBE.md`](VIBE.md) and keep claims falsifiable.

## Before opening a pull request

1. Describe the hypothesis and the cheapest falsification criterion.
2. Prove the change on CPU or a modest single GPU before proposing larger runs.
3. Run `uv sync --group dev`, `uv run pytest`, `uv run ruff check .`, and
   `./scripts/audit-public.sh`.
4. Keep code, documentation, UI text, and commit messages in English.
5. Do not include checkpoints, datasets, runtime memory, credentials, machine paths, or personal
   information.

Negative experimental results are useful and should be reported honestly. Architecture changes
that conflict with `VIBE.md` will not be accepted without an explicit project decision.
