# RTAI — instructions for agents (Claude and any subagents)

This project is **vibecoded**: a from-scratch, naturally fractal, self-modifying language
model with persistent in-weights memory. The project's direction and non-negotiable
invariants live in **`VIBE.md`**.

## ⚠️ Rule #1 — VIBE.md
- `VIBE.md` holds the **facts and core requirements** of the project (what the model IS and
  what must not break during a rabbit hole). It is a contract about direction, not a scratchpad.
- **Edit `VIBE.md` ONLY with the user's explicit consent.** Never change it in passing. Propose
  the change in chat, wait for approval, then write.
- **Before any larger architectural change or when going deep (a rabbit hole)**, read `VIBE.md`
  and check the proposal breaks none of its invariants. If it would, stop and ask.

## ✍️ Writing rules for code, comments, README, docs
- **English only.** All comments, docstrings, log/print messages, help text, UI strings,
  README and docs are written in English. (The user may chat in Czech; the repo stays English.)
- **No personal information.** Comments, README, docs, commit messages, etc. must not contain
  personal details about the user or their setup. Keep everything generic and impersonal:
  - No "your"/"the user's" framing.
  - Describe hardware generically — e.g. "a laptop-class GPU (e.g. RTX A2000 4GB)", not
    "your RTX A2000" or "the user's cluster".
  - No personal names, emails, or private paths.
  - The copyright holder in `NOTICE` is the only explicitly approved exception.

## Orientation
- `README.md` — map of the models and their states.
- `docs/EXPERIMENTS.md`, `docs/NEUROGENESIS.md` — experiment history and rationale.
- `fractal/` — the active model package (unit/cell/model/train/agent/grammar/viz_serve …).
- Dashboard: `uv run python -m fractal.viz_serve` (:8000); modes `VIZ_LEARN=1`, `VIZ_ATTACH=file`.

## Working style
- **Prove-in-small**: every claim must be shown on modest hardware (a single laptop GPU / CPU)
  before allocating the cluster. Negative results are reported honestly.
- Commit / push only when the user asks.
