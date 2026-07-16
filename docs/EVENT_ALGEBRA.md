# Predictive Event Algebra

Predictive Event Algebra adds delayed credit to the existing self-modifying fast-weight rule. It
does not replace the delta memory, tied recursion, raw-state values, or constant-size runtime state.
The implementation is experimental and has not yet passed the causal-transfer promotion gates.

Externally observed user spans and tool results receive a small autonomous surprise-driven credit.
Generated tokens still update working `W`, but do not certify themselves as evidence; they require
a later outcome or explicit rating before delayed consolidation.

## Memory levels

- `W` changes for every observed and generated token and remains the immediate session memory.
- `E` is an O(1) eligibility matrix with the same shape as each `W`. It accumulates causal writes
  that can receive delayed predictive or explicit credit.
- `W0` is the learned initial fast weight. Bounded consolidation into `W0` makes rated information
  available to a fresh session without modifying the base checkpoint.
- General shared weights `theta` remain optimizer-trained. A single rating never changes them;
  structural consolidation requires repeated evidence across distinct situations and a separate
  anchor-loss gate.

The implemented trace is:

```text
E <- lambda * E + beta * (v - Wk) outer k
W <- decay * W + beta * (v - Wk) outer k
W <- W + learning_rate * delayed_credit * E
```

The ordinary write remains unchanged. Event Algebra uses the recurrent kernel only when eligibility
is enabled so the trace is invariant to how a stream is chunked.

## Explicit ratings

The UI scale measures retention strength for both user and assistant messages:

| Rating | Credit | Meaning |
| ---: | ---: | --- |
| 1 | -1.0 | Actively weaken the association |
| 2 | -0.5 | Weaken it |
| 3 | 0.0 | Keep it as working context only |
| 4 | +0.5 | Consolidate it |
| 5 | +1.0 | Strongly consolidate it |

Changing a rating applies only the difference from the previous credit. Each revision is
idempotent. User-authority feedback is distinct from any future model self-assessment.

## Live dashboard

Start a tokenizer-compatible checkpoint in read/chat mode:

```bash
FRACTAL_CKPT=MODEL.pt \
VIZ_TOKENIZER=TOKENIZER.json \
VIZ_CHAT=1 \
VIZ_FEEDBACK=1 \
uv run python -m fractal.viz_serve
```

The dashboard exposes a persistent chat panel with rating buttons on user and assistant messages.
The model receives no replayed conversation window: UI history is a private audit log, while model
context remains in `W/E`. Runtime state, session metadata, feedback overlays, and the trainer queue
are stored separately. A W0 rollback overlay is written before every consolidation revision.

The HTTP interface is:

- `GET /api/session`
- `POST /api/chat` with `{"message": "..."}`
- `POST /api/feedback` with `{"message_id": "...", "rating": 1..5, "revision": N}`

The server binds to loopback by default. `VIZ_AUTH_TOKEN` is mandatory when `HOST` is not loopback;
open the UI once as `http://HOST:PORT/?token=TOKEN`. The token moves into session storage and is
removed from the visible URL before authenticated API polling begins.

## Training integration

Eligibility can be enabled during ordinary training with `--event_algebra`. A running trainer can
consume the live UI queue at safe batch boundaries:

```bash
uv run python -m fractal.train \
  --event_algebra \
  --feedback_queue fractal_feedback.jsonl \
  --out MODEL.pt \
  ...
```

Queue events are consolidated into `W0`. The W0 overlay and consumed event IDs are atomically
coupled in `MODEL.pt.feedback-state.pt`, providing exactly-once replay after a process restart.
Feedback processing performs an extra forward only when a new rating exists; it is absent from the
ordinary training hot path.

## Bounded mechanistic screen

Run the predeclared screen with ten-minute atomic reports:

```bash
uv run python -m fractal.exp_event_algebra \
  --ckpt MODEL.pt \
  --tokenizer TOKENIZER.json \
  --budget_minutes 90 \
  --report_minutes 10 \
  --results event_algebra_results.json
```

The screen compares the unchanged delta memory, surprise-driven credit, explicit rating-5 credit,
and W0 consolidation on identical held-out associative episodes. This is a mechanism test, not an
assistant-understanding claim. Natural-language causal transfer, contradiction, intervention, and
safe structural consolidation remain subsequent gates.

## Vector teaching screen

The first matched screen falsified the simplest scalar-credit hypothesis: increasing eligibility
credit changed update norms but produced no accuracy lift, while the strongest setting degraded
recall. Directly averaging the trace into `W0` produced no correct fresh-session answers.

An experimental vector-teaching arm therefore uses a confirmed next-token prediction error to
supply the update direction while the rating remains the authority gate. Only `W0` is trainable;
general parameters are frozen, updates are norm bounded, and every experimental episode restores a
snapshot afterward. The runner also measures answer probability, rank, held-out prompt hijacking,
and tensor-only overlay reload in a fresh process.

Unconstrained vector teaching can reach perfect fresh-session top-1 on a small screen, but a
counterfactual audit showed that most of the gain was a global taught-token bias. Restricting the
update to permanent scales and adding a held-out rank-margin anchor substantially improved
specificity.

A 126M-parameter confirmation used margin weight 5 across two independent seeds. The aggregate
held-out result was 49/62 fresh-session answers correct (79.0%) with 49/496 control prompts hijacked
(9.88%); both tensor-only W0 overlays survived a fresh process. This passes the predeclared
single-memory synthetic gates.

Sequential consolidation remains unsolved. After eight writes, unprotected W0 retained 25% of
associations. A fixed-capacity KL reservoir improved retention to 62.5% but raised learned-answer
hijacking to 56.25%; protecting every learned answer did not recover specificity. The current
mechanism therefore supports one durable prompt-specific association, not safe continual memory.
See `docs/results/event-algebra-dl580-2026-07-16.json` for the measurements and remaining gates. No
assistant-understanding claim follows from these synthetic results.

## Detached bounded window

The window launcher writes its PID and an atomic status before returning. Startup succeeds only
after the wrapper verifies that the detached process is alive and reports `starting` or `running`.
The managed Compose service is restored from an EXIT/INT/TERM trap.

```bash
scripts/start-event-algebra-window.sh \
  /srv/rtai /srv/rtai/.venv/bin/python 2026-07-17T07:20:00+02:00 \
  /srv/llama-swap vector-margin-confirm -- \
  --ckpt fractal_ckpt_eff_baseline.pt \
  --tokenizer fractal_tokenizer_32k.json \
  --val_bin fractal_data/val.bin \
  --distance 0 --n_facts 1 \
  --teacher_lr 0.3 --teacher_steps 32 --teacher_max_fraction 0.5 \
  --teacher_scope permanent \
  --teacher_anchor_mode margin --teacher_anchor_weight 3 \
  --teacher_anchors 8 --teacher_controls 8
```

Use a unique result name for every run. The launcher reserves five minutes before the supplied end
time for final reports and service restoration; the reserve and telemetry interval are configurable
through `EVENT_WINDOW_RESERVE_SECONDS`, `EVENT_WINDOW_REPORT_MINUTES`, and
`EVENT_WINDOW_POLL_SECONDS`.
