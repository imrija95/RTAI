#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 ROOT RUN_DIR SERVICE_DIR" >&2
  exit 2
fi

ROOT=$1
RUN_DIR=$2
SERVICE_DIR=$3
SERVICE_NAME=${NATURAL_CORTEX_SERVICE_NAME:-llama-swap}
PYTHON="$ROOT/.venv/bin/python"
TOKENIZER="$RUN_DIR/natural_tokenizer_24k.json"
DATA_DIR="$RUN_DIR/natural_data_240m"
AB_DIR="$RUN_DIR/natural_ab_10m"
MAIN_DIR="$RUN_DIR/natural_main"
STATUS="$RUN_DIR/status.json"
SERVICE_LOG="$RUN_DIR/inference-service.log"
SERVICE_WAS_RUNNING=0
CHILD_PID=""

mkdir -p "$RUN_DIR"

write_status() {
  local stage=$1
  local code=${2:-0}
  local temporary="${STATUS}.tmp"
  printf '{"schema_version":1,"stage":"%s","pid":%s,"exit_code":%s,"updated_epoch":%s}\n' \
    "$stage" "$$" "$code" "$(date +%s)" >"$temporary"
  mv "$temporary" "$STATUS"
}

restore_service() {
  if [ "$SERVICE_WAS_RUNNING" -eq 1 ]; then
    docker compose --project-directory "$SERVICE_DIR" up -d >>"$SERVICE_LOG" 2>&1
    SERVICE_WAS_RUNNING=0
  fi
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
  fi
  restore_service || code=1
  write_status "finished" "$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

run_child() {
  local code=0
  "$@" &
  CHILD_PID=$!
  wait "$CHILD_PID" || code=$?
  CHILD_PID=""
  return "$code"
}

write_status "starting"
cd "$ROOT"

if [ ! -s "$TOKENIZER" ]; then
  write_status "tokenizer"
  run_child env HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
    "$PYTHON" -m fractal.natural_data tokenizer \
    --out "$TOKENIZER" --vocab-size 24000 --max-chars 120000000 --seed 20260716 \
    >"$RUN_DIR/tokenizer.log" 2>&1
fi

if [ ! -s "$DATA_DIR/manifest.json" ]; then
  if [ -e "$DATA_DIR" ]; then
    echo "Incomplete data directory exists: $DATA_DIR" >&2
    exit 1
  fi
  write_status "data"
  run_child env HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30 \
    "$PYTHON" -m fractal.natural_data build \
    --tokenizer "$TOKENIZER" --out-dir "$DATA_DIR" \
    --train-tokens 240000000 --shard-tokens 5000000 \
    --val-permille 10 --seed 20260716 >"$RUN_DIR/data.log" 2>&1
fi

write_status "verify-data"
run_child "$PYTHON" -m fractal.natural_data verify \
  --tokenizer "$TOKENIZER" --data-dir "$DATA_DIR" >"$RUN_DIR/data-verify.json"

if docker ps --format '{{.Names}}' | grep -Fxq "$SERVICE_NAME"; then
  SERVICE_WAS_RUNNING=1
  write_status "stopping-inference-service"
  docker compose --project-directory "$SERVICE_DIR" stop >>"$SERVICE_LOG" 2>&1
fi

write_status "preflight"
run_child "$PYTHON" -m fractal.natural_preflight \
  --steps 5 --seq-len 512 --batch 8 --output "$RUN_DIR/preflight.json" \
  >"$RUN_DIR/preflight.log" 2>&1

GRADIENT_CHECKPOINTING=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gradient_checkpointing"]["selection"])' \
  "$RUN_DIR/preflight.json"
)
if [ "$GRADIENT_CHECKPOINTING" = "enabled" ]; then
  CKPT_ARGS=(--grad-ckpt)
elif [ "$GRADIENT_CHECKPOINTING" = "disabled" ]; then
  CKPT_ARGS=()
else
  echo "Production GPU preflight did not make a checkpointing decision" >&2
  exit 1
fi

if [ ! -s "$AB_DIR/report.json" ]; then
  write_status "dense-moe-ab"
  run_child "$PYTHON" -m fractal.natural_ab \
    --tokenizer "$TOKENIZER" --data-dir "$DATA_DIR" --out-dir "$AB_DIR" \
    --tokens 10000000 --max-gpu-hours 2 \
    --batch 8 --accum 1 --seq-len 512 --val-batch 1 --val-batches 16 \
    --bf16 --tf32 "${CKPT_ARGS[@]}" >"$RUN_DIR/ab.log" 2>&1
fi

readarray -t SELECTION < <(
  "$PYTHON" -c '
import json,sys
report=json.load(open(sys.argv[1]))
print(report["selected"])
print(report["continuation_checkpoint"])
print(report["continuation_state"])
' "$AB_DIR/report.json"
)
VARIANT=${SELECTION[0]}
INITIAL_CHECKPOINT="$AB_DIR/${SELECTION[1]}"
INITIAL_STATE="$AB_DIR/${SELECTION[2]}"

write_status "main-${VARIANT}"
MAIN_ARGS=(
  --variant "$VARIANT"
  --tokenizer "$TOKENIZER"
  --data-dir "$DATA_DIR"
  --out-dir "$MAIN_DIR"
  --max-tokens 240000000
  --max-gpu-hours 18
  --batch 8
  --accum 1
  --seq-len 512
  --val-batch 1
  --val-batches 16
  --bf16
  --tf32
  --telemetry "$RUN_DIR/main.telemetry.json"
)
if [ -s "$MAIN_DIR/run-state.pt" ]; then
  MAIN_ARGS+=(--resume)
else
  MAIN_ARGS+=(
    --init-checkpoint "$INITIAL_CHECKPOINT"
    --init-state "$INITIAL_STATE"
  )
fi
run_child "$PYTHON" -m fractal.natural_train "${MAIN_ARGS[@]}" >"$RUN_DIR/main.log" 2>&1

write_status "training-complete"
