#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 ROOT PYTHON END_ISO SERVICE_DIR RESULTS_NAME -- RUNNER_ARGS..." >&2
  exit 2
fi

ROOT=$1
PYTHON=$2
END_ISO=$3
SERVICE_DIR=$4
RESULTS_NAME=$5
shift 5
if [ "${1:-}" != "--" ]; then
  echo "Missing -- before Event Algebra runner arguments" >&2
  exit 2
fi
shift

RESULTS_DIR="$ROOT/results"
STATUS="$RESULTS_DIR/${RESULTS_NAME}.status.json"
PID_FILE="$RESULTS_DIR/${RESULTS_NAME}.pid"
LOG="$RESULTS_DIR/${RESULTS_NAME}.runner.log"
GPU_LOG="$RESULTS_DIR/${RESULTS_NAME}.gpu.csv"
RESULTS="$RESULTS_DIR/${RESULTS_NAME}.json"
END_EPOCH=$(date --date="$END_ISO" +%s)
CHILD_PID=""
SERVICE_STOPPED=0

mkdir -p "$RESULTS_DIR"

write_status() {
  local stage=$1
  local code=${2:-0}
  local temporary="${STATUS}.tmp"
  printf '{"schema_version":1,"stage":"%s","pid":%s,"exit_code":%s,"updated_epoch":%s,"end_epoch":%s}\n' \
    "$stage" "$$" "$code" "$(date +%s)" "$END_EPOCH" >"$temporary"
  mv "$temporary" "$STATUS"
}

restore_service() {
  local code=$?
  trap - EXIT INT TERM
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  if [ "$SERVICE_STOPPED" -eq 1 ]; then
    docker compose --project-directory "$SERVICE_DIR" up -d >>"$LOG" 2>&1 || code=1
  fi
  write_status "finished" "$code"
  exit "$code"
}
trap restore_service EXIT INT TERM

printf '%s\n' "$$" >"$PID_FILE"
write_status "starting"

NOW=$(date +%s)
RESERVE_SECONDS=${EVENT_WINDOW_RESERVE_SECONDS:-300}
BUDGET_SECONDS=$((END_EPOCH - NOW - RESERVE_SECONDS))
if [ "$BUDGET_SECONDS" -lt 60 ]; then
  echo "The requested window is too short after the restore reserve" >>"$LOG"
  exit 2
fi
BUDGET_MINUTES=$(awk -v seconds="$BUDGET_SECONDS" 'BEGIN { printf "%.3f", seconds / 60.0 }')

if [ "${EVENT_WINDOW_STOP_SERVICE:-1}" = "1" ]; then
  SERVICE_STOPPED=1
  docker compose --project-directory "$SERVICE_DIR" stop >>"$LOG" 2>&1
fi

write_status "running"
cd "$ROOT"
RUN_PREFIX=()
if [ -n "${EVENT_WINDOW_CPUS:-}" ]; then
  RUN_PREFIX=(numactl "--physcpubind=${EVENT_WINDOW_CPUS}")
  if [ -n "${EVENT_WINDOW_NUMA_NODE:-}" ]; then
    RUN_PREFIX+=("--membind=${EVENT_WINDOW_NUMA_NODE}")
  fi
fi
MODULE=${EVENT_WINDOW_MODULE:-fractal.exp_event_algebra}
WINDOW_ARGS=(--results "$RESULTS")
if [ "${EVENT_WINDOW_BOUNDED_ARGS:-1}" = "1" ]; then
  WINDOW_ARGS=(
    --budget_minutes "$BUDGET_MINUTES"
    --report_minutes "${EVENT_WINDOW_REPORT_MINUTES:-10}"
    "${WINDOW_ARGS[@]}"
  )
fi
"${RUN_PREFIX[@]}" "$PYTHON" -m "$MODULE" "${WINDOW_ARGS[@]}" "$@" >>"$LOG" 2>&1 &
CHILD_PID=$!
POLL_SECONDS=${EVENT_WINDOW_POLL_SECONDS:-30}

while kill -0 "$CHILD_PID" 2>/dev/null; do
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
  fi
  if [ "$(date +%s)" -ge "$END_EPOCH" ]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    break
  fi
  sleep "$POLL_SECONDS"
done

wait "$CHILD_PID"
CHILD_PID=""
write_status "experiment-complete"
