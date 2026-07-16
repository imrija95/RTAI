#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 ROOT RUN_DIR SERVICE_DIR" >&2
  exit 2
fi

ROOT=$1
RUN_DIR=$2
mkdir -p "$RUN_DIR"
LAUNCH_LOG="$RUN_DIR/launch.log"
PID_FILE="$RUN_DIR/orchestrator.pid"
STATUS="$RUN_DIR/status.json"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

nohup "$SCRIPT_DIR/run-natural-cortex-gpu.sh" "$@" >"$LAUNCH_LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

for _attempt in $(seq 1 30); do
  if kill -0 "$PID" 2>/dev/null && [ -s "$STATUS" ] && \
     grep -Fq "\"pid\":${PID}," "$STATUS" && \
     grep -Eq '"stage":"(starting|tokenizer|data|verify-data|stopping-inference-service|preflight|dense-moe-ab|main-[^"]+|training-complete)"' "$STATUS"; then
    echo "natural_cortex_pid=$PID"
    echo "natural_cortex_status=$STATUS"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Natural Cortex orchestrator exited before the startup handshake" >&2
    tail -40 "$LAUNCH_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Natural Cortex orchestrator did not complete the startup handshake" >&2
kill -TERM "$PID" 2>/dev/null || true
exit 1
