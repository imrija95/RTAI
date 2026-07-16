#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 ROOT PYTHON SERVICE_DIR RESULTS_PREFIX" >&2
  exit 2
fi

ROOT=$1
PREFIX=$4
RESULTS_DIR="$ROOT/results"
STATUS="$RESULTS_DIR/${PREFIX}.status.json"
PID_FILE="$RESULTS_DIR/${PREFIX}.pid"
LAUNCH_LOG="$RESULTS_DIR/${PREFIX}.launch.log"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$RESULTS_DIR"
nohup "$SCRIPT_DIR/run-growing-cortex-dl580.sh" "$@" >"$LAUNCH_LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

for _attempt in $(seq 1 30); do
  if kill -0 "$PID" 2>/dev/null && [ -s "$STATUS" ] && \
     grep -Eq '"stage":"(starting|running|running-[0-9]+)"' "$STATUS"; then
    echo "growing_cortex_pid=$PID"
    echo "growing_cortex_status=$STATUS"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Growing Cortex suite exited before the startup handshake" >&2
    tail -40 "$LAUNCH_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Growing Cortex suite did not complete the startup handshake" >&2
kill -TERM "$PID" 2>/dev/null || true
exit 1
