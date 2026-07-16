#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 ROOT PYTHON END_ISO SERVICE_DIR RESULTS_NAME -- RUNNER_ARGS..." >&2
  exit 2
fi

ROOT=$1
RESULTS_NAME=$5
STATUS="$ROOT/results/${RESULTS_NAME}.status.json"
PID_FILE="$ROOT/results/${RESULTS_NAME}.pid"
LAUNCH_LOG="$ROOT/results/${RESULTS_NAME}.launch.log"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$ROOT/results"
nohup "$SCRIPT_DIR/run-event-algebra-window.sh" "$@" >"$LAUNCH_LOG" 2>&1 </dev/null &
LAUNCH_PID=$!

for _attempt in $(seq 1 30); do
  if [ -s "$PID_FILE" ] && [ -s "$STATUS" ]; then
    RUN_PID=$(tr -cd '0-9' <"$PID_FILE")
    if [ "$RUN_PID" = "$LAUNCH_PID" ] && kill -0 "$RUN_PID" 2>/dev/null && \
       grep -Eq '"stage":"(starting|running)"' "$STATUS"; then
      echo "event_window_pid=$RUN_PID"
      echo "event_window_status=$STATUS"
      exit 0
    fi
  fi
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "Event Algebra window exited before the startup handshake" >&2
    tail -40 "$LAUNCH_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Event Algebra window did not complete the startup handshake within 30 seconds" >&2
kill -TERM "$LAUNCH_PID" 2>/dev/null || true
exit 1
