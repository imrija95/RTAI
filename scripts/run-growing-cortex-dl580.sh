#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 ROOT PYTHON SERVICE_DIR RESULTS_PREFIX" >&2
  exit 2
fi

ROOT=$1
PYTHON=$2
SERVICE_DIR=$3
PREFIX=$4
RESULTS_DIR="$ROOT/results"
STATUS="$RESULTS_DIR/${PREFIX}.status.json"
GPU_LOG="$RESULTS_DIR/${PREFIX}.gpu.csv"
MAIN_LOG="$RESULTS_DIR/${PREFIX}.gpu.log"
SERVICE_STOPPED=0
PIDS=()
ROUTER_THRESHOLD=${GROWING_ROUTER_THRESHOLD:-0.72}
GPU_RESUME_ARGS=()
if [ -n "${GROWING_GPU_RESUME_CHECKPOINT:-}" ]; then
  GPU_RESUME_ARGS=(--resume_checkpoint "$GROWING_GPU_RESUME_CHECKPOINT")
fi

mkdir -p "$RESULTS_DIR"

write_status() {
  local stage=$1
  local code=${2:-0}
  local temporary="${STATUS}.tmp"
  printf '{"schema_version":1,"stage":"%s","pid":%s,"exit_code":%s,"updated_epoch":%s}\n' \
    "$stage" "$$" "$code" "$(date +%s)" >"$temporary"
  mv "$temporary" "$STATUS"
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [ "$SERVICE_STOPPED" -eq 1 ]; then
    docker compose --project-directory "$SERVICE_DIR" up -d \
      >>"$RESULTS_DIR/${PREFIX}.service.log" 2>&1 || code=1
  fi
  write_status "finished" "$code"
  exit "$code"
}
trap cleanup EXIT INT TERM

write_status "starting"
if [ "${GROWING_CORTEX_STOP_SERVICE:-1}" = "1" ]; then
  SERVICE_STOPPED=1
  docker compose --project-directory "$SERVICE_DIR" stop \
    >>"$RESULTS_DIR/${PREFIX}.service.log" 2>&1
fi

cd "$ROOT"
write_status "running"

OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 numactl --cpunodebind=3 --membind=3 \
  "$PYTHON" -m fractal.exp_growing_cortex \
  --seed 20260716 \
  --interpreter_steps "${GROWING_GPU_INTERPRETER_STEPS:-8000}" \
  --steps "${GROWING_GPU_COMPILER_STEPS:-8000}" \
  --meta_tasks 8 --queries 16 --eval_tasks 16 --eval_inputs 64 \
  --n_embd 256 --n_head 8 --depth 6 --n_scales 2 --skill_rank 16 \
  --router_threshold "$ROUTER_THRESHOLD" --lr 0.001 --bf16 --tf32 --report_every 100 \
  "${GPU_RESUME_ARGS[@]}" \
  --telemetry "$RESULTS_DIR/${PREFIX}.tele.json" \
  --results "$RESULTS_DIR/${PREFIX}.gpu.json" >"$MAIN_LOG" 2>&1 &
PIDS+=("$!")

CPU_SEEDS=(20260717 20260718 20260719)
CPU_RANKS=(4 8 12)
for node in 0 1 2; do
  CPU_RESUME_ARGS=()
  if [ -n "${GROWING_CPU_RESUME_PREFIX:-}" ]; then
    CPU_RESUME_ARGS=(
      --resume_checkpoint "${GROWING_CPU_RESUME_PREFIX}.cpu${node}.restart.pt"
    )
  fi
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
    numactl "--cpunodebind=$node" "--membind=$node" \
    "$PYTHON" -m fractal.exp_growing_cortex \
    --seed "${CPU_SEEDS[$node]}" \
    --interpreter_steps "${GROWING_CPU_INTERPRETER_STEPS:-3000}" \
    --steps "${GROWING_CPU_COMPILER_STEPS:-3000}" \
    --meta_tasks 4 --queries 8 --eval_tasks 12 --eval_inputs 32 \
    --n_embd 96 --n_head 4 --depth 3 --n_scales 2 \
    --skill_rank "${CPU_RANKS[$node]}" --router_threshold "$ROUTER_THRESHOLD" --lr 0.002 \
    "${CPU_RESUME_ARGS[@]}" \
    --report_every 200 \
    --results "$RESULTS_DIR/${PREFIX}.cpu${node}.json" \
    >"$RESULTS_DIR/${PREFIX}.cpu${node}.log" 2>&1 &
  PIDS+=("$!")
done

while :; do
  live=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    fi
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >>"$GPU_LOG" 2>/dev/null || true
  fi
  write_status "running-${live}"
  if [ "$live" -eq 0 ]; then
    break
  fi
  sleep "${GROWING_CORTEX_POLL_SECONDS:-30}"
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
PIDS=()
if [ "$failed" -ne 0 ]; then
  exit 1
fi
write_status "experiments-complete"
