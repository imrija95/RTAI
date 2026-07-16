#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 CHECKPOINT TOKENIZER RUNTIME_DIR [PORT]" >&2
  exit 2
fi

checkpoint=$1
tokenizer=$2
runtime_dir=$3
port=${4:-8000}

mkdir -p "$runtime_dir"

FRACTAL_CKPT="$checkpoint" \
VIZ_TOKENIZER="$tokenizer" \
VIZ_CHAT=1 \
VIZ_CHAT_STATE="$runtime_dir/fast-weights.pt" \
VIZ_CHAT_SESSION="$runtime_dir/session.json" \
VIZ_SKILL_BANK="$runtime_dir/skill-bank" \
PORT="$port" \
uv run python -m fractal.viz_serve
