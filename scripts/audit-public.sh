#!/usr/bin/env bash
set -euo pipefail

failed=0

report_matches() {
  local label=$1
  local pattern=$2
  local output
  output=$(git grep -n -I -E "$pattern" -- . ':!scripts/audit-public.sh' || true)
  if [[ -n "$output" ]]; then
    echo "ERROR: $label"
    echo "$output"
    failed=1
  fi
}

report_matches "credential-like value found" \
  '(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{20,})'
report_matches "private filesystem path found" '(/home/[^ /"]+|/Users/[^ /"]+)'
private_ips=$(git grep -n -I -E \
  '(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})' \
  -- . ':!uv.lock' ':!scripts/audit-public.sh' || true)
if [[ -n "$private_ips" ]]; then
  echo "ERROR: private IPv4 address found"
  echo "$private_ips"
  failed=1
fi

forbidden=$(git ls-files | rg -i \
  '(^|/)(\.env($|\.)|\.claude/|\.agents/|\.codex/|\.playwright-cli/)|\.(pt|pth|safetensors|resume)$|(^|/)(persist|chat_state)\.(pt|pth)$' \
  || true)
if [[ -n "$forbidden" ]]; then
  echo "ERROR: forbidden generated or private files are tracked"
  echo "$forbidden"
  failed=1
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  metadata=$(git log --format='%ae%n%ce' | rg -v '(^$|@users\.noreply\.github\.com$)' || true)
  if [[ -n "$metadata" ]]; then
    echo "ERROR: commit history contains a non-noreply email"
    echo "$metadata"
    failed=1
  fi
fi

if (( failed )); then
  exit 1
fi
echo "Public-tree audit passed."
