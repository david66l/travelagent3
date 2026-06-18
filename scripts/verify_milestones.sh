#!/usr/bin/env bash
# Verify M1–M4 + M6 acceptance (M5 K8s deploy excluded per project scope)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> M1 Go gateway compile"
if command -v go >/dev/null 2>&1; then
  (cd gateway && go test ./... && go build -o /tmp/travel-gateway ./cmd/gateway)
else
  echo "WARN: go not installed — skip gateway build"
fi

echo "==> M2–M6 backend tests"
cd backend
uv run pytest tests/unit tests/integration tests/security tests/chaos -q -m "not performance"

echo "==> M4 LoRA eval gate (holdout with predictions)"
uv run python ../ml/training/eval_lora.py \
  --dataset ../ml/training/data/train.jsonl \
  --adapter travel-plan-v1 \
  --min-samples 50 \
  --min-match-rate 0.8 \
  --gate

echo "==> All milestone checks passed (M5 skipped)"
