#!/usr/bin/env bash
# Performance gate for CI / pre-release (M6)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

echo "==> Unit + integration tests (excluding slow performance benchmarks)"
uv run pytest tests/unit tests/integration tests/security tests/chaos -q \
  -m "not performance"

echo "==> Optional: run k6 load test when k6 is installed"
if command -v k6 >/dev/null 2>&1; then
  k6 run "$ROOT/scripts/load/k6_sse.js" -e VUS=10 -e DURATION=30s
else
  echo "k6 not installed — skipping load test (install from https://k6.io)"
fi

echo "==> Performance gate complete"
