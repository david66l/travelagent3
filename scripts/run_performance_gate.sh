#!/usr/bin/env bash
# Performance gate for CI / pre-release (M6)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

echo "==> Unit + integration tests (excluding slow performance benchmarks)"
uv run pytest tests/unit tests/integration tests/security tests/chaos -q \
  -m "not performance"

echo "==> Deterministic 500-way intent concurrency and fallback gate"
VUS="${INTENT_VUS:-500}" uv run pytest tests/load/test_intent_concurrency.py -q -s

echo "==> Live HTTP/PostgreSQL/Redis admission load gate"
if command -v k6 >/dev/null 2>&1; then
  mkdir -p "$ROOT/artifacts/performance"
  k6 run "$ROOT/scripts/load/k6_sse.js" \
    -e "BASE_URL=${BASE_URL:-http://localhost:8000}" \
    -e "VUS=${K6_VUS:-20}" \
    -e "DURATION=${K6_DURATION:-30s}" \
    --summary-export "$ROOT/artifacts/performance/k6-summary.json"
elif [[ "${REQUIRE_K6:-0}" == "1" ]]; then
  echo "k6 is required for this release gate but is not installed" >&2
  exit 1
else
  echo "k6 not installed; local load test skipped (CI sets REQUIRE_K6=1)"
fi

echo "==> Performance gate complete"
