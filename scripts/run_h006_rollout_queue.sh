#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/root/autodl-tmp/TravelAgent2-h005-eval-20260903"
readonly DATA_ROOT="$REPO_ROOT/artifacts/native-react-posttraining/step4-h006-data-20260903"
readonly CHECKPOINT_ROOT="/root/autodl-tmp/TravelAgent2-verifier-repair/artifacts/native-react-posttraining"
readonly H001="$CHECKPOINT_ROOT/h001-canonical-h48-seed20260923/checkpoint-48"
readonly H004="$CHECKPOINT_ROOT/h004-dpo-reason-repair-on-r2"
readonly WAIT_PID="${1:-}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO_ROOT/backend/src:$REPO_ROOT"

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
fi

run_rollout() {
  local checkpoint="$1"
  local corpus="$2"
  local output="$3"
  local tasks="$4"
  local iterations="$5"
  local seed="$6"
  python scripts/generate_h006_local_donor_rollouts.py \
    --checkpoint "$checkpoint" \
    --corpus-file "$corpus" \
    --output-dir "$output" \
    --max-tasks "$tasks" \
    --samples-per-task 4 \
    --temperature 0.8 \
    --max-new-tokens 192 \
    --max-tool-calling-iterations "$iterations" \
    --seed "$seed" \
    --load-in-4bit
  test -s "$output/report.json"
}

require_successful_sources() {
  local output="$1"
  local minimum="$2"
  python - "$output/report.json" "$minimum" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
observed = int(report.get("successful_sources") or 0)
minimum = int(sys.argv[2])
if observed < minimum:
    raise SystemExit(f"only {observed} successful sources; required {minimum}")
PY
}

run_rollout \
  "$H004" \
  "$DATA_ROOT/hard-donor-supplement/train.jsonl" \
  "$DATA_ROOT/hard-h004-wave2" \
  60 1 20260921
require_successful_sources "$DATA_ROOT/hard-h004-wave2" 40

run_rollout \
  "$H004" \
  "$DATA_ROOT/hard-donor-candidates/validation.jsonl" \
  "$DATA_ROOT/hard-h004-validation" \
  54 1 20260922
require_successful_sources "$DATA_ROOT/hard-h004-validation" 24

run_rollout \
  "$H001" \
  "$DATA_ROOT/h001-boundary-candidates/train.jsonl" \
  "$DATA_ROOT/h001-boundary-train" \
  90 1 20260923
require_successful_sources "$DATA_ROOT/h001-boundary-train" 75

run_rollout \
  "$H001" \
  "$DATA_ROOT/h001-boundary-candidates/validation.jsonl" \
  "$DATA_ROOT/h001-boundary-validation" \
  30 1 20260924
require_successful_sources "$DATA_ROOT/h001-boundary-validation" 24

run_rollout \
  "$H001" \
  "$DATA_ROOT/easy-anchor-candidates/validation.jsonl" \
  "$DATA_ROOT/h001-easy-smoke" \
  3 14 20260925
require_successful_sources "$DATA_ROOT/h001-easy-smoke" 2

run_rollout \
  "$H001" \
  "$DATA_ROOT/easy-anchor-candidates/train.jsonl" \
  "$DATA_ROOT/h001-easy-train" \
  120 14 20260926
require_successful_sources "$DATA_ROOT/h001-easy-train" 70

run_rollout \
  "$H001" \
  "$DATA_ROOT/easy-anchor-candidates/validation.jsonl" \
  "$DATA_ROOT/h001-easy-validation" \
  36 14 20260927
require_successful_sources "$DATA_ROOT/h001-easy-validation" 20

echo "H006 rollout queue completed"
