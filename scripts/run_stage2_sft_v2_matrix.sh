#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset="ml/agentic/datasets/stage2-sft-v2-curriculum-v1"
model="ml/agentic/checkpoints/stage2-policy-driven-sft-smoke20-v1"
report_dir="ml/agentic/reports/stage2-sft-v2-training-v1"
mkdir -p "$report_dir"

for spec in \
  "stage2-sft-v2-lr3e6:3e-6" \
  "stage2-sft-v2-lr8e6:8e-6" \
  "stage2-sft-v2-lr1e5:1e-5"
do
  run_id="${spec%%:*}"
  learning_rate="${spec##*:}"
  output_dir="ml/agentic/checkpoints/${run_id}"
  log_file="${report_dir}/${run_id}.log"
  if [[ -f "${output_dir}/training_report.json" ]]; then
    echo "skip completed ${run_id}" | tee -a "$log_file"
    continue
  fi
  python ml/agentic/training/train_sft.py \
    --dataset-dir "$dataset" \
    --output-dir "$output_dir" \
    --model "$model" \
    --minimum-train-examples 1000 \
    --max-length 2048 \
    --epochs 1 \
    --max-steps 60 \
    --learning-rate "$learning_rate" \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --termination-token-weight 1.0 \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --logging-steps 5 \
    --eval-steps 10 \
    --save-steps 10 \
    --save-total-limit 7 \
    --eval-during-smoke \
    --max-eval-examples 128 \
    2>&1 | tee "$log_file"
done
