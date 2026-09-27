#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/TravelAgent2-h005-eval-20260903"
DATASET="$REPO/artifacts/native-react-posttraining/step4-h006-data-20260903/h006-internal-corrective-mix-v2"
OUTPUT="$REPO/artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930"
LOG_DIR="$REPO/artifacts/native-react-posttraining/logs"
LOG="$LOG_DIR/h006-internal-corrective-sft-v2-seed20260930.log"

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
cd "$REPO"

export PYTHONPATH="$REPO/backend/src:$REPO"
export CUDA_VISIBLE_DEVICES=0

/root/miniconda3/bin/python ml/agentic/training/train_sft.py \
  --dataset-dir "$DATASET" \
  --output-dir "$OUTPUT" \
  --model /root/autodl-tmp/TravelAgent2-verifier-repair/artifacts/native-react-posttraining/h001-canonical-h48-seed20260923/checkpoint-48 \
  --tokenizer /root/autodl-tmp/models/Qwen3-1.7B \
  --minimum-train-examples 300 \
  --max-length 6144 \
  --epochs 1 \
  --learning-rate 5e-6 \
  --optim adamw_torch \
  --batch-size 1 \
  --gradient-accumulation 12 \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed 20260930 \
  --logging-steps 1 \
  --eval-steps 8 \
  --save-steps 8 \
  --save-total-limit 4 \
  --warmup-ratio 0.05 \
  --weight-decay 0 \
  --lr-scheduler-type linear \
  --max-grad-norm 1 \
  --source-equal-loss \
  --source-lineage "$DATASET/lineage.jsonl" \
  --source-snapshot-manifest "$DATASET/run_input_snapshot.json" \
  --external-test-evaluation \
  --quarantine \
  2>&1 | tee "$LOG"
