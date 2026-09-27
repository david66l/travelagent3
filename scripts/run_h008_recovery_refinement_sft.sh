#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/TravelAgent2-h005-eval-20260903"
DATASET="$REPO/artifacts/native-react-posttraining/h008-readiness-20260905/h008-recovery-refinement-v3"
MODEL="$REPO/artifacts/native-react-posttraining/h008-multiturn-second-pass-sft-v2-seed20260905/checkpoint-64"
OUTPUT="$REPO/artifacts/native-react-posttraining/h008-recovery-refinement-sft-v3-seed20260905"
LOG_DIR="$REPO/artifacts/native-react-posttraining/h008-readiness-20260905/logs"
LOG="$LOG_DIR/h008-recovery-refinement-sft-v3-seed20260905.log"

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
  --model "$MODEL" \
  --tokenizer /root/autodl-tmp/models/Qwen3-1.7B \
  --minimum-train-examples 300 \
  --max-length 6144 \
  --epochs 1 \
  --learning-rate 1e-6 \
  --optim adamw_torch \
  --batch-size 1 \
  --gradient-accumulation 12 \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed 20260905 \
  --logging-steps 1 \
  --eval-steps 4 \
  --save-steps 4 \
  --save-total-limit 5 \
  --warmup-ratio 0.05 \
  --weight-decay 0 \
  --lr-scheduler-type linear \
  --max-grad-norm 1 \
  --termination-token-weight 2 \
  --external-test-evaluation \
  --quarantine \
  2>&1 | tee "$LOG"
