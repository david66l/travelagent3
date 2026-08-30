#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TravelAgent2

CASES="ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1/vllm-cases.jsonl"
REPORTS="ml/agentic/reports"
VLLM_PYTHON="/root/stage19-vllm-0.9.2/bin/python"
SERVER_MODULE="vllm.entrypoints.openai.api_server"

stop_server() {
  local name="$1"
  screen -S "$name" -X quit >/dev/null 2>&1 || true
  sleep 4
  screen -wipe >/dev/null 2>&1 || true
}

wait_ready() {
  local log_path="$1"
  for _ in $(seq 1 45); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null; then
      return 0
    fi
    sleep 2
  done
  tail -80 "$log_path" || true
  return 1
}

run_arm() {
  local model="$1"
  local output="$2"
  if [[ -s "$output/report.json" && -s "$output/runs.jsonl" ]]; then
    echo "SKIP complete arm: $model"
    return 0
  fi
  echo "RUN arm: $model"
  .venv/bin/python scripts/benchmark_vllm_http.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model "$model" \
    --cases-file "$CASES" \
    --output-dir "$output" \
    --thinking disabled \
    --max-tokens 192 \
    --temperature 0 \
    --repetitions 1 \
    --warmup 2 \
    --concurrency 8 \
    --timeout-seconds 120
}

stop_server stage29_4b_vllm
stop_server stage29_8b_vllm

screen -dmS stage29_4b_vllm bash -lc "cd /root/autodl-tmp/TravelAgent2 && exec $VLLM_PYTHON -m $SERVER_MODULE \
  --model /root/models/Qwen3-4B \
  --served-model-name travel-policy-qwen3-4b-base \
  --dtype half \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-lora \
  --lora-modules \
    travel-policy-qwen3-4b-stage21-sft=/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-4b-stage21-sft-balanced-formal-v1 \
    travel-policy-qwen3-4b-stage22-dpo=/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-4b-stage22-dpo-balanced-formal-v1 \
    travel-policy-qwen3-4b-stage28-sft-v2=/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-4b-stage28-sft-abort-diverse-v2 \
    travel-policy-qwen3-4b-stage28-dpo-v2=/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-4b-stage28-dpo-abort-diverse-v2 \
  --max-lora-rank 16 \
  --max-loras 4 \
  --enforce-eager \
  --disable-log-requests > /tmp/stage29_4b_vllm.log 2>&1"

wait_ready /tmp/stage29_4b_vllm.log
echo "READY 4B server"

run_arm travel-policy-qwen3-4b-base "$REPORTS/stage29-deepseek-base4b-v1"
run_arm travel-policy-qwen3-4b-stage21-sft "$REPORTS/stage29-deepseek-stage21-sft4b-v1"
run_arm travel-policy-qwen3-4b-stage22-dpo "$REPORTS/stage29-deepseek-stage22-dpo4b-v1"
run_arm travel-policy-qwen3-4b-stage28-sft-v2 "$REPORTS/stage29-deepseek-stage28-sft4b-v2"
run_arm travel-policy-qwen3-4b-stage28-dpo-v2 "$REPORTS/stage29-deepseek-stage28-dpo4b-v2"

stop_server stage29_4b_vllm

screen -dmS stage29_8b_vllm bash -lc "cd /root/autodl-tmp/TravelAgent2 && exec $VLLM_PYTHON -m $SERVER_MODULE \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --served-model-name travel-policy-qwen3-8b-base \
  --dtype half \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  --disable-log-requests > /tmp/stage29_8b_vllm.log 2>&1"

wait_ready /tmp/stage29_8b_vllm.log
echo "READY 8B server"

run_arm travel-policy-qwen3-8b-base "$REPORTS/stage29-deepseek-teacher8b-v1"

stop_server stage29_8b_vllm
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "STAGE29_REMOTE_BENCHMARK_COMPLETE"
