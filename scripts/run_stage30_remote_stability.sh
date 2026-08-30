#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/TravelAgent2

CASE_ROOT="ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1/stage30-routed-v1"
REPORTS="ml/agentic/reports"
VLLM_PYTHON="/root/stage19-vllm-0.9.2/bin/python"
SERVER_MODULE="vllm.entrypoints.openai.api_server"
GPU_LOG="/tmp/stage30_gpu_metrics.log"

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

record_gpu() {
  local label="$1"
  printf '%s ' "$label" >> "$GPU_LOG"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader >> "$GPU_LOG"
}

run_benchmark() {
  local model="$1"
  local cases="$2"
  local output="$3"
  local temperature="$4"
  local repetitions="$5"
  local concurrency="$6"
  if [[ -s "$output/report.json" && -s "$output/runs.jsonl" ]]; then
    echo "SKIP complete: $output"
    return 0
  fi
  echo "RUN model=$model temperature=$temperature repetitions=$repetitions concurrency=$concurrency"
  .venv/bin/python scripts/benchmark_vllm_http.py \
    --base-url http://127.0.0.1:8000/v1 \
    --model "$model" \
    --cases-file "$cases" \
    --output-dir "$output" \
    --thinking disabled \
    --max-tokens 192 \
    --temperature "$temperature" \
    --repetitions "$repetitions" \
    --warmup 2 \
    --concurrency "$concurrency" \
    --timeout-seconds 120
}

: > "$GPU_LOG"
stop_server stage30_4b_vllm
stop_server stage30_8b_vllm

screen -dmS stage30_4b_vllm bash -lc "cd /root/autodl-tmp/TravelAgent2 && exec $VLLM_PYTHON -m $SERVER_MODULE \
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
  --lora-modules travel-policy-qwen3-4b-stage28-dpo-v2=/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-4b-stage28-dpo-abort-diverse-v2 \
  --max-lora-rank 16 \
  --max-loras 1 \
  --enforce-eager \
  --disable-log-requests > /tmp/stage30_4b_vllm.log 2>&1"

wait_ready /tmp/stage30_4b_vllm.log
record_gpu student_4b_idle

run_benchmark travel-policy-qwen3-4b-stage28-dpo-v2 "$CASE_ROOT/student-cases.jsonl" \
  "$REPORTS/stage30-student-stability-c8-t0-r5-v1" 0 5 8
run_benchmark travel-policy-qwen3-4b-stage28-dpo-v2 "$CASE_ROOT/student-cases.jsonl" \
  "$REPORTS/stage30-student-stochastic-c8-t02-r3-v1" 0.2 3 8
for concurrency in 1 4 16; do
  run_benchmark travel-policy-qwen3-4b-stage28-dpo-v2 "$CASE_ROOT/student-cases.jsonl" \
    "$REPORTS/stage30-student-performance-c${concurrency}-t0-r1-v1" 0 1 "$concurrency"
done

stop_server stage30_4b_vllm

screen -dmS stage30_8b_vllm bash -lc "cd /root/autodl-tmp/TravelAgent2 && exec $VLLM_PYTHON -m $SERVER_MODULE \
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
  --disable-log-requests > /tmp/stage30_8b_vllm.log 2>&1"

wait_ready /tmp/stage30_8b_vllm.log
record_gpu teacher_8b_idle

run_benchmark travel-policy-qwen3-8b-base "$CASE_ROOT/teacher-cases.jsonl" \
  "$REPORTS/stage30-teacher-stability-c8-t0-r5-v1" 0 5 8
run_benchmark travel-policy-qwen3-8b-base "$CASE_ROOT/teacher-cases.jsonl" \
  "$REPORTS/stage30-teacher-stochastic-c8-t02-r3-v1" 0.2 3 8
for concurrency in 1 4 16; do
  run_benchmark travel-policy-qwen3-8b-base "$CASE_ROOT/teacher-cases.jsonl" \
    "$REPORTS/stage30-teacher-performance-c${concurrency}-t0-r1-v1" 0 1 "$concurrency"
done

stop_server stage30_8b_vllm
record_gpu final_idle
cat "$GPU_LOG"
echo "STAGE30_REMOTE_STABILITY_COMPLETE"
