#!/usr/bin/env bash
set -euo pipefail

# Serve the stable SFT generalist and narrow GRPO specialist on one 1.7B base.
# Override these paths in the environment when deploying outside the training host.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen3-1.7B}"
SFT_ADAPTER="${SFT_ADAPTER:-/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-1.7b-native-react-sft-decision-bridge-step3-v1}"
GRPO_ADAPTER="${GRPO_ADAPTER:-/root/autodl-tmp/TravelAgent2/ml/agentic/checkpoints/qwen3-1.7b-native-react-grpo-decision-kl001-lr5e7-step1-seed06-v3}"
POLICY_PORT="${POLICY_PORT:-8001}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_ROOT}/backend/src/agentic/templates/qwen3_agent_prefix_preserving_v1.jinja}"
CHAT_TEMPLATE_SHA256="dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63"

if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "Agent chat template is missing: ${CHAT_TEMPLATE}" >&2
  exit 2
fi
OBSERVED_TEMPLATE_SHA256="$(sha256sum "${CHAT_TEMPLATE}" | awk '{print $1}')"
if [[ "${OBSERVED_TEMPLATE_SHA256}" != "${CHAT_TEMPLATE_SHA256}" ]]; then
  echo "Agent chat template hash mismatch: ${OBSERVED_TEMPLATE_SHA256}" >&2
  exit 2
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --served-model-name travel-qwen3-1.7b-base \
  --host 0.0.0.0 \
  --port "${POLICY_PORT}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-lora \
  --lora-modules \
    "travel-sft=${SFT_ADAPTER}" \
    "travel-grpo-poi=${GRPO_ADAPTER}" \
  --max-lora-rank 16 \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --enable-prefix-caching \
  --enforce-eager \
  --disable-log-requests
