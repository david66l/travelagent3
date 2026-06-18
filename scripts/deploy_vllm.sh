#!/usr/bin/env bash
# deploy_vllm.sh — One-click vLLM deployment for TravelAgent2 (M4)
#
# Usage:
#   ./scripts/deploy_vllm.sh [registry/tag]
#
# Defaults:
#   IMAGE_REGISTRY=ghcr.io/travelagent
#   IMAGE_TAG=latest
#
# This script:
# 1. Builds the vLLM inference image (Dockerfile.vllm)
# 2. Pushes it to the configured registry
# 3. Applies the K8s Deployment + Service
# 4. Waits for the vLLM pod to become ready
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE="${NAMESPACE:-travel-agent}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/travelagent}"
IMAGE_TAG="${IMAGE_TAG:-${1:-latest}}"
IMAGE="${IMAGE_REGISTRY}/travel-agent-vllm:${IMAGE_TAG}"

echo "== TravelAgent2 vLLM Deploy =="
echo "Image: ${IMAGE}"
echo "Namespace: ${NAMESPACE}"

# Ensure namespace exists
kubectl apply -f "${ROOT}/k8s/namespace.yaml"

# Build and push image
echo "[1/4] Building vLLM inference image..."
docker build -f "${ROOT}/Dockerfile.vllm" -t "${IMAGE}" "${ROOT}"

echo "[2/4] Pushing image to registry..."
docker push "${IMAGE}"

# Update image in manifest and apply
# We use envsubst to inject the image tag without mutating the repo file.
echo "[3/4] Applying K8s manifests..."
export VLLM_IMAGE="${IMAGE}"
envsubst '${VLLM_IMAGE}' < "${ROOT}/k8s/vllm-deployment.yaml" | kubectl apply -f -
kubectl apply -f "${ROOT}/k8s/vllm-service.yaml"

# Wait for rollout
echo "[4/4] Waiting for vLLM rollout..."
kubectl -n "${NAMESPACE}" rollout status deployment/vllm --timeout=600s

# Print access info
echo ""
echo "vLLM deployed successfully."
echo "  In-cluster URL: http://vllm.${NAMESPACE}.svc.cluster.local:8000/v1"
echo "  Test health:    kubectl -n ${NAMESPACE} exec deploy/vllm -- curl -s http://localhost:8000/health"
