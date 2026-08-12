#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAMESPACE="${NAMESPACE:-travel-agent}"

kubectl apply -f "$ROOT/k8s/namespace.yaml"
kubectl -n "$NAMESPACE" create configmap gateway-nginx-config \
  --from-file=nginx.conf="$ROOT/gateway/nginx.conf" \
  --dry-run=client -o yaml | kubectl apply -f -

for manifest in "$ROOT/k8s"/*.yaml; do
  case "$(basename "$manifest")" in
    gateway-configmap.yaml) continue ;;
  esac
  kubectl apply -f "$manifest"
done

echo "K8s manifests applied to namespace: $NAMESPACE"
