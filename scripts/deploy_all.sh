#!/bin/bash
# TravelAgent2 完整部署脚本 (M5)
set -euo pipefail

NAMESPACE="travel-agent"
echo "🚀 Deploying TravelAgent2 to K8s namespace: $NAMESPACE"

# 1. Create namespace
kubectl apply -f k8s/namespace.yaml

# 2. Config & Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 3. Databases
echo "📦 Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/postgres-service.yaml

echo "📦 Deploying Redis Cluster..."
kubectl apply -f k8s/redis-statefulset.yaml
kubectl apply -f k8s/redis-service.yaml

kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=120s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=120s

# 4. vLLM (optional — needs GPU)
if [ "${DEPLOY_VLLM:-false}" = "true" ]; then
  echo "🧠 Deploying vLLM..."
  kubectl apply -f k8s/vllm-deployment.yaml
  kubectl apply -f k8s/vllm-service.yaml
fi

# 5. MLflow
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml

# 6. Backend
echo "🔧 Deploying Backend..."
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml

# 7. Celery Workers
echo "⚙️  Deploying Celery Workers..."
kubectl apply -f k8s/celery-worker-deployment.yaml
kubectl apply -f k8s/celery-beat-deployment.yaml

# 8. Gateway
echo "🌐 Deploying Gateway..."
kubectl apply -f k8s/gateway-deployment.yaml
kubectl apply -f k8s/gateway-service.yaml
kubectl apply -f k8s/gateway-ingress.yaml
kubectl apply -f k8s/gateway-configmap.yaml

# 9. Frontend
echo "🎨 Deploying Frontend..."
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml

# 10. HPA + PDB
echo "📈 Applying HPA + PDB..."
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/pdb.yaml

# 11. Backup CronJob
kubectl apply -f k8s/backup-cronjob.yaml

echo ""
echo "✅ TravelAgent2 deployed!"
echo "   kubectl get pods -n $NAMESPACE"
echo "   kubectl get ingress -n $NAMESPACE"
