#!/usr/bin/env bash
# Provisions an EKS cluster with up to 3 g6e.xlarge GPU nodes for the LLM benchmark.
# Requires: aws-cli v2, eksctl >= 0.180, kubectl, helm
#
# Usage:
#   export AWS_REGION=us-east-1
#   export CLUSTER_NAME=hospital-llm
#   bash scripts/setup_eks.sh 
#
# Tear-down:
#   eksctl delete cluster --name "$CLUSTER_NAME" --region "$AWS_REGION"

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-hospital-llm}"
AWS_REGION="${AWS_REGION:-ap-south-1}"
K8S_VERSION="1.30"
NAMESPACE="hospital-llm"
HF_TOKEN="${HF_TOKEN:-}"   # set to download Qwen2.5-9B from HuggingFace

echo "==> Creating EKS cluster: $CLUSTER_NAME in $AWS_REGION"

eksctl create cluster \
  --name "$CLUSTER_NAME" \
  --version "$K8S_VERSION" \
  --region "$AWS_REGION" \
  --with-oidc \
  --without-nodegroup   # system nodegroup created separately below

# ---------- system node group (CPU, for Prometheus / Grafana / benchmark service) ----------
eksctl create nodegroup \
  --cluster "$CLUSTER_NAME" \
  --region  "$AWS_REGION" \
  --name    system \
  --node-type  m5.large \
  --nodes      1 \
  --nodes-min  1 \
  --nodes-max  2 \
  --asg-access \
  --managed

# ---------- GPU node group (g6e.xlarge — L40S 48 GB VRAM) ----------
eksctl create nodegroup \
  --cluster "$CLUSTER_NAME" \
  --region  "$AWS_REGION" \
  --name    gpu-inference \
  --node-type  g6e.xlarge \
  --nodes      1 \
  --nodes-min  1 \
  --nodes-max  3 \
  --node-labels "workload=gpu-inference" \
  --asg-access \
  --managed

# --node-taints is unsupported for managed nodegroups via CLI; apply after nodes are Ready
echo "==> Waiting for GPU nodes to be Ready before applying taint"
kubectl wait --for=condition=Ready nodes -l workload=gpu-inference --timeout=300s
kubectl taint nodes -l workload=gpu-inference nvidia.com/gpu=present:NoSchedule --overwrite

echo "==> Updating kubeconfig"
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

# ---------- NVIDIA device plugin (makes GPU allocatable in k8s) ----------
echo "==> Installing NVIDIA device plugin"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

# ---------- Namespace ----------
kubectl apply -f k8s/namespace.yaml

# ---------- Secrets ----------
echo "==> Creating secrets"

if [[ -z "$HF_TOKEN" ]]; then
  echo "WARN: HF_TOKEN not set — vLLM will fail to pull gated model weights"
fi

kubectl create secret generic hf-credentials \
  --namespace "$NAMESPACE" \
  --from-literal=token="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Substitute real values before running in production
DB_URL="${DB_URL:-postgresql+asyncpg://inference_user:inference_pass@postgres-svc:5432/hospital}"
RABBITMQ_URL="${RABBITMQ_URL:-amqp://guest:guest@rabbitmq-svc:5672/}"

kubectl create secret generic db-credentials \
  --namespace "$NAMESPACE" \
  --from-literal=inference-url="$DB_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic rabbitmq-credentials \
  --namespace "$NAMESPACE" \
  --from-literal=url="$RABBITMQ_URL" \
  --dry-run=client -o yaml | kubectl apply -f -

# ---------- DCGM exporter (GPU metrics → Prometheus) ----------
echo "==> Deploying DCGM exporter"
kubectl apply -f k8s/monitoring/dcgm-daemonset.yaml

# ---------- Inference deployment + HPA ----------
echo "==> Deploying inference + vLLM"
kubectl apply -f k8s/inference/deployment.yaml
kubectl apply -f k8s/inference/hpa.yaml

# ---------- kube-state-metrics (for HPA replica count in Grafana) ----------
echo "==> Installing kube-state-metrics via Helm"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install kube-state-metrics prometheus-community/kube-state-metrics \
  --namespace kube-system \
  --set image.tag=v2.13.0

echo ""
echo "=== Cluster ready ==="
echo "GPU nodes:   $(kubectl get nodes -l workload=gpu-inference --no-headers | wc -l)"
echo "Namespace:   $NAMESPACE"
echo ""
echo "Next steps:"
echo "  1. Deploy Prometheus + Grafana (see docker-compose.yml for local reference)"
echo "  2. Run quality benchmark:  curl -X POST http://<benchmark-svc>/v1/run?limit=100"
echo "  3. Run infra load test:    k6 run --env VLLM_URL=http://<alb-dns>:8001 load-testing/k6/infra_load.js"
echo "  4. Tear down:              eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION"
