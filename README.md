# Healthcare Inference

End-to-end LLM inference pipeline for clinical document processing, deployed on AWS EKS. Simulates hospital encounters, runs multi-task inference via vLLM, and evaluates output quality against ground-truth labels.

## Architecture

```
Simulator → RabbitMQ → Inference Service → PostgreSQL → Benchmark Service
                            ↑
                          vLLM (Qwen/Qwen3.5-9B)
```

| Component | Role |
|---|---|
| **Simulator** | Generates synthetic hospital encounters (admissions, labs, clinical notes, discharge) and publishes to RabbitMQ queues |
| **Inference Service** | Consumes RabbitMQ messages, buffers documents per encounter, runs 5 parallel inference tasks via vLLM on discharge |
| **vLLM** | Serves `Qwen/Qwen3.5-9B` in bfloat16 with guided JSON output; runs as a sidecar in the inference pod |
| **Benchmark Service** | Evaluates inference results against ground-truth labels across quality and performance metrics |
| **PostgreSQL** | Stores encounters, inference results, ground-truth labels, and benchmark results |

## Inference Tasks

Each encounter triggers 5 tasks:

- `diagnosis` — primary diagnosis + ICD-10 code + confidence
- `entity_extraction` — medications, diagnoses, lab findings
- `structured_json` — full structured discharge summary
- `summarization` — free-text clinical summary
- `risk_classification` — mortality and readmission risk scores

## Benchmark Results (Qwen/Qwen3.5-9B, 100 encounters)

**LLM Quality**

| Metric | Value |
|---|---|
| JSON valid rate | 1.00 |
| Function calling success | 75.9% |
| Diagnosis exact match (ICD-10) | 38% |
| Diagnosis chapter match | 46% |
| Structured JSON score | 51.6% |
| ROUGE-L (summarization) | 0.292 |
| Risk Brier score | 0.088 |
| Hallucination rate | 39% |

**Infrastructure (g6e.xlarge / L40S 48GB)**

| Metric | Value |
|---|---|
| p50 latency | 1,607 ms |
| p95 latency | 1,778 ms |
| p50 TTFT | 98 ms |
| p95 TTFT | 112 ms |
| p50 ITL | 32 ms |
| Avg tokens/sec | 29.8 |
| Cost per 1M tokens | $21.09 |

## Infrastructure

- **Cluster**: AWS EKS `hospital-llm`, region `ap-south-1`, Kubernetes 1.30
- **GPU node**: `g6e.xlarge` (NVIDIA L40S 48GB VRAM) — managed node group, scales 1–3
- **System nodes**: `m5.large` — managed node group, scales 1–2
- **Storage**: EBS CSI driver with IRSA for PersistentVolumeClaims

## Stack

- **Runtime**: Python 3.12, FastAPI, asyncpg, SQLAlchemy (async), aio-pika
- **Inference**: vLLM 0.22+, OpenAI-compatible API, guided JSON via `guided_json`
- **Messaging**: RabbitMQ 3.13 (official image), 4 queues: admissions, labs, clinical_notes, discharge
- **Observability**: Prometheus + Grafana (kube-prometheus-stack), DCGM Exporter for GPU metrics
- **Container registry**: Amazon ECR

## Repository Layout

```
├── inference-service/     # Inference app + vLLM consumer
├── benchmark-service/     # Benchmark API + metric evaluators
├── simulator-service/     # Synthetic encounter generator
├── k8s/                   # Kubernetes manifests
│   ├── inference/
│   ├── benchmark/
│   ├── simulator/
│   ├── rabbitmq/
│   └── monitoring/
└── scripts/
    ├── setup_eks.sh               # Cluster bootstrap
    ├── export_benchmark_report.sh # Fetch benchmark JSON via port-forward
    └── export_benchmark_csv.sh    # Export benchmark_results table as CSV
```

## Running the Pipeline

**1. Bootstrap the cluster**
```bash
bash scripts/setup_eks.sh
```

**2. Simulate encounters**
```bash
kubectl port-forward svc/simulator-svc 8000:8000 -n hospital-llm
curl -X POST "http://localhost:8000/v1/simulate?n=100"
```

**3. Run benchmark** (after inference completes)
```bash
kubectl port-forward svc/benchmark-svc 8003:8002 -n hospital-llm
curl -X POST "http://localhost:8003/v1/run?limit=100"
```

**4. Export results**
```bash
bash scripts/export_benchmark_report.sh <run_id>
bash scripts/export_benchmark_csv.sh <run_id>
```

## Model Notes

- Qwen3.5-9B is a reasoning model; thinking mode is explicitly disabled (`enable_thinking: false`) for direct JSON output
- `--gpu-memory-utilization 0.92` on L40S 48GB with `--max-model-len 16384`
- `--load-format safetensors --safetensors-load-strategy mmap` reduces cold-start time
- vLLM startup takes ~9 minutes; liveness probe has `initialDelaySeconds: 600`
