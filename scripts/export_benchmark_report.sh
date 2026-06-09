#!/usr/bin/env bash
# Export benchmark results to a JSON report file.
# Usage:
#   bash scripts/export_benchmark_report.sh <run_id>
#   bash scripts/export_benchmark_report.sh          # uses latest run_id

set -euo pipefail

NAMESPACE="hospital-llm"
BENCHMARK_SVC="benchmark-svc"
PORT=8002
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_reports}"

mkdir -p "$OUTPUT_DIR"

# Port-forward in background
kubectl port-forward svc/$BENCHMARK_SVC $PORT:$PORT -n $NAMESPACE &
PF_PID=$!
trap "kill $PF_PID 2>/dev/null" EXIT
sleep 3

if [[ -z "${1:-}" ]]; then
    # Fetch latest run_id from DB
    RUN_ID=$(kubectl exec -n $NAMESPACE postgres-svc-0 -- \
        psql -U postgres -d hospital -At \
        -c "SELECT run_id FROM benchmark_results ORDER BY evaluated_at DESC LIMIT 1;")
    if [[ -z "$RUN_ID" ]]; then
        echo "ERROR: No benchmark results found in DB. Run a benchmark first:"
        echo "  curl -X POST http://localhost:$PORT/v1/run?limit=100"
        exit 1
    fi
    echo "Using latest run_id: $RUN_ID"
else
    RUN_ID="$1"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/report_${RUN_ID:0:8}_${TIMESTAMP}.json"

curl -sf "http://localhost:$PORT/v1/results?run_id=$RUN_ID" | python3 -m json.tool > "$OUTPUT_FILE"

echo "Report saved to: $OUTPUT_FILE"
cat "$OUTPUT_FILE"
