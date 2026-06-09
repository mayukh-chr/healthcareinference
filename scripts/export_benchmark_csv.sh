#!/usr/bin/env bash
# Export all benchmark_results rows to a CSV file, directly from the DB.
# Usage:
#   bash scripts/export_benchmark_csv.sh [run_id]
#   bash scripts/export_benchmark_csv.sh          # exports all runs

set -euo pipefail

NAMESPACE="hospital-llm"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_reports}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

if [[ -z "${1:-}" ]]; then
    WHERE_CLAUSE=""
    OUTPUT_FILE="$OUTPUT_DIR/benchmark_all_${TIMESTAMP}.csv"
else
    WHERE_CLAUSE="WHERE run_id = '$1'"
    OUTPUT_FILE="$OUTPUT_DIR/benchmark_${1:0:8}_${TIMESTAMP}.csv"
fi

kubectl exec -n $NAMESPACE postgres-svc-0 -- \
    env PGPASSWORD=postgres_pass psql -U postgres -d hospital -c \
    "\COPY (
        SELECT
            run_id,
            encounter_id,
            model_name,
            ROUND(entity_f1::numeric, 4)                 AS entity_f1,
            diagnosis_exact,
            diagnosis_chapter,
            ROUND(structured_json_score::numeric, 4)      AS structured_json_score,
            ROUND(rouge_l::numeric, 4)                    AS rouge_l,
            ROUND(risk_brier::numeric, 4)                 AS risk_brier,
            latency_ms,
            ttft_ms,
            ROUND(tokens_per_sec::numeric, 2)             AS tokens_per_sec,
            ROUND(json_valid_rate::numeric, 4)            AS json_valid_rate,
            ROUND(function_calling_success::numeric, 4)   AS function_calling_success,
            ROUND(instruction_following_score::numeric, 4) AS instruction_following_score,
            ROUND(hallucination_rate::numeric, 4)          AS hallucination_rate,
            evaluated_at
        FROM benchmark_results
        $WHERE_CLAUSE
        ORDER BY evaluated_at DESC
    ) TO STDOUT WITH CSV HEADER" > "$OUTPUT_FILE"

echo "CSV saved to: $OUTPUT_FILE"
echo "Rows: $(wc -l < "$OUTPUT_FILE")"
