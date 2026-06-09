from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI, Query
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from benchmark.evaluator import evaluate_encounter

logger = structlog.get_logger()

# g6e.xlarge on-demand price in us-east-1 (USD/hr) — update if region differs
_GPU_COST_USD_PER_HOUR = 2.264
_SECONDS_PER_HOUR = 3600


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    database_url: str = "postgresql+asyncpg://benchmark_user:benchmark_pass@localhost:5432/hospital"
    service_name: str = "hospital-benchmark"


settings = Settings()
engine = create_async_engine(settings.database_url)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

app = FastAPI(title="Hospital Benchmark Service", version="0.1.0")
Instrumentator().instrument(app).expose(app)


class RunResponse(BaseModel):
    run_id: str
    n_encounters: int
    message: str


@app.post("/v1/run", response_model=RunResponse)
async def run_benchmark(
    model_name: str = Query(default="Qwen/Qwen3.5-9B"),
    limit: int = Query(default=100, ge=1, le=10_000),
) -> RunResponse:
    run_id = str(uuid.uuid4())
    async with SessionLocal() as session:
        rows = await session.execute(
            text("""
                SELECT
                    ir.encounter_id,
                    ir.task,
                    ir.inferred_output,
                    ir.latency_ms,
                    ir.ttft_ms,
                    ir.itl_ms,
                    ir.completion_tokens,
                    gt.primary_icd10,
                    gt.mortality_risk,
                    gt.readmission_risk,
                    gt.expected_structured_output,
                    gt.expected_summary
                FROM inference_results ir
                JOIN ground_truth_labels gt ON gt.encounter_id = ir.encounter_id
                WHERE ir.model_name = :model_name
                  AND ir.encounter_id IN (
                      SELECT encounter_id FROM (
                          SELECT DISTINCT encounter_id, MAX(created_at) AS last_seen
                          FROM inference_results
                          WHERE model_name = :model_name
                          GROUP BY encounter_id
                          ORDER BY last_seen DESC
                          LIMIT :limit
                      ) sub
                  )
                ORDER BY ir.encounter_id, ir.task, ir.created_at DESC
            """),
            {"model_name": model_name, "limit": limit},
        )
        data = rows.fetchall()

    by_encounter: dict[str, dict] = {}
    for row in data:
        eid = str(row.encounter_id)
        if eid not in by_encounter:
            by_encounter[eid] = {
                "tasks": {},
                "ground_truth": {
                    "primary_icd10": row.primary_icd10,
                    "mortality_risk": float(row.mortality_risk),
                    "readmission_risk": float(row.readmission_risk),
                    "expected_structured_output": row.expected_structured_output,
                    "expected_summary": row.expected_summary,
                },
                "latency_ms": row.latency_ms,
                "ttft_ms": row.ttft_ms,
                "itl_ms": row.itl_ms,
                "completion_tokens": row.completion_tokens,
            }
        by_encounter[eid]["tasks"][row.task] = row.inferred_output

    async with SessionLocal() as session:
        for encounter_id, enc_data in by_encounter.items():
            result = evaluate_encounter(
                run_id=run_id,
                encounter_id=encounter_id,
                model_name=model_name,
                inference_results=enc_data["tasks"],
                ground_truth=enc_data["ground_truth"],
                latency_ms=enc_data.get("latency_ms"),
                ttft_ms=enc_data.get("ttft_ms"),
                itl_ms=enc_data.get("itl_ms"),
                completion_tokens=enc_data.get("completion_tokens"),
            )
            await session.execute(
                text("""
                    INSERT INTO benchmark_results
                        (result_id, run_id, encounter_id, model_name,
                         entity_f1, diagnosis_exact, diagnosis_chapter,
                         structured_json_score, rouge_l, risk_brier,
                         latency_ms, ttft_ms, itl_ms, tokens_per_sec,
                         json_valid_rate, function_calling_success,
                         instruction_following_score, hallucination_rate)
                    VALUES
                        (:result_id, :run_id, :encounter_id, :model_name,
                         :entity_f1, :diagnosis_exact, :diagnosis_chapter,
                         :structured_json_score, :rouge_l, :risk_brier,
                         :latency_ms, :ttft_ms, :itl_ms, :tokens_per_sec,
                         :json_valid_rate, :function_calling_success,
                         :instruction_following_score, :hallucination_rate)
                """),
                {
                    "result_id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "encounter_id": encounter_id,
                    "model_name": model_name,
                    "entity_f1": result.entity_f1,
                    "diagnosis_exact": result.diagnosis_exact,
                    "diagnosis_chapter": result.diagnosis_chapter,
                    "structured_json_score": result.structured_json_score,
                    "rouge_l": result.rouge_l,
                    "risk_brier": result.risk_brier,
                    "latency_ms": result.latency_ms,
                    "ttft_ms": result.ttft_ms,
                    "itl_ms": result.itl_ms,
                    "tokens_per_sec": result.tokens_per_sec,
                    "json_valid_rate": result.json_valid_rate,
                    "function_calling_success": result.function_calling_success,
                    "instruction_following_score": result.instruction_following_score,
                    "hallucination_rate": result.hallucination_rate,
                },
            )
        await session.commit()

    logger.info("benchmark_run_completed", run_id=run_id, n=len(by_encounter))
    return RunResponse(run_id=run_id, n_encounters=len(by_encounter), message="Benchmark run complete")


@app.get("/v1/results")
async def get_results(run_id: str = Query(...)) -> dict:
    async with SessionLocal() as session:
        rows = await session.execute(
            text("""
                SELECT
                    AVG(entity_f1)                                                  AS avg_entity_f1,
                    AVG(CASE WHEN diagnosis_exact THEN 1.0 ELSE 0.0 END)           AS diagnosis_exact_rate,
                    AVG(CASE WHEN diagnosis_chapter THEN 1.0 ELSE 0.0 END)         AS diagnosis_chapter_rate,
                    AVG(structured_json_score)                                      AS avg_structured_json,
                    AVG(rouge_l)                                                    AS avg_rouge_l,
                    AVG(risk_brier)                                                 AS avg_risk_brier,
                    AVG(json_valid_rate)                                            AS avg_json_valid_rate,
                    AVG(function_calling_success)                                   AS avg_function_calling_success,
                    AVG(instruction_following_score)                                AS avg_instruction_following,
                    AVG(hallucination_rate)                                         AS avg_hallucination_rate,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)       AS p50_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)       AS p95_latency_ms,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)       AS p99_latency_ms,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ttft_ms)          AS p50_ttft_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ttft_ms)          AS p95_ttft_ms,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY itl_ms)           AS p50_itl_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY itl_ms)           AS p95_itl_ms,
                    AVG(tokens_per_sec)                                             AS avg_tokens_per_sec,
                    COUNT(*)                                                        AS n_encounters
                FROM benchmark_results
                WHERE run_id = :run_id
            """),
            {"run_id": run_id},
        )
        row = rows.fetchone()

    if not row:
        return {"error": "run_id not found"}

    avg_tps = float(row.avg_tokens_per_sec) if row.avg_tokens_per_sec else 0.0
    cost_per_1m = (
        (_GPU_COST_USD_PER_HOUR / _SECONDS_PER_HOUR) / (avg_tps / 1_000_000)
        if avg_tps > 0
        else None
    )

    return {
        "run_id": run_id,
        "n_encounters": row.n_encounters,
        "llm_quality": {
            "json_valid_rate": row.avg_json_valid_rate,
            "function_calling_success": row.avg_function_calling_success,
            "instruction_following": row.avg_instruction_following,
            "structured_json_score": row.avg_structured_json,
            "entity_f1": row.avg_entity_f1,
            "diagnosis_exact_rate": row.diagnosis_exact_rate,
            "diagnosis_chapter_rate": row.diagnosis_chapter_rate,
            "rouge_l": row.avg_rouge_l,
            "risk_brier": row.avg_risk_brier,
            "hallucination_rate": row.avg_hallucination_rate,
        },
        "infra_performance": {
            "p50_latency_ms": row.p50_latency_ms,
            "p95_latency_ms": row.p95_latency_ms,
            "p99_latency_ms": row.p99_latency_ms,
            "p50_ttft_ms": row.p50_ttft_ms,
            "p95_ttft_ms": row.p95_ttft_ms,
            "p50_itl_ms": row.p50_itl_ms,
            "p95_itl_ms": row.p95_itl_ms,
            "avg_tokens_per_sec": avg_tps,
            "cost_per_1m_tokens_usd": cost_per_1m,
        },
    }


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}
