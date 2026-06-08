from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.metrics import (
    diagnosis_accuracy,
    entity_extraction,
    risk_classification,
    structured_json,
    summarization,
)
from benchmark.metrics.llm_quality import evaluate_llm_quality


@dataclass
class EncounterBenchmarkResult:
    encounter_id: str
    run_id: str
    model_name: str
    # Existing quality metrics
    entity_f1: float | None = None
    diagnosis_exact: bool | None = None
    diagnosis_chapter: bool | None = None
    structured_json_score: float | None = None
    rouge_l: float | None = None
    risk_brier: float | None = None
    # Performance
    latency_ms: int | None = None
    ttft_ms: int | None = None
    itl_ms: float | None = None
    tokens_per_sec: float | None = None
    # LLM quality layer
    json_valid_rate: float | None = None
    function_calling_success: float | None = None
    instruction_following_score: float | None = None
    hallucination_rate: float | None = None


def evaluate_encounter(
    run_id: str,
    encounter_id: str,
    model_name: str,
    inference_results: dict[str, dict],
    ground_truth: dict,
    latency_ms: int | None = None,
    ttft_ms: int | None = None,
    itl_ms: float | None = None,
    completion_tokens: int | None = None,
) -> EncounterBenchmarkResult:
    result = EncounterBenchmarkResult(
        encounter_id=encounter_id,
        run_id=run_id,
        model_name=model_name,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
    )

    if latency_ms and completion_tokens:
        result.tokens_per_sec = round(completion_tokens / (latency_ms / 1000), 2) if latency_ms > 0 else None

    if "diagnosis" in inference_results:
        diag = diagnosis_accuracy.evaluate(inference_results["diagnosis"], ground_truth)
        result.diagnosis_exact = diag.get("diagnosis_exact")
        result.diagnosis_chapter = diag.get("diagnosis_chapter")

    if "entity_extraction" in inference_results:
        ent = entity_extraction.evaluate(inference_results["entity_extraction"], ground_truth)
        result.entity_f1 = ent.get("entity_f1")

    if "structured_json" in inference_results:
        sj = structured_json.evaluate(inference_results["structured_json"], ground_truth)
        result.structured_json_score = sj.get("structured_json_score")

    if "summarization" in inference_results:
        sm = summarization.evaluate(inference_results["summarization"], ground_truth)
        result.rouge_l = sm.get("rouge_l")

    if "risk_classification" in inference_results:
        rc = risk_classification.evaluate(inference_results["risk_classification"], ground_truth)
        result.risk_brier = rc.get("risk_brier")

    quality = evaluate_llm_quality(inference_results, ground_truth)
    result.json_valid_rate = quality["json_valid_rate"]
    result.function_calling_success = quality["function_calling_success"]
    result.instruction_following_score = quality["instruction_following_score"]
    result.hallucination_rate = quality["hallucination_rate"]

    return result
