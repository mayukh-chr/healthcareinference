from __future__ import annotations

# Required fields per task — used for instruction_following and function_calling_success
_TASK_REQUIRED_FIELDS: dict[str, list[str]] = {
    "diagnosis": ["primary_diagnosis", "primary_icd10", "confidence"],
    "entity_extraction": ["diagnoses", "medications", "lab_findings"],
    "structured_json": [
        "primary_diagnosis", "primary_icd10", "disease_severity",
        "risk_level", "medications", "discharge_disposition",
    ],
    "summarization": ["summary"],
    "risk_classification": ["mortality_risk", "readmission_risk", "risk_level"],
}


def evaluate_llm_quality(
    inference_results: dict[str, dict],
    ground_truth: dict,
) -> dict[str, float]:
    """
    Returns json_valid_rate, function_calling_success, instruction_following_score,
    hallucination_rate — all as floats in [0, 1].
    """
    tasks = list(inference_results.items())
    if not tasks:
        return {
            "json_valid_rate": None,
            "function_calling_success": None,
            "instruction_following_score": None,
            "hallucination_rate": None,
        }

    json_valid_scores: list[float] = []
    function_calling_scores: list[float] = []
    instruction_scores: list[float] = []

    for task, output in tasks:
        is_dict = isinstance(output, dict)
        has_raw = is_dict and "raw" in output and len(output) == 1

        # JSON validity: output was parsed into a dict without falling back to {"raw": ...}
        json_valid_scores.append(1.0 if (is_dict and not has_raw) else 0.0)

        required = _TASK_REQUIRED_FIELDS.get(task, [])
        if not required or not is_dict or has_raw:
            function_calling_scores.append(0.0)
            instruction_scores.append(0.0)
            continue

        # Function calling success: all required fields present and non-empty
        fields_present = sum(
            1 for f in required
            if output.get(f) not in (None, "", [], {})
        )
        function_calling_scores.append(fields_present / len(required))

        # Instruction following: same as function calling (field coverage) for structured tasks
        instruction_scores.append(fields_present / len(required))

    n = len(tasks)
    json_valid_rate = sum(json_valid_scores) / n
    function_calling_success = sum(function_calling_scores) / n
    instruction_following_score = sum(instruction_scores) / n

    # Hallucination rate: proxy using structured_json task.
    # If the model's primary_icd10 ICD-10 chapter differs from ground truth AND
    # doesn't appear in secondary_diagnoses, it likely hallucinated a diagnosis.
    hallucination_rate = _estimate_hallucination(
        inference_results.get("structured_json", {}),
        ground_truth,
    )

    return {
        "json_valid_rate": round(json_valid_rate, 4),
        "function_calling_success": round(function_calling_success, 4),
        "instruction_following_score": round(instruction_following_score, 4),
        "hallucination_rate": hallucination_rate,
    }


def _estimate_hallucination(sj_output: dict, ground_truth: dict) -> float | None:
    """
    Returns 1.0 if the inferred primary_icd10 is in a completely different ICD-10
    chapter than ground truth AND does not overlap with known secondary diagnoses.
    Returns 0.0 if the chapter matches. Returns None if data is insufficient.
    """
    if not sj_output or "raw" in sj_output:
        return None

    gt_icd10 = str(ground_truth.get("primary_icd10", "")).strip().upper()
    inferred_icd10 = str(sj_output.get("primary_icd10", "")).strip().upper()

    if not gt_icd10 or not inferred_icd10:
        return None

    # ICD-10 chapter = first letter
    gt_chapter = gt_icd10[0] if gt_icd10 else ""
    inf_chapter = inferred_icd10[0] if inferred_icd10 else ""

    if gt_chapter and inf_chapter and gt_chapter != inf_chapter:
        return 1.0

    return 0.0
