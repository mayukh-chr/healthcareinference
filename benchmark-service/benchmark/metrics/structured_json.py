from __future__ import annotations


_FIELDS = [
    "primary_diagnosis",
    "primary_icd10",
    "disease_severity",
    "risk_level",
    "discharge_disposition",
]


def evaluate(inferred: dict, ground_truth: dict) -> dict[str, float | None]:
    gt = ground_truth.get("expected_structured_output", {})
    if not gt:
        return {"structured_json_score": None}

    correct = 0
    total = len(_FIELDS)

    for field in _FIELDS:
        inferred_val = str(inferred.get(field) or "").strip().lower()
        gt_val = str(gt.get(field, "")).strip().lower()
        if inferred_val and gt_val and inferred_val == gt_val:
            correct += 1

    score = correct / total if total > 0 else 0.0
    return {"structured_json_score": round(score, 4)}
