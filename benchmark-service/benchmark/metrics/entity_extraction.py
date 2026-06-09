from __future__ import annotations


def _tokenize(text: str) -> set[str]:
    return {t.lower().strip(".,;:()") for t in text.split() if len(t) > 2}


def _list_f1(inferred: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0 if not inferred else 0.0

    inferred_tokens: set[str] = set()
    for item in inferred:
        inferred_tokens |= _tokenize(item)

    expected_tokens: set[str] = set()
    for item in expected:
        expected_tokens |= _tokenize(item)

    if not inferred_tokens:
        return 0.0

    tp = len(inferred_tokens & expected_tokens)
    precision = tp / len(inferred_tokens)
    recall = tp / len(expected_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(inferred: dict, ground_truth: dict) -> dict[str, float | None]:
    gt_output = ground_truth.get("expected_structured_output", {})

    medication_f1 = _list_f1(
        inferred.get("medications") or [],
        gt_output.get("medications") or [],
    )

    # Lab findings: compare keys
    inferred_labs = list(inferred.get("lab_findings") or [])
    expected_labs = [f"{k}: {v}" for k, v in gt_output.get("key_lab_findings", {}).items()]
    lab_f1 = _list_f1(inferred_labs, expected_labs)

    entity_f1 = (medication_f1 + lab_f1) / 2.0

    return {"entity_f1": round(entity_f1, 4)}
