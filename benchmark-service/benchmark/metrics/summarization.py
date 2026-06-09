from __future__ import annotations

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def evaluate(inferred: dict, ground_truth: dict) -> dict[str, float | None]:
    inferred_summary = (inferred.get("summary") or "").strip()
    expected_summary = ground_truth.get("expected_summary", "").strip()

    if not inferred_summary or not expected_summary:
        return {"rouge_l": None}

    scores = _scorer.score(expected_summary, inferred_summary)
    rouge_l = scores["rougeL"].fmeasure

    return {"rouge_l": round(rouge_l, 4)}
