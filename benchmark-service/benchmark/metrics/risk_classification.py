from __future__ import annotations


def _brier(predicted: float, actual: float) -> float:
    return (predicted - actual) ** 2


def evaluate(inferred: dict, ground_truth: dict) -> dict[str, float | None]:
    try:
        pred_mortality = float(inferred.get("mortality_risk", 0.0))
        pred_readmission = float(inferred.get("readmission_risk", 0.0))
    except (TypeError, ValueError):
        return {"risk_brier": None}

    actual_mortality = float(ground_truth.get("mortality_risk", 0.0))
    actual_readmission = float(ground_truth.get("readmission_risk", 0.0))

    brier = (_brier(pred_mortality, actual_mortality) + _brier(pred_readmission, actual_readmission)) / 2.0

    return {"risk_brier": round(brier, 4)}
