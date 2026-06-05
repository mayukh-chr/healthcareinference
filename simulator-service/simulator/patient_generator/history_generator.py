from __future__ import annotations

import numpy as np

from simulator.models.patient import Demographics, Sex

# Comorbidities correlated with primary disease
_DISEASE_COMORBIDITIES: dict[str, list[tuple[str, float]]] = {
    "sepsis": [
        ("Diabetes mellitus", 0.35),
        ("Chronic kidney disease", 0.28),
        ("Immunosuppression", 0.20),
        ("Malignancy", 0.18),
        ("Liver cirrhosis", 0.12),
        ("COPD", 0.15),
    ],
    "pneumonia": [
        ("COPD", 0.30),
        ("Asthma", 0.20),
        ("Diabetes mellitus", 0.25),
        ("Heart failure", 0.22),
        ("Immunosuppression", 0.15),
        ("Smoking history", 0.40),
    ],
    "heart_failure": [
        ("Hypertension", 0.75),
        ("Coronary artery disease", 0.55),
        ("Diabetes mellitus", 0.38),
        ("Atrial fibrillation", 0.42),
        ("Chronic kidney disease", 0.30),
        ("Hyperlipidemia", 0.60),
    ],
    "diabetes": [
        ("Hypertension", 0.65),
        ("Hyperlipidemia", 0.58),
        ("Obesity", 0.55),
        ("Coronary artery disease", 0.28),
        ("Peripheral neuropathy", 0.30),
        ("Chronic kidney disease", 0.22),
    ],
    "ckd": [
        ("Hypertension", 0.80),
        ("Diabetes mellitus", 0.60),
        ("Anemia", 0.55),
        ("Hyperphosphatemia", 0.45),
        ("Cardiovascular disease", 0.35),
        ("Hyperkalemia", 0.30),
    ],
    "stroke": [
        ("Hypertension", 0.78),
        ("Atrial fibrillation", 0.38),
        ("Diabetes mellitus", 0.30),
        ("Hyperlipidemia", 0.55),
        ("Coronary artery disease", 0.28),
        ("Prior TIA or stroke", 0.20),
    ],
}

_COMMON_ALLERGIES = [
    "Penicillin",
    "Sulfa drugs",
    "Aspirin",
    "NSAIDs",
    "Morphine",
    "Codeine",
    "Latex",
    "Contrast dye",
]


def generate_history(
    rng: np.random.Generator,
    primary_disease: str,
    demographics: Demographics,
) -> tuple[list[str], list[str]]:
    """Returns (chronic_conditions, allergies)."""
    comorbidity_pool = _DISEASE_COMORBIDITIES.get(primary_disease, [])

    # Age-correlated extra comorbidities
    chronic_conditions: list[str] = []
    for condition, base_prob in comorbidity_pool:
        # Older patients more likely to have comorbidities
        age_factor = min(1.8, 1.0 + (demographics.age - 50) * 0.01)
        adjusted_prob = min(0.95, base_prob * age_factor)
        if rng.random() < adjusted_prob:
            chronic_conditions.append(condition)

    # Allergies: ~25% of patients report at least one
    allergies: list[str] = []
    if rng.random() < 0.25:
        n_allergies = int(rng.integers(1, 3))
        chosen = rng.choice(_COMMON_ALLERGIES, size=min(n_allergies, len(_COMMON_ALLERGIES)), replace=False)
        allergies = [str(a) for a in chosen]

    return chronic_conditions, allergies
