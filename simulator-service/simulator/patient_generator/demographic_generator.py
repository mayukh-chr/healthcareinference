from __future__ import annotations

import numpy as np
from faker import Faker

from simulator.models.patient import Demographics, Sex, SmokingStatus

_fake = Faker()

# US census-approximate age distribution for hospitalized adults (18–100)
_AGE_WEIGHTS = {
    range(18, 35): 0.10,
    range(35, 50): 0.15,
    range(50, 65): 0.28,
    range(65, 80): 0.32,
    range(80, 101): 0.15,
}

_ETHNICITIES = [
    "White / Non-Hispanic",
    "Black / African American",
    "Hispanic / Latino",
    "Asian",
    "American Indian / Alaska Native",
    "Other / Multiracial",
]
_ETHNICITY_WEIGHTS = [0.60, 0.13, 0.18, 0.06, 0.01, 0.02]

_SMOKING_WEIGHTS = {
    SmokingStatus.NEVER: 0.55,
    SmokingStatus.FORMER: 0.28,
    SmokingStatus.CURRENT: 0.17,
}


def generate_demographics(rng: np.random.Generator) -> Demographics:
    age = _sample_age(rng)
    sex = Sex(str(rng.choice(["male", "female"], p=[0.49, 0.51])))
    ethnicity = str(rng.choice(_ETHNICITIES, p=_ETHNICITY_WEIGHTS))

    # BMI-derived weight/height — US adult distributions
    if sex == Sex.MALE:
        height_cm = float(rng.normal(177.0, 7.5))
        bmi = float(rng.normal(28.5, 6.0))
    else:
        height_cm = float(rng.normal(163.0, 7.0))
        bmi = float(rng.normal(27.5, 6.5))

    height_cm = float(max(100.0, min(220.0, height_cm)))
    bmi = float(max(14.0, min(60.0, bmi)))
    weight_kg = bmi * ((height_cm / 100) ** 2)
    weight_kg = float(max(30.0, min(300.0, weight_kg)))

    return Demographics(
        age=age,
        sex=sex,
        ethnicity=str(ethnicity),
        weight_kg=round(weight_kg, 1),
        height_cm=round(height_cm, 1),
        smoking_status=SmokingStatus(str(rng.choice(
            [s.value for s in SmokingStatus],
            p=list(_SMOKING_WEIGHTS.values()),
        ))),
    )


def _sample_age(rng: np.random.Generator) -> int:
    buckets = list(_AGE_WEIGHTS.keys())
    weights = list(_AGE_WEIGHTS.values())
    chosen = buckets[int(rng.choice(len(buckets), p=weights))]
    return int(rng.integers(chosen.start, chosen.stop))
