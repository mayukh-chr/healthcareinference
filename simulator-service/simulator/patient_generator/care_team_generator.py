from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faker import Faker

_fake = Faker()

_SPECIALTIES: dict[str, str] = {
    "sepsis": "Infectious Disease",
    "pneumonia": "Pulmonology",
    "heart_failure": "Cardiology",
    "diabetes": "Endocrinology",
    "ckd": "Nephrology",
    "stroke": "Neurology",
}

_SPECIALIST_ROLES: dict[str, str] = {
    "sepsis": "Infectious Disease Fellow",
    "pneumonia": "Pulmonology Fellow",
    "heart_failure": "Cardiology Fellow",
    "diabetes": "Endocrinology Fellow",
    "ckd": "Nephrology Fellow",
    "stroke": "Neurology Fellow",
}


@dataclass
class Provider:
    name: str
    role: str
    department: str
    shift: str | None  # None for attendings (no fixed shift)


@dataclass
class CareTeam:
    attending: Provider
    primary_resident: Provider    # authors the admission note
    covering_resident: Provider   # authors the progress note (different shift)
    specialist: Provider | None   # consulting service, if applicable


def generate_care_team(
    rng: np.random.Generator,
    disease: str,
    severity: str,
    encounter_type: str,
) -> CareTeam:
    Faker.seed(int(rng.integers(0, 2**31)))

    department = "Internal Medicine"
    attending = Provider(
        name=_fake.name(),
        role="Attending Physician",
        department=department,
        shift=None,
    )

    pgy = 2 if severity in ("mild", "moderate") else 3
    primary_resident = Provider(
        name=_fake.name(),
        role=f"Medical Resident PGY-{pgy}",
        department=department,
        shift="Day (07:00–19:00)",
    )

    # Covering resident: always a different person on a different shift
    covering_pgy = 2 if encounter_type != "icu" else 3
    covering_resident = Provider(
        name=_fake.name(),
        role=f"Medical Resident PGY-{covering_pgy}",
        department=department,
        shift="Night (19:00–07:00)",
    )

    # Specialist consult for moderate-critical cases
    specialist: Provider | None = None
    if severity in ("moderate", "severe", "critical"):
        specialist = Provider(
            name=_fake.name(),
            role=_SPECIALIST_ROLES.get(disease, "Internal Medicine Fellow"),
            department=_SPECIALTIES.get(disease, "Internal Medicine"),
            shift=None,
        )

    return CareTeam(
        attending=attending,
        primary_resident=primary_resident,
        covering_resident=covering_resident,
        specialist=specialist,
    )
