from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class SmokingStatus(str, Enum):
    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"


class EncounterType(str, Enum):
    INPATIENT = "inpatient"
    ICU = "icu"
    ED = "ed"


class DischargeDisposition(str, Enum):
    HOME = "home"
    SNF = "skilled_nursing_facility"
    DEATH = "death"


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Demographics(BaseModel):
    age: Annotated[int, Field(ge=18, le=100)]
    sex: Sex
    ethnicity: str
    weight_kg: Annotated[float, Field(ge=30.0, le=300.0)]
    height_cm: Annotated[float, Field(ge=100.0, le=220.0)]
    smoking_status: SmokingStatus

    @property
    def bmi(self) -> float:
        return self.weight_kg / ((self.height_cm / 100) ** 2)

    @property
    def pronoun_subject(self) -> str:
        if self.sex == Sex.MALE:
            return "He"
        if self.sex == Sex.FEMALE:
            return "She"
        return "They"

    @property
    def pronoun_object(self) -> str:
        if self.sex == Sex.MALE:
            return "him"
        if self.sex == Sex.FEMALE:
            return "her"
        return "them"


class RiskScores(BaseModel):
    mortality_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    readmission_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_level: RiskLevel

    @model_validator(mode="after")
    def derive_risk_level(self) -> "RiskScores":
        p = self.mortality_probability
        if p >= 0.4:
            object.__setattr__(self, "risk_level", RiskLevel.CRITICAL)
        elif p >= 0.2:
            object.__setattr__(self, "risk_level", RiskLevel.HIGH)
        elif p >= 0.08:
            object.__setattr__(self, "risk_level", RiskLevel.MEDIUM)
        else:
            object.__setattr__(self, "risk_level", RiskLevel.LOW)
        return self


class HiddenPatientState(BaseModel):
    """
    The ground truth. Frozen after creation.
    Never passed to document_generator — enforced at the type level.
    """

    model_config = {"frozen": True}

    patient_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    encounter_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    demographics: Demographics
    chronic_conditions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    primary_disease: str
    severity: Severity
    treatment_plan: list[str] = Field(default_factory=list)
    risk_scores: RiskScores
    discharge_disposition: DischargeDisposition
    encounter_type: EncounterType
    generation_seed: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def icu_requires_severe_or_critical(self) -> "HiddenPatientState":
        if self.encounter_type == EncounterType.ICU and self.severity not in (
            Severity.SEVERE,
            Severity.CRITICAL,
        ):
            raise ValueError("ICU encounter requires severe or critical severity")
        return self
