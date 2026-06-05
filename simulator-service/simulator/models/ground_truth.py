from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from simulator.models.patient import DischargeDisposition, RiskLevel, Severity


class ExpectedStructuredOutput(BaseModel):
    """
    The JSON an LLM should produce when given the clinical documents.
    Generated deterministically from HiddenPatientState — never from documents.
    """

    primary_diagnosis: str
    primary_icd10: str
    secondary_diagnoses: list[str] = Field(default_factory=list)
    disease_severity: Severity
    risk_level: RiskLevel
    medications: list[str] = Field(default_factory=list)
    key_lab_findings: dict[str, str] = Field(
        default_factory=dict,
        description="{'WBC': 'elevated', 'Creatinine': 'elevated'}",
    )
    discharge_disposition: DischargeDisposition


class GroundTruthLabels(BaseModel):
    label_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    encounter_id: uuid.UUID
    patient_id: uuid.UUID
    primary_diagnosis: str
    primary_icd10: str
    secondary_diagnoses: list[str] = Field(default_factory=list)
    disease_severity: Severity
    mortality_risk: float = Field(ge=0.0, le=1.0)
    readmission_risk: float = Field(ge=0.0, le=1.0)
    expected_structured_output: ExpectedStructuredOutput
    expected_summary: str = Field(description="2–3 sentence gold summary")
    created_at: datetime = Field(default_factory=datetime.utcnow)
