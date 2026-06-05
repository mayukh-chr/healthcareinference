from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class DiagnosisOutput(BaseModel):
    primary_diagnosis: str
    primary_icd10: str
    secondary_diagnoses: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class EntityOutput(BaseModel):
    diagnoses: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    lab_findings: list[str] = Field(default_factory=list)


class StructuredJsonOutput(BaseModel):
    primary_diagnosis: str
    primary_icd10: str
    secondary_diagnoses: list[str] = Field(default_factory=list)
    disease_severity: str
    risk_level: str
    medications: list[str] = Field(default_factory=list)
    key_lab_findings: dict[str, str] = Field(default_factory=dict)
    discharge_disposition: str


class SummaryOutput(BaseModel):
    summary: str


class RiskOutput(BaseModel):
    mortality_risk: float = Field(ge=0.0, le=1.0)
    readmission_risk: float = Field(ge=0.0, le=1.0)
    risk_level: str


TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "diagnosis": DiagnosisOutput,
    "entity_extraction": EntityOutput,
    "structured_json": StructuredJsonOutput,
    "summarization": SummaryOutput,
    "risk_classification": RiskOutput,
}
