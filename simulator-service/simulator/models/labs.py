from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CBCPanel(BaseModel):
    wbc: float = Field(description="10³/µL — normal 4.5–11.0")
    rbc: float = Field(description="10⁶/µL — normal 4.5–5.5")
    hemoglobin: float = Field(description="g/dL — normal 13.5–17.5")
    hematocrit: float = Field(description="% — normal 41–53")
    platelets: float = Field(description="10³/µL — normal 150–400")


class BMPPanel(BaseModel):
    sodium: float = Field(description="mEq/L — normal 136–145")
    potassium: float = Field(description="mEq/L — normal 3.5–5.0")
    chloride: float = Field(description="mEq/L — normal 98–106")
    bicarbonate: float = Field(description="mEq/L — normal 22–29")
    bun: float = Field(description="mg/dL — normal 7–20")
    creatinine: float = Field(description="mg/dL — normal 0.6–1.2")
    glucose: float = Field(description="mg/dL — normal 70–100 fasting")
    calcium: float = Field(description="mg/dL — normal 8.5–10.5")

    @property
    def anion_gap(self) -> float:
        return self.sodium - self.chloride - self.bicarbonate


class InflammatoryPanel(BaseModel):
    crp: float = Field(description="mg/L — normal <10")
    esr: float = Field(description="mm/hr — normal <20 male / <30 female")


class DiseaseSpecificPanel(BaseModel):
    glucose: float | None = Field(default=None, description="mg/dL — for diabetes")
    creatinine: float | None = Field(default=None, description="mg/dL — for CKD")
    bnp: float | None = Field(default=None, description="pg/mL — for heart failure")
    lactate: float | None = Field(default=None, description="mmol/L — for sepsis, normal 0.5–2.0")
    procalcitonin: float | None = Field(default=None, description="ng/mL — normal <0.5")
    inr: float | None = Field(default=None, description="— for stroke/anticoagulation")


class LabResult(BaseModel):
    result_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    encounter_id: uuid.UUID
    patient_id: uuid.UUID
    order_event_id: uuid.UUID
    collected_at: datetime
    resulted_at: datetime
    cbc: CBCPanel
    bmp: BMPPanel
    inflammatory: InflammatoryPanel
    disease_specific: DiseaseSpecificPanel
    narrative_report: str
    critical_flags: list[str] = Field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return len(self.critical_flags) > 0
