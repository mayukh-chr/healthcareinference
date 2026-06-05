from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    ADMISSION_NOTE = "admission_note"
    LAB_REPORT = "lab_report"
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"


class ClinicalDocument(BaseModel):
    document_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    encounter_id: uuid.UUID
    patient_id: uuid.UUID
    document_type: DocumentType
    authored_at: datetime
    author_name: str
    author_role: str
    content: str

    # document_generator never receives HiddenPatientState:
    # it only receives ClinicalEvent list + LabResult + author_pool
