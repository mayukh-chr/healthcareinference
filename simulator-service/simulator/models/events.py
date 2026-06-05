from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    PATIENT_ADMITTED = "patient_admitted"
    LAB_RESULTED = "lab_resulted"
    PROGRESS_NOTE_CREATED = "progress_note_created"
    TREATMENT_UPDATED = "treatment_updated"
    DISCHARGE_COMPLETED = "discharge_completed"


class ClinicalEvent(BaseModel):
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    patient_id: uuid.UUID
    encounter_id: uuid.UUID
    timestamp: datetime
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence_number: int = Field(ge=0, le=4)
    causal_event_id: uuid.UUID | None = None

    def routing_key(self) -> str:
        mapping = {
            EventType.PATIENT_ADMITTED: "hospital.events.admission",
            EventType.LAB_RESULTED: "hospital.events.lab",
            EventType.PROGRESS_NOTE_CREATED: "hospital.events.note",
            EventType.TREATMENT_UPDATED: "hospital.events.treatment",
            EventType.DISCHARGE_COMPLETED: "hospital.events.discharge",
        }
        return mapping[self.event_type]
