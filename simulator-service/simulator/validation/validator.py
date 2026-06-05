from __future__ import annotations

from dataclasses import dataclass, field

from simulator.disease_engine.base_disease import LabValueSet
from simulator.models.events import ClinicalEvent, EventType


@dataclass
class ValidationError:
    rule: str
    detail: str


@dataclass
class ValidationReport:
    encounter_id: str
    passed: bool
    errors: list[ValidationError] = field(default_factory=list)


def validate_encounter(
    events: list[ClinicalEvent],
    lab_values: LabValueSet,
    disease: str,
    severity: str,
) -> ValidationReport:
    errors: list[ValidationError] = []
    enc_id = str(events[0].encounter_id) if events else "unknown"

    # --- Temporal ordering ---
    sorted_events = sorted(events, key=lambda e: e.sequence_number)

    if sorted_events[0].event_type != EventType.PATIENT_ADMITTED:
        errors.append(ValidationError("temporal_first_event", "First event must be patient_admitted"))

    if sorted_events[-1].event_type != EventType.DISCHARGE_COMPLETED:
        errors.append(ValidationError("temporal_last_event", "Last event must be discharge_completed"))

    # All events must be in chronological order
    for i in range(1, len(sorted_events)):
        if sorted_events[i].timestamp < sorted_events[i - 1].timestamp:
            errors.append(ValidationError(
                "temporal_chronological",
                f"Event {sorted_events[i].event_type} at seq {i} precedes prior event",
            ))

    # Lab must come after admission
    lab_events = [e for e in sorted_events if e.event_type == EventType.LAB_RESULTED]
    admission_events = [e for e in sorted_events if e.event_type == EventType.PATIENT_ADMITTED]
    if lab_events and admission_events:
        if lab_events[0].timestamp <= admission_events[0].timestamp:
            errors.append(ValidationError("temporal_lab_after_admission", "Lab result must occur after admission"))

    # --- Clinical consistency ---
    ds = lab_values.disease_specific
    inf = lab_values.inflammatory

    if disease == "sepsis" and severity in ("moderate", "severe", "critical"):
        if ds.procalcitonin is not None and ds.procalcitonin < 0.5:
            errors.append(ValidationError(
                "clinical_sepsis_procalcitonin",
                f"Sepsis {severity} must have procalcitonin ≥ 0.5, got {ds.procalcitonin:.2f}",
            ))
        if inf.crp < 50:
            errors.append(ValidationError(
                "clinical_sepsis_crp",
                f"Sepsis {severity} must have CRP ≥ 50 mg/L, got {inf.crp:.1f}",
            ))

    if disease == "diabetes" and severity in ("severe", "critical"):
        bmp = lab_values.bmp
        if bmp.glucose < 250:
            errors.append(ValidationError(
                "clinical_dka_glucose",
                f"DKA must have glucose ≥ 250, got {bmp.glucose:.1f}",
            ))
        if bmp.anion_gap < 12:
            errors.append(ValidationError(
                "clinical_dka_anion_gap",
                f"DKA must have anion gap ≥ 12, got {bmp.anion_gap:.1f}",
            ))

    if disease == "heart_failure" and ds.bnp is not None:
        if severity in ("moderate", "severe", "critical") and ds.bnp < 200:
            errors.append(ValidationError(
                "clinical_hf_bnp",
                f"ADHF {severity} must have BNP ≥ 200, got {ds.bnp:.0f}",
            ))

    if disease == "ckd" and severity in ("severe", "critical"):
        bmp = lab_values.bmp
        if bmp.creatinine < 3.0:
            errors.append(ValidationError(
                "clinical_ckd_creatinine",
                f"CKD {severity} must have creatinine ≥ 3.0, got {bmp.creatinine:.2f}",
            ))

    return ValidationReport(
        encounter_id=enc_id,
        passed=len(errors) == 0,
        errors=errors,
    )
