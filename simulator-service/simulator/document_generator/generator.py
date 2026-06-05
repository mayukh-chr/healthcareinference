from __future__ import annotations

from datetime import datetime, timedelta

from faker import Faker

from simulator.document_generator.template_engine import render
from simulator.disease_engine.base_disease import LabValueSet, NoteFragments
from simulator.models.documents import ClinicalDocument, DocumentType
from simulator.models.events import ClinicalEvent, EventType
from simulator.models.labs import LabResult
from simulator.patient_generator.care_team_generator import CareTeam

_CHIEF_COMPLAINTS: dict[str, str] = {
    "sepsis": "Fever and chills — I've been feeling really unwell for the past two days",
    "pneumonia": "Cough and trouble breathing",
    "heart_failure": "My legs are so swollen and I can't breathe when I lie down",
    "diabetes": "My blood sugars have been really high and I just feel awful",
    "ckd": "I've been feeling weak and nauseous, not myself at all",
    "stroke": "I suddenly couldn't move my arm and my speech got garbled",
}


def generate_all_documents(
    events: list[ClinicalEvent],
    lab_result: LabResult,
    followup_lab_result: LabResult,
    note_fragments: NoteFragments,
    lab_values: LabValueSet,
    disease: str,
    severity: str,
    demographics: dict,
    chronic_conditions: list[str],
    allergies: list[str],
    discharge_disposition: str,
    encounter_type: str,
    primary_diagnosis: str,
    primary_icd10: str,
    care_team: CareTeam,
) -> list[ClinicalDocument]:
    """
    Generates all 4 clinical documents from events + lab data.
    Does NOT receive HiddenPatientState — structural enforcement.
    """
    patient_id = str(events[0].patient_id)
    encounter_id = events[0].encounter_id
    service = care_team.specialist.department if care_team.specialist else care_team.attending.department

    admission_event = next(e for e in events if e.event_type == EventType.PATIENT_ADMITTED)
    discharge_event = next(e for e in events if e.event_type == EventType.DISCHARGE_COMPLETED)
    los_days = (discharge_event.timestamp - admission_event.timestamp).days

    docs: list[ClinicalDocument] = []

    # 1. Admission Note — authored by primary resident, cosigned by attending
    admission_ctx = {
        "admitted_at": admission_event.timestamp.strftime("%Y-%m-%d %H:%M"),
        "author_name": care_team.primary_resident.name,
        "author_role": care_team.primary_resident.role,
        "author_shift": care_team.primary_resident.shift,
        "cosign_name": care_team.attending.name,
        "service": service,
        "specialist": care_team.specialist,
        "patient_id": patient_id,
        "chief_complaint": _CHIEF_COMPLAINTS.get(disease, "Acute illness"),
        "hpi": note_fragments.hpi,
        "chronic_conditions": chronic_conditions,
        "allergies": allergies,
        "smoking_status": demographics.get("smoking_status", "never"),
        "severity": severity,
        "primary_diagnosis": primary_diagnosis,
        "plan": note_fragments.plan,
        "secondary_diagnoses": note_fragments.secondary_diagnoses,
        "encounter_type": encounter_type,
        "interventions": note_fragments.interventions,
    }
    docs.append(ClinicalDocument(
        encounter_id=encounter_id,
        patient_id=events[0].patient_id,
        document_type=DocumentType.ADMISSION_NOTE,
        authored_at=admission_event.timestamp,
        author_name=care_team.primary_resident.name,
        author_role=care_team.primary_resident.role,
        content=render("admission_note.j2", admission_ctx),
    ))

    # 2. Lab Report — authored by clinical laboratory (unchanged)
    lab_narrative = _build_lab_narrative(lab_values, disease, severity)
    lab_ctx = {
        "resulted_at": lab_result.resulted_at.strftime("%Y-%m-%d %H:%M"),
        "patient_id": patient_id,
        "author_name": care_team.primary_resident.name,
        "critical_flags": lab_result.critical_flags,
        "cbc": lab_result.cbc,
        "bmp": lab_result.bmp,
        "inflammatory": lab_result.inflammatory,
        "disease_specific": lab_result.disease_specific,
        "narrative": lab_narrative,
    }
    docs.append(ClinicalDocument(
        encounter_id=encounter_id,
        patient_id=events[0].patient_id,
        document_type=DocumentType.LAB_REPORT,
        authored_at=lab_result.resulted_at,
        author_name="Clinical Laboratory",
        author_role="Laboratory Technician",
        content=render("lab_report.j2", lab_ctx),
    ))

    # 3. Progress Note — authored by covering resident (different shift from admission)
    note_event = next(e for e in events if e.event_type == EventType.PROGRESS_NOTE_CREATED)
    hospital_day = (note_event.timestamp - admission_event.timestamp).days + 1
    overnight_map = {
        "mild": "No acute events overnight. Patient resting comfortably.",
        "moderate": "Patient had one episode of fever to 38.5°C. Responded to acetaminophen.",
        "severe": "Hemodynamic instability noted at 0200. IV fluids bolused with transient improvement.",
        "critical": "Multiple critical events overnight. Vasopressor requirements increased.",
    }
    progress_ctx = {
        "note_date": note_event.timestamp.strftime("%Y-%m-%d %H:%M"),
        "author_name": care_team.covering_resident.name,
        "author_role": care_team.covering_resident.role,
        "author_shift": care_team.covering_resident.shift,
        "attending_name": care_team.attending.name,
        "service": service,
        "patient_id": patient_id,
        "hospital_day": hospital_day,
        "severity": severity,
        "chief_complaint_short": _CHIEF_COMPLAINTS.get(disease, "illness"),
        "overnight_events": overnight_map.get(severity, "No acute events."),
        "cbc": followup_lab_result.cbc,
        "bmp": followup_lab_result.bmp,
        "inflammatory": followup_lab_result.inflammatory,
        "disease_specific": followup_lab_result.disease_specific,
        "admission_cbc": lab_result.cbc,
        "admission_bmp": lab_result.bmp,
        "assessment": note_fragments.assessment,
        "plan": note_fragments.plan,
        "medications": note_fragments.medications,
        "interventions": note_fragments.interventions,
    }
    docs.append(ClinicalDocument(
        encounter_id=encounter_id,
        patient_id=events[0].patient_id,
        document_type=DocumentType.PROGRESS_NOTE,
        authored_at=note_event.timestamp,
        author_name=care_team.covering_resident.name,
        author_role=care_team.covering_resident.role,
        content=render("progress_note.j2", progress_ctx),
    ))

    # 4. Discharge Summary — authored by attending
    course_map = {
        "mild": (
            f"Patient was admitted with {_CHIEF_COMPLAINTS.get(disease, 'acute illness')} "
            f"and responded well to treatment with {note_fragments.medications[0] if note_fragments.medications else 'standard therapy'}. "
            f"Symptoms improved significantly over {los_days} days. Discharged in stable condition."
        ),
        "moderate": (
            f"Patient admitted for {primary_diagnosis}. Required IV therapy and close monitoring. "
            f"Gradual improvement noted over {los_days} days. Transitioned to oral regimen prior to discharge."
        ),
        "severe": (
            f"Patient had a complex {los_days}-day hospitalization for {primary_diagnosis}. "
            f"Required intensive medical management. Improvement was slow but steady. "
            f"Discharged with close outpatient follow-up arranged."
        ),
        "critical": (
            f"Patient required critical care-level management over {los_days} days for {primary_diagnosis}. "
            f"Multiple organ systems involved. Stabilized with aggressive supportive care. "
            f"Disposition: {discharge_disposition}."
        ),
    }
    discharge_ctx = {
        "discharge_date": discharge_event.timestamp.strftime("%Y-%m-%d %H:%M"),
        "author_name": care_team.attending.name,
        "author_role": care_team.attending.role,
        "service": service,
        "specialist": care_team.specialist,
        "patient_id": patient_id,
        "admitted_at": admission_event.timestamp.strftime("%Y-%m-%d"),
        "los_days": los_days,
        "primary_diagnosis": primary_diagnosis,
        "primary_icd10": primary_icd10,
        "severity": severity,
        "secondary_diagnoses": note_fragments.secondary_diagnoses,
        "hospital_course": course_map.get(severity, "Clinical course as documented."),
        "procedures": [],
        "cbc": lab_result.cbc,
        "bmp": lab_result.bmp,
        "inflammatory": lab_result.inflammatory,
        "disease_specific": lab_result.disease_specific,
        "discharge_cbc": followup_lab_result.cbc,
        "discharge_bmp": followup_lab_result.bmp,
        "discharge_disposition": discharge_disposition,
        "discharge_medications": note_fragments.medications,
        "followup_days": 7 if severity in ("mild", "moderate") else 3,
        "specialist_followup": service,
    }
    docs.append(ClinicalDocument(
        encounter_id=encounter_id,
        patient_id=events[0].patient_id,
        document_type=DocumentType.DISCHARGE_SUMMARY,
        authored_at=discharge_event.timestamp,
        author_name=care_team.attending.name,
        author_role=care_team.attending.role,
        content=render("discharge_summary.j2", discharge_ctx),
    ))

    return docs


def _build_lab_narrative(lab_values: LabValueSet, disease: str, severity: str) -> str:
    flags = lab_values.critical_flags
    ds = lab_values.disease_specific
    cbc = lab_values.cbc
    bmp = lab_values.bmp
    inf = lab_values.inflammatory

    lines = [f"Laboratory results consistent with {severity} {disease.replace('_', ' ')}."]

    if cbc.wbc > 11.0:
        lines.append(f"Leukocytosis noted (WBC {cbc.wbc:.1f}) suggesting active infection or inflammation.")
    elif cbc.wbc < 4.5:
        lines.append(f"Leukopenia noted (WBC {cbc.wbc:.1f}) — possible bone marrow suppression.")

    if bmp.creatinine > 1.3:
        lines.append(f"Elevated creatinine ({bmp.creatinine:.2f} mg/dL) consistent with renal impairment.")

    if inf.crp > 10:
        lines.append(f"Elevated CRP ({inf.crp:.1f} mg/L) consistent with systemic inflammation.")

    if ds.lactate and ds.lactate > 2.0:
        lines.append(f"Elevated lactate ({ds.lactate:.2f} mmol/L) — tissue hypoperfusion suspected.")

    if ds.bnp and ds.bnp > 500:
        lines.append(f"Markedly elevated BNP ({ds.bnp:.0f} pg/mL) consistent with heart failure.")

    if ds.glucose and ds.glucose > 250:
        lines.append(f"Severe hyperglycemia (glucose {ds.glucose:.1f} mg/dL). DKA protocol initiated.")

    if bmp.anion_gap > 12:
        lines.append(f"Elevated anion gap ({bmp.anion_gap:.1f} mEq/L) — metabolic acidosis workup warranted.")

    if flags:
        lines.append("Provider notified of critical values.")

    return " ".join(lines)
