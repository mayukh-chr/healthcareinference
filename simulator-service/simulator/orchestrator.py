from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from simulator.db.models import (
    ClinicalDocumentORM,
    EncounterORM,
    EventORM,
    GroundTruthORM,
    LabResultORM,
)
from simulator.db.session import AsyncSessionLocal
from simulator.disease_engine import DISEASE_REGISTRY
from simulator.document_generator.generator import generate_all_documents
from simulator.ground_truth.label_generator import generate_ground_truth
from simulator.models.events import ClinicalEvent, EventType
from simulator.models.labs import LabResult
from simulator.models.patient import (
    DischargeDisposition,
    EncounterType,
    HiddenPatientState,
    RiskScores,
    Severity,
)
from simulator.patient_generator.care_team_generator import generate_care_team
from simulator.patient_generator.demographic_generator import generate_demographics
from simulator.patient_generator.history_generator import generate_history
from simulator.patient_generator.seed_manager import SeedManager
from simulator.streaming.rabbitmq_publisher import publisher
from simulator.validation.validator import validate_encounter

logger = structlog.get_logger()

_DISEASES = list(DISEASE_REGISTRY.keys())
_DISEASE_WEIGHTS = [0.18, 0.20, 0.18, 0.16, 0.14, 0.14]  # sepsis, pneumonia, hf, diabetes, ckd, stroke
_SEVERITIES = ["mild", "moderate", "severe", "critical"]
_SEVERITY_WEIGHTS = [0.35, 0.35, 0.20, 0.10]


async def generate_encounter_batch(run_id: str, n: int, global_seed: int) -> None:
    logger.info("batch_started", run_id=run_id, n=n, seed=global_seed)
    success = 0
    failed = 0
    for i in range(n):
        patient_id = str(uuid.uuid4())
        try:
            ok = await _generate_single_encounter(patient_id, global_seed + i)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            logger.error("encounter_failed", patient_id=patient_id, error=str(exc))
            failed += 1
    logger.info("batch_completed", run_id=run_id, success=success, failed=failed)


async def _generate_single_encounter(patient_id: str, global_seed: int, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        seed = global_seed + attempt * 1_000_000
        sm = SeedManager(patient_id=patient_id, global_seed=seed)

        # Step 1: Demographics
        demo = generate_demographics(sm.get_rng("demographics"))

        # Step 2: Disease selection
        disease_rng = sm.get_rng("disease_selection")
        disease = str(disease_rng.choice(_DISEASES, p=_DISEASE_WEIGHTS))
        severity_str = str(disease_rng.choice(_SEVERITIES, p=_SEVERITY_WEIGHTS))
        severity = Severity(severity_str)

        # Step 3: History
        history_rng = sm.get_rng("history")
        chronic_conditions, allergies = generate_history(history_rng, disease, demo)

        # Step 4: Disease engine — lab values and note fragments
        engine = DISEASE_REGISTRY[disease]()
        lab_rng = sm.get_rng("lab_values")
        lab_values = engine.get_lab_values(lab_rng, severity_str)
        note_fragments = engine.get_note_fragments(severity_str, {
            "age": demo.age, "sex": demo.sex.value,
        })

        # Step 5: Risk scores
        mortality = engine.get_mortality_risk(severity_str)
        readmission = engine.get_readmission_risk(severity_str)
        risk_scores = RiskScores(
            mortality_probability=mortality,
            readmission_probability=readmission,
            risk_level="low",  # auto-derived by validator
        )

        # Step 6: Encounter type based on severity
        enc_type = EncounterType.ICU if severity in (Severity.SEVERE, Severity.CRITICAL) and disease_rng.random() < 0.4 else EncounterType.INPATIENT
        disp = engine.get_discharge_disposition(severity_str, disease_rng)

        # Step 7: Build HiddenPatientState (frozen)
        hidden = HiddenPatientState(
            patient_id=uuid.UUID(patient_id),
            encounter_id=uuid.uuid4(),
            demographics=demo,
            chronic_conditions=chronic_conditions,
            allergies=allergies,
            primary_disease=disease,
            severity=severity,
            treatment_plan=note_fragments.medications,
            risk_scores=risk_scores,
            discharge_disposition=disp,
            encounter_type=enc_type,
            generation_seed=seed,
        )

        # Step 8: Build event timeline
        events = _build_timeline(hidden, sm)

        # Step 9: Build admission LabResult
        admitted_at = events[0].timestamp
        lab_result = LabResult(
            encounter_id=hidden.encounter_id,
            patient_id=hidden.patient_id,
            order_event_id=events[1].event_id,
            collected_at=events[1].timestamp - timedelta(minutes=30),
            resulted_at=events[1].timestamp,
            cbc=lab_values.cbc,
            bmp=lab_values.bmp,
            inflammatory=lab_values.inflammatory,
            disease_specific=lab_values.disease_specific,
            narrative_report="",
            critical_flags=lab_values.critical_flags,
        )

        # Step 9b: Build follow-up LabResult (~24-48h post-admission)
        trajectory = _get_trajectory(severity_str, disp)
        followup_lab_values = engine.get_followup_labs(sm.get_rng("lab_followup"), lab_values, trajectory)
        followup_lab_result = LabResult(
            encounter_id=hidden.encounter_id,
            patient_id=hidden.patient_id,
            order_event_id=events[2].event_id,
            collected_at=events[2].timestamp - timedelta(hours=1),
            resulted_at=events[2].timestamp,
            cbc=followup_lab_values.cbc,
            bmp=followup_lab_values.bmp,
            inflammatory=followup_lab_values.inflammatory,
            disease_specific=followup_lab_values.disease_specific,
            narrative_report="",
            critical_flags=followup_lab_values.critical_flags,
        )

        # Step 10: Validate
        report = validate_encounter(events, lab_values, disease, severity_str)
        if not report.passed:
            logger.warning("validation_failed", patient_id=patient_id, attempt=attempt,
                           errors=[e.rule for e in report.errors])
            continue

        # Step 11: Generate + store ground truth BEFORE documents
        gt = generate_ground_truth(hidden, lab_values, note_fragments.secondary_diagnoses, note_fragments.medications)

        # Step 12: Generate documents (no hidden_state passed)
        demographics_dict = {"age": demo.age, "sex": demo.sex.value, "smoking_status": demo.smoking_status.value}
        care_team = generate_care_team(sm.get_rng("care_team"), disease, severity_str, enc_type.value)
        documents = generate_all_documents(
            events=events,
            lab_result=lab_result,
            followup_lab_result=followup_lab_result,
            note_fragments=note_fragments,
            lab_values=lab_values,
            disease=disease,
            severity=severity_str,
            demographics=demographics_dict,
            chronic_conditions=chronic_conditions,
            allergies=allergies,
            discharge_disposition=disp.value,
            encounter_type=enc_type.value,
            primary_diagnosis=gt.primary_diagnosis,
            primary_icd10=gt.primary_icd10,
            care_team=care_team,
        )

        # Step 13: Persist to database atomically
        async with AsyncSessionLocal() as session:
            await _persist(session, hidden, events, lab_result, followup_lab_result, documents, gt)

        # Step 14: Publish to RabbitMQ
        for event in events:
            await publisher.publish_event(event)
        await publisher.publish_ground_truth(gt)

        logger.info("encounter_generated", patient_id=patient_id, disease=disease, severity=severity_str)
        return True

    return False


def _build_timeline(hidden: HiddenPatientState, sm: SeedManager) -> list[ClinicalEvent]:
    rng = sm.get_rng("timeline")
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=int(rng.integers(1, 30)))

    admission_offset = timedelta(0)
    lab_offset = timedelta(hours=float(rng.uniform(2, 6)))
    note_offset = timedelta(hours=float(rng.uniform(12, 24)))
    treatment_offset = timedelta(hours=float(rng.uniform(24, 48)))
    los_hours = float(rng.uniform(48, 120))
    discharge_offset = timedelta(hours=los_hours)

    pid = hidden.patient_id
    eid = hidden.encounter_id

    return [
        ClinicalEvent(patient_id=pid, encounter_id=eid, timestamp=base + admission_offset,
                      event_type=EventType.PATIENT_ADMITTED, sequence_number=0,
                      payload={"encounter_type": hidden.encounter_type.value, "disease": hidden.primary_disease}),
        ClinicalEvent(patient_id=pid, encounter_id=eid, timestamp=base + lab_offset,
                      event_type=EventType.LAB_RESULTED, sequence_number=1,
                      payload={"panels": ["CBC", "BMP", "Inflammatory"]}),
        ClinicalEvent(patient_id=pid, encounter_id=eid, timestamp=base + note_offset,
                      event_type=EventType.PROGRESS_NOTE_CREATED, sequence_number=2,
                      payload={"hospital_day": int(note_offset.total_seconds() / 86400) + 1}),
        ClinicalEvent(patient_id=pid, encounter_id=eid, timestamp=base + treatment_offset,
                      event_type=EventType.TREATMENT_UPDATED, sequence_number=3,
                      payload={"medications": hidden.treatment_plan}),
        ClinicalEvent(patient_id=pid, encounter_id=eid, timestamp=base + discharge_offset,
                      event_type=EventType.DISCHARGE_COMPLETED, sequence_number=4,
                      payload={"disposition": hidden.discharge_disposition.value}),
    ]


def _get_trajectory(severity: str, disposition: DischargeDisposition) -> str:
    if disposition == DischargeDisposition.DEATH:
        return "declining"
    if severity in ("severe", "critical") and disposition == DischargeDisposition.SNF:
        return "partial"
    return "improving"


async def _persist(
    session: AsyncSession,
    hidden: HiddenPatientState,
    events: list[ClinicalEvent],
    lab_result: LabResult,
    followup_lab_result: LabResult,
    documents: list,
    gt,
) -> None:
    admitted_at = events[0].timestamp
    discharged_at = events[-1].timestamp

    session.add(EncounterORM(
        encounter_id=hidden.encounter_id,
        patient_id=hidden.patient_id,
        encounter_type=hidden.encounter_type.value,
        primary_disease=hidden.primary_disease,
        severity=hidden.severity.value,
        admitted_at=admitted_at,
        discharged_at=discharged_at,
        generation_seed=hidden.generation_seed,
    ))

    # Ground truth persisted FIRST
    session.add(GroundTruthORM(
        label_id=gt.label_id,
        encounter_id=gt.encounter_id,
        patient_id=gt.patient_id,
        primary_diagnosis=gt.primary_diagnosis,
        primary_icd10=gt.primary_icd10,
        secondary_diagnoses=gt.secondary_diagnoses,
        disease_severity=gt.disease_severity.value,
        mortality_risk=gt.mortality_risk,
        readmission_risk=gt.readmission_risk,
        expected_structured_output=gt.expected_structured_output.model_dump(),
        expected_summary=gt.expected_summary,
    ))

    for event in events:
        session.add(EventORM(
            event_id=event.event_id,
            encounter_id=event.encounter_id,
            patient_id=event.patient_id,
            timestamp=event.timestamp,
            event_type=event.event_type.value,
            sequence_number=event.sequence_number,
            payload=event.payload,
            causal_event_id=event.causal_event_id,
        ))

    for lr in (lab_result, followup_lab_result):
        session.add(LabResultORM(
            result_id=lr.result_id,
            encounter_id=lr.encounter_id,
            patient_id=lr.patient_id,
            collected_at=lr.collected_at,
            resulted_at=lr.resulted_at,
            cbc=lr.cbc.model_dump(),
            bmp=lr.bmp.model_dump(),
            inflammatory=lr.inflammatory.model_dump(),
            disease_specific=lr.disease_specific.model_dump(),
            narrative_report=lr.narrative_report,
            critical_flags=lr.critical_flags,
        ))

    for doc in documents:
        session.add(ClinicalDocumentORM(
            document_id=doc.document_id,
            encounter_id=doc.encounter_id,
            patient_id=doc.patient_id,
            document_type=doc.document_type.value,
            authored_at=doc.authored_at,
            author_name=doc.author_name,
            author_role=doc.author_role,
            content=doc.content,
        ))

    await session.commit()
