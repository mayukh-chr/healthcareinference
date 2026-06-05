from __future__ import annotations

from simulator.disease_engine.base_disease import LabValueSet
from simulator.models.ground_truth import ExpectedStructuredOutput, GroundTruthLabels
from simulator.models.patient import HiddenPatientState

_ICD10_MAP: dict[str, str] = {
    "sepsis": "A41.9",
    "pneumonia": "J18.9",
    "heart_failure": "I50.9",
    "diabetes": "E11.9",
    "ckd": "N18.3",
    "stroke": "I63.9",
}

_DISEASE_DISPLAY: dict[str, str] = {
    "sepsis": "Sepsis",
    "pneumonia": "Community-Acquired Pneumonia",
    "heart_failure": "Acute Decompensated Heart Failure",
    "diabetes": "Type 2 Diabetes Mellitus",
    "ckd": "Chronic Kidney Disease",
    "stroke": "Acute Ischemic Stroke",
}


def _key_lab_findings(disease: str, severity: str, lab_values: LabValueSet) -> dict[str, str]:
    findings: dict[str, str] = {}

    cbc = lab_values.cbc
    bmp = lab_values.bmp
    inf = lab_values.inflammatory
    ds = lab_values.disease_specific

    if cbc.wbc > 11.0:
        findings["WBC"] = "elevated"
    elif cbc.wbc < 4.5:
        findings["WBC"] = "decreased"

    if bmp.creatinine > 1.3:
        findings["Creatinine"] = "elevated"

    if inf.crp > 10:
        findings["CRP"] = "elevated"

    if ds.lactate is not None and ds.lactate > 2.0:
        findings["Lactate"] = "elevated"

    if ds.bnp is not None and ds.bnp > 100:
        findings["BNP"] = "elevated"

    if ds.glucose is not None and ds.glucose > 126:
        findings["Glucose"] = "elevated"

    if ds.procalcitonin is not None and ds.procalcitonin > 0.5:
        findings["Procalcitonin"] = "elevated"

    if bmp.potassium > 5.0:
        findings["Potassium"] = "elevated"

    if bmp.bicarbonate < 22:
        findings["Bicarbonate"] = "decreased"

    return findings


def _summary(disease: str, severity: str, disposition: str, medications: list[str]) -> str:
    disp_text = {
        "home": "discharged home",
        "skilled_nursing_facility": "discharged to skilled nursing facility",
        "death": "expired during hospitalization",
    }.get(disposition, "discharged")

    med_str = medications[0] if medications else "standard therapy"
    display = _DISEASE_DISPLAY.get(disease, disease)

    return (
        f"Patient was admitted with {severity} {display} and treated with {med_str}. "
        f"Clinical course was {'uncomplicated' if severity in ('mild', 'moderate') else 'complex'} "
        f"and the patient was {disp_text}. "
        f"Close outpatient follow-up is recommended."
    )


def generate_ground_truth(
    hidden_state: HiddenPatientState,
    lab_values: LabValueSet,
    secondary_diagnoses: list[str],
    medications: list[str],
) -> GroundTruthLabels:
    disease = hidden_state.primary_disease
    severity = hidden_state.severity.value
    disposition = hidden_state.discharge_disposition.value

    primary_icd10 = _ICD10_MAP.get(disease, "Z00.0")
    primary_diagnosis = _DISEASE_DISPLAY.get(disease, disease)

    key_labs = _key_lab_findings(disease, severity, lab_values)

    expected_output = ExpectedStructuredOutput(
        primary_diagnosis=primary_diagnosis,
        primary_icd10=primary_icd10,
        secondary_diagnoses=secondary_diagnoses,
        disease_severity=hidden_state.severity,
        risk_level=hidden_state.risk_scores.risk_level,
        medications=medications,
        key_lab_findings=key_labs,
        discharge_disposition=hidden_state.discharge_disposition,
    )

    return GroundTruthLabels(
        encounter_id=hidden_state.encounter_id,
        patient_id=hidden_state.patient_id,
        primary_diagnosis=primary_diagnosis,
        primary_icd10=primary_icd10,
        secondary_diagnoses=secondary_diagnoses,
        disease_severity=hidden_state.severity,
        mortality_risk=hidden_state.risk_scores.mortality_probability,
        readmission_risk=hidden_state.risk_scores.readmission_probability,
        expected_structured_output=expected_output,
        expected_summary=_summary(disease, severity, disposition, medications),
    )
