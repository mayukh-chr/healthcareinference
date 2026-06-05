from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class PneumoniaEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Community-Acquired Pneumonia"
    ICD10_PRIMARY = "J18.9"

    MORTALITY_BY_SEVERITY = {"mild": 0.01, "moderate": 0.06, "severe": 0.18, "critical": 0.40}
    READMISSION_BY_SEVERITY = {"mild": 0.08, "moderate": 0.14, "severe": 0.25, "critical": 0.38}
    DISPOSITION_WEIGHTS = {
        "mild": [0.94, 0.05, 0.01],
        "moderate": [0.80, 0.16, 0.04],
        "severe": [0.55, 0.32, 0.13],
        "critical": [0.22, 0.38, 0.40],
    }

    _WBC = {"mild": (12, 2.5), "moderate": (15, 3), "severe": (18, 4), "critical": (22, 5)}
    _CRP = {"mild": (45, 15), "moderate": (90, 25), "severe": (160, 45), "critical": (220, 60)}

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cbc = self._base_cbc(rng)
        wbc_mean, wbc_std = self._WBC[severity]
        cbc = CBCPanel(
            wbc=self._normal(rng, wbc_mean, wbc_std, low=2.0),
            rbc=cbc.rbc,
            hemoglobin=cbc.hemoglobin,
            hematocrit=cbc.hematocrit,
            platelets=self._normal(rng, 300, 70, low=50.0),
        )

        bmp = self._base_bmp(rng)
        # Hyponatremia common in severe pneumonia
        na = 138 if severity in ("mild", "moderate") else self._normal(rng, 132, 4)
        bmp = BMPPanel(
            sodium=self._normal(rng, na, 2.5),
            potassium=bmp.potassium,
            chloride=bmp.chloride,
            bicarbonate=bmp.bicarbonate,
            bun=self._normal(rng, 16, 5, low=5.0),
            creatinine=bmp.creatinine,
            glucose=bmp.glucose,
            calcium=bmp.calcium,
        )

        crp_mean, crp_std = self._CRP[severity]
        inflammatory = InflammatoryPanel(
            crp=self._normal(rng, crp_mean, crp_std, low=5.0),
            esr=self._normal(rng, 45, 15, low=5.0),
        )

        pct = self._normal(rng, 1.0, 0.5, low=0.1) if severity in ("severe", "critical") else None
        disease_specific = DiseaseSpecificPanel(procalcitonin=pct)

        flags: list[str] = []
        if severity == "critical":
            flags.append("CRITICAL: Severe CAP — consider ICU admission")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 65)
        sex = demographics.get("sex", "male")
        pronoun = "He" if sex == "male" else "She"

        hpi_map = {
            "mild": (
                f"{pronoun} is a {age}-year-old presenting with 3 days of productive cough, "
                "low-grade fever, and mild dyspnea. Temperature 38.1°C, HR 88, BP 128/78, "
                "RR 18, SpO2 97% on room air. CXR: right lower lobe infiltrate."
            ),
            "moderate": (
                f"{pronoun} is a {age}-year-old with 4 days of productive cough, fever to 38.8°C, "
                "and moderate dyspnea. HR 100, BP 116/72, RR 22, SpO2 93% on room air. "
                "PSI class III. CXR: bilateral lower lobe infiltrates."
            ),
            "severe": (
                f"{pronoun} is a {age}-year-old presenting with high fever, severe dyspnea, "
                "and confusion for 1 day. Temperature 39.5°C, HR 118, BP 98/62, RR 28, "
                "SpO2 87% on room air. PSI class V. Meets CURB-65 score 4."
            ),
            "critical": (
                f"{pronoun} is a {age}-year-old in respiratory failure secondary to severe CAP. "
                "Temperature 39.9°C, HR 130, BP 88/54, RR 34, SpO2 82% despite 15L NRB mask. "
                "CURB-65 score 5. Requires emergent intubation."
            ),
        }

        assessment_map = {
            "mild": "Community-acquired pneumonia, mild severity. PSI class I–II. Outpatient therapy appropriate.",
            "moderate": "Community-acquired pneumonia, moderate severity. PSI class III. Hospital admission warranted.",
            "severe": "Severe community-acquired pneumonia. PSI class IV–V. ICU-level monitoring required.",
            "critical": "Critical CAP with respiratory failure. Intubated. ICU admission mandatory.",
        }

        plan_map = {
            "mild": ["Azithromycin 500mg PO daily ×5 days", "Supportive care", "Close outpatient follow-up in 48h"],
            "moderate": ["Ceftriaxone 1g IV q24h + azithromycin 500mg IV/PO daily", "IV fluids as needed", "SpO2 monitoring", "Chest PT"],
            "severe": ["Ceftriaxone 2g IV q24h + azithromycin 500mg IV daily", "High-flow nasal cannula or BiPAP", "ICU monitoring", "Repeat CXR in 24h"],
            "critical": ["Intubation with lung-protective ventilation", "Piperacillin-tazobactam 4.5g IV q8h + azithromycin", "Consider oseltamivir if flu season", "Prone positioning if P/F ratio < 150", "ICU admission"],
        }

        meds_map = {
            "mild": ["Azithromycin 500mg PO daily", "Acetaminophen 650mg q6h PRN"],
            "moderate": ["Ceftriaxone 1g IV q24h", "Azithromycin 500mg PO daily", "Albuterol nebulizer PRN"],
            "severe": ["Ceftriaxone 2g IV q24h", "Azithromycin 500mg IV daily", "Methylprednisolone 1mg/kg IV daily"],
            "critical": ["Piperacillin-tazobactam 4.5g IV q8h", "Azithromycin 500mg IV daily", "Oseltamivir 75mg PO BID", "Propofol infusion for sedation", "Fentanyl 25 mcg/hr IV"],
        }

        secondary = ["Hypoxemic respiratory failure", "Pleural effusion"]
        if severity == "critical":
            secondary += ["ARDS", "Sepsis"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed",
                "Blood cultures x2 drawn prior to antibiotics",
                "Supplemental O2 via nasal cannula at 2L/min",
                "Pulse oximetry continuous monitoring initiated",
            ],
            "moderate": [
                "Peripheral IV placed",
                "Blood cultures x2 drawn prior to antibiotics",
                "Sputum culture sent",
                "O2 via nasal cannula titrated to SpO2 > 92%",
                "Incentive spirometry initiated",
            ],
            "severe": [
                "2 peripheral IVs placed",
                "Blood cultures x2 drawn prior to antibiotics",
                "Sputum culture sent",
                "High-flow nasal cannula at 40L/min, FiO2 60%",
                "Arterial blood gas obtained — right radial",
                "Chest physiotherapy initiated",
            ],
            "critical": [
                "Emergent orotracheal intubation — 7.5 ETT at 23cm, CXR confirmed",
                "Mechanical ventilation: AC/VC, TV 6mL/kg IBW, PEEP 8, FiO2 100%",
                "Arterial line placed — right radial",
                "Central venous catheter placed — right internal jugular",
                "Blood cultures x2 drawn",
                "Prone positioning initiated for P/F ratio < 150",
            ],
        }

        return NoteFragments(
            hpi=hpi_map[severity],
            assessment=assessment_map[severity],
            plan=plan_map[severity],
            medications=meds_map[severity],
            secondary_diagnoses=secondary,
            interventions=interventions_map[severity],
        )
