from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class SepsisEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Sepsis"
    ICD10_PRIMARY = "A41.9"

    MORTALITY_BY_SEVERITY = {"mild": 0.05, "moderate": 0.12, "severe": 0.28, "critical": 0.55}
    READMISSION_BY_SEVERITY = {"mild": 0.15, "moderate": 0.22, "severe": 0.32, "critical": 0.42}
    DISPOSITION_WEIGHTS = {
        "mild": [0.85, 0.13, 0.02],
        "moderate": [0.70, 0.22, 0.08],
        "severe": [0.45, 0.35, 0.20],
        "critical": [0.20, 0.38, 0.42],
    }

    # (mean, std) tuples per severity
    _WBC = {"mild": (14, 3), "moderate": (18, 4), "severe": (22, 5), "critical": (24, 6)}
    _CRP = {"mild": (60, 20), "moderate": (120, 40), "severe": (180, 50), "critical": (250, 70)}
    _PCT = {"mild": (1.5, 0.5), "moderate": (6, 2), "severe": (15, 5), "critical": (28, 8)}
    _LACTATE = {"mild": (1.8, 0.3), "moderate": (2.5, 0.5), "severe": (3.5, 0.8), "critical": (5.5, 1.5)}
    _CREATININE_DELTA = {"mild": 0.1, "moderate": 0.3, "severe": 0.8, "critical": 1.8}
    _PLATELETS = {"mild": (220, 40), "moderate": (160, 40), "severe": (100, 30), "critical": (60, 25)}

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cbc = self._base_cbc(rng)
        wbc_mean, wbc_std = self._WBC[severity]
        plt_mean, plt_std = self._PLATELETS[severity]
        cbc = CBCPanel(
            wbc=self._normal(rng, wbc_mean, wbc_std, low=1.0),
            rbc=cbc.rbc,
            hemoglobin=cbc.hemoglobin,
            hematocrit=cbc.hematocrit,
            platelets=self._normal(rng, plt_mean, plt_std, low=10.0),
        )

        bmp = self._base_bmp(rng)
        cr_delta = self._CREATININE_DELTA[severity]
        bmp = BMPPanel(
            sodium=self._normal(rng, 136, 4),
            potassium=bmp.potassium,
            chloride=bmp.chloride,
            bicarbonate=self._normal(rng, 20 if severity == "critical" else 23, 3),
            bun=self._normal(rng, 22, 6, low=5.0),
            creatinine=self._normal(rng, 0.9 + cr_delta, 0.25, low=0.3),
            glucose=bmp.glucose,
            calcium=bmp.calcium,
        )

        crp_mean, crp_std = self._CRP[severity]
        esr_val = self._normal(rng, 55, 20, low=10.0)
        inflammatory = InflammatoryPanel(
            crp=self._normal(rng, crp_mean, crp_std, low=5.0),
            esr=esr_val,
        )

        pct_mean, pct_std = self._PCT[severity]
        lac_mean, lac_std = self._LACTATE[severity]
        disease_specific = DiseaseSpecificPanel(
            procalcitonin=self._normal(rng, pct_mean, pct_std, low=0.5),
            lactate=self._normal(rng, lac_mean, lac_std, low=0.5),
        )

        flags: list[str] = []
        if disease_specific.lactate and disease_specific.lactate > 4.0:
            flags.append("CRITICAL: Lactate > 4.0 mmol/L")
        if cbc.platelets < 80:
            flags.append("CRITICAL: Platelets < 80")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_followup_labs(
        self, rng: np.random.Generator, admission: LabValueSet, trajectory: str
    ) -> LabValueSet:
        base = super().get_followup_labs(rng, admission, trajectory)
        ds = admission.disease_specific
        if ds.lactate is not None:
            factor = {"improving": 0.5, "partial": 0.85, "declining": 1.3}[trajectory]
            new_lactate = float(max(0.5, ds.lactate * factor + rng.normal(0, 0.1)))
        else:
            new_lactate = None
        new_ds = DiseaseSpecificPanel(
            procalcitonin=ds.procalcitonin,
            lactate=new_lactate,
        )
        return LabValueSet(base.cbc, base.bmp, base.inflammatory, new_ds, [])

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 65)
        sex = demographics.get("sex", "male")
        pronoun = "He" if sex == "male" else "She"

        hpi_map = {
            "mild": (
                f"{pronoun} is a {age}-year-old who presents with fever, chills, and malaise "
                "for 2 days. Vital signs notable for temperature 38.4°C, heart rate 102 bpm, "
                "blood pressure 118/72 mmHg, respiratory rate 20 breaths/min."
            ),
            "moderate": (
                f"{pronoun} is a {age}-year-old presenting with high fever, rigors, and confusion "
                "for 1 day. Vital signs: temperature 39.1°C, HR 118 bpm, BP 104/62 mmHg, "
                "RR 24, SpO2 94% on room air. Suspected source: urinary."
            ),
            "severe": (
                f"{pronoun} is a {age}-year-old with 12-hour history of fever, altered mental status, "
                "and decreased urine output. Temperature 39.5°C, HR 128 bpm, BP 92/58 mmHg, "
                "RR 28, SpO2 90% on 4L NC. SOFA score 6. Meets Sepsis-3 criteria."
            ),
            "critical": (
                f"{pronoun} is a {age}-year-old presenting in septic shock. Temperature 39.8°C, "
                "HR 138 bpm, BP 78/42 mmHg despite 2L IV fluids, RR 32, SpO2 86% on 6L NC. "
                "Requires vasopressor support. SOFA score >10. Lactate 5.8 mmol/L."
            ),
        }

        assessment_map = {
            "mild": "Sepsis, likely urinary source. SOFA score 1-2. Hemodynamically stable.",
            "moderate": "Sepsis with organ dysfunction. SOFA score 3-5. Responding to initial resuscitation.",
            "severe": "Severe sepsis with multi-organ dysfunction. SOFA score 6-9. Requires ICU monitoring.",
            "critical": "Septic shock. SOFA score ≥10. Vasopressor-dependent. Guarded prognosis.",
        }

        plan_map = {
            "mild": [
                "Blood cultures x2 before antibiotics",
                "IV ceftriaxone 2g q24h",
                "IV fluid resuscitation 30 mL/kg",
                "Monitor lactate trending",
                "Urine culture and sensitivity",
            ],
            "moderate": [
                "Blood cultures x2",
                "IV piperacillin-tazobactam 4.5g q8h",
                "30 mL/kg IV crystalloid bolus",
                "Serial lactate monitoring q4h",
                "Step down to oral antibiotics when clinically improved",
            ],
            "severe": [
                "Blood cultures x2 STAT",
                "IV meropenem 1g q8h + vancomycin",
                "Aggressive fluid resuscitation",
                "ICU transfer for monitoring",
                "Repeat lactate in 2h; target < 2 mmol/L",
                "Infectious disease consultation",
            ],
            "critical": [
                "Blood cultures x2 STAT; do not delay antibiotics",
                "IV meropenem + vancomycin + micafungin",
                "Vasopressors: norepinephrine titrated to MAP ≥65 mmHg",
                "Intubation if respiratory failure progresses",
                "ICU admission, SICU team activated",
                "Renal replacement therapy if oliguria persists",
                "Family meeting regarding prognosis",
            ],
        }

        meds_map = {
            "mild": ["Ceftriaxone 2g IV q24h", "Normal saline 1L IV bolus", "Acetaminophen 650mg q6h PRN fever"],
            "moderate": ["Piperacillin-tazobactam 4.5g IV q8h", "Lactated Ringer's 30mL/kg IV", "Ondansetron 4mg IV PRN"],
            "severe": ["Meropenem 1g IV q8h", "Vancomycin 25mg/kg IV load", "Norepinephrine 0.1 mcg/kg/min (if needed)", "Hydrocortisone 200mg/day IV"],
            "critical": ["Meropenem 2g IV q8h", "Vancomycin 25mg/kg IV", "Micafungin 100mg IV daily", "Norepinephrine titrated", "Vasopressin 0.03 units/min", "Hydrocortisone 200mg/day IV"],
        }

        secondary = ["Acute kidney injury", "Lactic acidosis"]
        if severity == "critical":
            secondary += ["ARDS", "Disseminated intravascular coagulation"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed — right antecubital",
                "NS 500mL bolus administered over 30 min",
                "Blood cultures x2 drawn prior to antibiotics",
                "Urine culture and sensitivity sent",
            ],
            "moderate": [
                "2 peripheral IVs placed",
                "NS 30mL/kg bolus administered over 3 hours",
                "Blood cultures x2 drawn prior to antibiotics",
                "Urine culture sent",
                "Foley catheter placed for strict I&Os",
            ],
            "severe": [
                "2 large-bore peripheral IVs placed",
                "NS 30mL/kg bolus over 3 hours",
                "Arterial line placed — right radial",
                "Foley catheter placed for strict I&Os",
                "Blood cultures x2 peripheral + 1 central drawn prior to antibiotics",
                "Continuous cardiac monitoring initiated",
            ],
            "critical": [
                "Central venous catheter placed — right internal jugular",
                "Arterial line placed — right radial for continuous BP monitoring",
                "Foley catheter for strict I&Os",
                "Emergent orotracheal intubation — 7.5 ETT, confirmed by CXR",
                "Norepinephrine infusion started at 0.05 mcg/kg/min, titrated to MAP ≥65",
                "Vasopressin 0.03 units/min added for refractory shock",
                "Continuous cardiac and SpO2 monitoring",
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
