from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class DiabetesEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Type 2 Diabetes Mellitus"
    ICD10_PRIMARY = "E11.9"

    MORTALITY_BY_SEVERITY = {"mild": 0.01, "moderate": 0.04, "severe": 0.14, "critical": 0.35}
    READMISSION_BY_SEVERITY = {"mild": 0.12, "moderate": 0.20, "severe": 0.30, "critical": 0.42}
    DISPOSITION_WEIGHTS = {
        "mild": [0.95, 0.04, 0.01],
        "moderate": [0.84, 0.13, 0.03],
        "severe": [0.62, 0.28, 0.10],
        "critical": [0.30, 0.40, 0.30],
    }

    # Glucose by severity: controlled, uncontrolled, DKA, severe DKA
    _GLUCOSE = {"mild": (160, 30), "moderate": (280, 50), "severe": (420, 80), "critical": (520, 100)}
    _BICARB = {"mild": (24, 2), "moderate": (21, 3), "severe": (12, 3), "critical": (8, 2)}
    _POTASSIUM = {"mild": (4.0, 0.4), "moderate": (4.5, 0.5), "severe": (5.5, 0.8), "critical": (5.8, 1.0)}

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cbc = self._base_cbc(rng)

        glc_mean, glc_std = self._GLUCOSE[severity]
        bicarb_mean, bicarb_std = self._BICARB[severity]
        k_mean, k_std = self._POTASSIUM[severity]

        bmp = BMPPanel(
            sodium=self._normal(rng, 136, 3),
            potassium=self._normal(rng, k_mean, k_std),
            chloride=self._normal(rng, 102, 3),
            bicarbonate=self._normal(rng, bicarb_mean, bicarb_std, low=4.0),
            bun=self._normal(rng, 18, 5, low=5.0),
            creatinine=self._normal(rng, 1.1, 0.3, low=0.4),
            glucose=self._normal(rng, glc_mean, glc_std, low=50.0),
            calcium=self._normal(rng, 9.2, 0.6),
        )

        inflammatory = self._base_inflammatory(rng)

        disease_specific = DiseaseSpecificPanel(
            glucose=self._normal(rng, glc_mean, glc_std, low=50.0),
        )

        flags: list[str] = []
        if bmp.glucose > 500:
            flags.append("CRITICAL: Glucose > 500 mg/dL")
        if bmp.bicarbonate < 10:
            flags.append("CRITICAL: Severe metabolic acidosis (DKA)")
        if bmp.anion_gap > 20:
            flags.append("Elevated anion gap — DKA suspected")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_followup_labs(
        self, rng: np.random.Generator, admission: LabValueSet, trajectory: str
    ) -> LabValueSet:
        base = super().get_followup_labs(rng, admission, trajectory)
        ds = admission.disease_specific
        if ds.glucose is not None:
            factor = {"improving": 0.55, "partial": 0.80, "declining": 1.15}[trajectory]
            new_glucose = float(max(70, ds.glucose * factor + rng.normal(0, 10)))
        else:
            new_glucose = None
        new_ds = DiseaseSpecificPanel(glucose=new_glucose)
        return LabValueSet(base.cbc, base.bmp, base.inflammatory, new_ds, [])

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 55)
        sex = demographics.get("sex", "female")
        pronoun = "He" if sex == "male" else "She"

        hpi_map = {
            "mild": (
                f"{pronoun} is a {age}-year-old with Type 2 DM presenting for poorly controlled "
                "blood sugars. Recent FSBS readings 200–300 mg/dL. No polyuria, polydipsia, "
                "or nausea. A1c 9.2%."
            ),
            "moderate": (
                f"{pronoun} is a {age}-year-old with T2DM presenting with 3 days of polyuria, "
                "polydipsia, and fatigue. BG on arrival 318 mg/dL. Mild nausea, no vomiting. "
                "Minimal ketonuria on UA."
            ),
            "severe": (
                f"{pronoun} is a {age}-year-old with T2DM presenting with nausea, vomiting, "
                "abdominal pain, and Kussmaul respirations for 12 hours. BG 435 mg/dL. "
                "Anion gap 22. Bicarbonate 12. Urine ketones large. Meets DKA criteria."
            ),
            "critical": (
                f"{pronoun} is a {age}-year-old with T2DM presenting in severe DKA with altered "
                "mental status. BG 540 mg/dL, bicarbonate 8, pH 7.05 on ABG. Kussmaul breathing. "
                "Anion gap 26. Obtunded. Requires ICU-level management."
            ),
        }

        assessment_map = {
            "mild": "T2DM, poorly controlled. A1c elevated. Medication reconciliation and education needed.",
            "moderate": "T2DM, uncontrolled hyperglycemia. Mild DKA vs HHS. Requires IV insulin and hydration.",
            "severe": "Diabetic ketoacidosis (DKA), moderate-severe. Anion gap acidosis. IV insulin protocol.",
            "critical": "Severe DKA with altered mental status. pH < 7.1. ICU admission mandatory.",
        }

        plan_map = {
            "mild": ["Adjust insulin regimen", "Diabetes education", "Endocrine consultation", "Follow-up in 1 week"],
            "moderate": ["IV normal saline 500 mL/hr ×2h then 250 mL/hr", "Regular insulin 0.1 units/kg/hr", "Potassium repletion", "Q1h glucose checks"],
            "severe": ["IV insulin drip 0.1 units/kg/hr (target glucose reduction 75 mg/dL/hr)", "Aggressive IV fluid resuscitation", "Potassium 20–40 mEq/hr if K < 5.5", "Q1h BMP and glucose", "Transition to SQ insulin when AG normalizes"],
            "critical": ["ICU transfer", "IV insulin drip", "Bicarb infusion if pH < 6.9", "Continuous cardiac monitoring", "Foley catheter for strict UO", "Endocrine STAT consultation"],
        }

        meds_map = {
            "mild": ["Insulin glargine 20 units SQ qHS", "Metformin 500mg PO BID (hold if contrast needed)"],
            "moderate": ["Normal saline 1L IV bolus", "Regular insulin 6 units IV bolus then infusion", "KCl 20 mEq IV over 2h"],
            "severe": ["Regular insulin 0.1 units/kg/hr IV infusion", "Normal saline 1L/hr then 250 mL/hr", "KCl 40 mEq IV q4h", "D5W added when glucose < 250"],
            "critical": ["Regular insulin drip", "Sodium bicarbonate 150 mEq in D5W if pH < 6.9", "KCl aggressive repletion", "Magnesium sulfate PRN"],
        }

        secondary = ["Hypertension", "Hyperlipidemia"]
        if severity in ("severe", "critical"):
            secondary += ["Diabetic ketoacidosis", "Metabolic acidosis"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed",
                "Point-of-care glucose monitoring initiated q4h",
                "Dietary consult placed",
                "Diabetes education nurse notified",
            ],
            "moderate": [
                "Peripheral IV placed",
                "NS 1L IV bolus administered over 1 hour",
                "Regular insulin infusion initiated per hyperglycemia protocol",
                "Point-of-care glucose monitoring q1h",
                "Urine dipstick for ketones: trace positive",
                "Potassium 20mEq IV over 2h initiated (K 4.2)",
            ],
            "severe": [
                "2 peripheral IVs placed",
                "NS 1L/hr IV infusion started",
                "Regular insulin 0.1 units/kg/hr IV infusion initiated (DKA protocol)",
                "Potassium 40mEq IV over 4h (K 5.1, per DKA protocol)",
                "Foley catheter placed for strict I&Os",
                "Continuous cardiac monitoring for hyperkalemia-related arrhythmia",
            ],
            "critical": [
                "2 large-bore peripheral IVs placed",
                "NS 1L bolus over 30 min, then 500mL/hr",
                "Regular insulin infusion: 0.14 units/kg/hr IV",
                "Foley catheter placed",
                "Arterial line — right radial for serial ABG monitoring",
                "Continuous cardiac monitoring",
                "ICU transfer initiated",
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
