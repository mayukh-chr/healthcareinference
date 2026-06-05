from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class CKDEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Chronic Kidney Disease"
    ICD10_PRIMARY = "N18.3"

    MORTALITY_BY_SEVERITY = {"mild": 0.02, "moderate": 0.07, "severe": 0.18, "critical": 0.42}
    READMISSION_BY_SEVERITY = {"mild": 0.15, "moderate": 0.25, "severe": 0.38, "critical": 0.52}
    DISPOSITION_WEIGHTS = {
        "mild": [0.90, 0.09, 0.01],
        "moderate": [0.76, 0.20, 0.04],
        "severe": [0.52, 0.34, 0.14],
        "critical": [0.24, 0.42, 0.34],
    }

    # Creatinine by CKD stage (mild=3, moderate=4, severe=4-5, critical=5/ESRD)
    _CREATININE = {"mild": (2.2, 0.4), "moderate": (4.0, 0.8), "severe": (6.5, 1.2), "critical": (9.0, 2.0)}
    _BUN = {"mild": (28, 8), "moderate": (55, 12), "severe": (88, 18), "critical": (120, 25)}
    _POTASSIUM = {"mild": (4.5, 0.4), "moderate": (5.0, 0.5), "severe": (5.8, 0.7), "critical": (6.4, 0.8)}
    _BICARB = {"mild": (22, 2), "moderate": (19, 3), "severe": (15, 3), "critical": (11, 3)}
    _HGB = {"mild": (11.5, 1.0), "moderate": (10.0, 1.2), "severe": (8.5, 1.3), "critical": (7.0, 1.5)}

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cr_mean, cr_std = self._CREATININE[severity]
        bun_mean, bun_std = self._BUN[severity]
        k_mean, k_std = self._POTASSIUM[severity]
        bicarb_mean, bicarb_std = self._BICARB[severity]
        hgb_mean, hgb_std = self._HGB[severity]

        cbc_base = self._base_cbc(rng)
        cbc = CBCPanel(
            wbc=cbc_base.wbc,
            rbc=self._normal(rng, 3.2, 0.5, low=1.5),
            hemoglobin=self._normal(rng, hgb_mean, hgb_std, low=5.0),
            hematocrit=self._normal(rng, hgb_mean * 3.0, hgb_std * 3.0, low=15.0),
            platelets=cbc_base.platelets,
        )

        bmp = BMPPanel(
            sodium=self._normal(rng, 137, 3),
            potassium=self._normal(rng, k_mean, k_std),
            chloride=self._normal(rng, 104, 3),
            bicarbonate=self._normal(rng, bicarb_mean, bicarb_std, low=4.0),
            bun=self._normal(rng, bun_mean, bun_std, low=5.0),
            creatinine=self._normal(rng, cr_mean, cr_std, low=0.5),
            glucose=self._normal(rng, 95, 15, low=40.0),
            calcium=self._normal(rng, 8.0, 0.8),
        )

        inflammatory = InflammatoryPanel(
            crp=self._normal(rng, 15, 8, low=1.0),
            esr=self._normal(rng, 35, 12, low=5.0),
        )

        disease_specific = DiseaseSpecificPanel(
            creatinine=self._normal(rng, cr_mean, cr_std, low=0.5),
        )

        flags: list[str] = []
        if bmp.potassium > 6.0:
            flags.append("CRITICAL: Hyperkalemia > 6.0 mEq/L — cardiac monitoring required")
        if bmp.bicarbonate < 12:
            flags.append("CRITICAL: Severe metabolic acidosis")
        if severity == "critical":
            flags.append("CRITICAL: ESRD — nephrology urgent consult")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_followup_labs(
        self, rng: np.random.Generator, admission: LabValueSet, trajectory: str
    ) -> LabValueSet:
        base = super().get_followup_labs(rng, admission, trajectory)
        ds = admission.disease_specific
        if ds.creatinine is not None:
            factor = {"improving": -0.30, "partial": -0.05, "declining": 0.15}[trajectory]
            new_cr = float(max(0.5, ds.creatinine + ds.creatinine * factor + rng.normal(0, 0.1)))
        else:
            new_cr = None
        new_ds = DiseaseSpecificPanel(creatinine=new_cr)
        return LabValueSet(base.cbc, base.bmp, base.inflammatory, new_ds, [])

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 62)
        sex = demographics.get("sex", "male")
        pronoun = "He" if sex == "male" else "She"

        stage_map = {"mild": "Stage 3", "moderate": "Stage 4", "severe": "Stage 4–5", "critical": "Stage 5 (ESRD)"}
        stage = stage_map[severity]

        hpi_map = {
            "mild": f"{pronoun} is a {age}-year-old with CKD {stage} presenting for routine follow-up. Creatinine has risen from baseline. No edema, nausea, or dyspnea. eGFR 35 mL/min.",
            "moderate": f"{pronoun} is a {age}-year-old with CKD {stage} presenting with fatigue, nausea, and ankle edema. Creatinine 4.1 mg/dL (baseline 3.2). eGFR 18 mL/min. BP 168/98.",
            "severe": f"{pronoun} is a {age}-year-old with CKD {stage} presenting with uremic symptoms: fatigue, nausea, vomiting, and pruritus. Creatinine 6.8, eGFR 8. BP 178/104. Uremic frost noted.",
            "critical": f"{pronoun} is a {age}-year-old with ESRD not yet on dialysis presenting with altered mental status, severe hyperkalemia (K 6.8), and pulmonary edema. Oliguric. Emergent dialysis required.",
        }

        assessment_map = {
            "mild": f"CKD {stage}. Stable. Optimize BP and avoid nephrotoxins.",
            "moderate": f"CKD {stage} with acute-on-chronic worsening. Nephrology consultation.",
            "severe": f"CKD {stage} with uremic syndrome. Pre-dialysis planning indicated.",
            "critical": "ESRD with uremic emergency. Emergent hemodialysis indicated.",
        }

        plan_map = {
            "mild": ["ACE inhibitor or ARB for BP and proteinuria", "Low-phosphorus, low-potassium diet", "Avoid NSAIDs and nephrotoxic agents", "Renal ultrasound", "Nephrology follow-up"],
            "moderate": ["Nephrology consultation for dialysis planning", "Erythropoiesis-stimulating agent for anemia", "Bicarbonate supplementation", "Phosphate binder", "Dietary restriction (K, P, Na)"],
            "severe": ["Urgent nephrology consultation", "AV fistula or permcath placement", "Sodium bicarbonate infusion", "Kayexalate for hyperkalemia", "Restrict IV fluids"],
            "critical": ["Emergency hemodialysis", "IV calcium gluconate for hyperkalemia cardiac protection", "Sodium bicarbonate", "ICU admission", "Nephrology and critical care co-management"],
        }

        meds_map = {
            "mild": ["Lisinopril 10mg PO daily", "Amlodipine 5mg PO daily", "Sevelamer 800mg PO TID with meals"],
            "moderate": ["Sodium bicarbonate 650mg PO TID", "Darbepoetin alfa SQ monthly", "Sevelamer 1600mg PO TID", "Calcitriol 0.25 mcg PO daily"],
            "severe": ["IV sodium bicarbonate", "Kayexalate 30g PO", "IV calcium gluconate 1g (if K > 6.5)", "Hold all nephrotoxic medications"],
            "critical": ["IV calcium gluconate 10mL of 10% solution", "Sodium bicarbonate 150 mEq IV", "Albuterol 10–20 mg nebulized (emergent hyperkalemia)", "Insulin regular 10 units + D50W 50mL"],
        }

        secondary = ["Hypertension", "Anemia of chronic kidney disease", "Hyperphosphatemia"]
        if severity in ("severe", "critical"):
            secondary += ["Uremia", "Hyperkalemia", "Metabolic acidosis"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed",
                "Strict I&O monitoring initiated",
                "Dietary consult placed for renal diet counseling",
                "Nephrotoxic medications reviewed and held",
            ],
            "moderate": [
                "Peripheral IV placed",
                "Foley catheter placed for strict I&Os",
                "Nephrology consultation placed",
                "Potassium-restricted diet initiated",
                "NS at 50mL/hr IV (cautious to avoid volume overload)",
            ],
            "severe": [
                "Peripheral IV placed",
                "Foley catheter placed",
                "IV sodium bicarbonate infusion initiated",
                "Continuous cardiac monitoring for hyperkalemia",
                "AV fistula mapping ordered for dialysis access planning",
                "Dietary restrictions enforced: K < 2g/day, phosphorus < 800mg/day",
            ],
            "critical": [
                "Dialysis catheter (permcath) placed — right internal jugular under ultrasound guidance",
                "Emergency hemodialysis initiated",
                "Continuous cardiac monitoring",
                "IV calcium gluconate 10mL of 10% solution administered for membrane stabilization",
                "Foley catheter placed",
                "ICU transfer completed",
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
