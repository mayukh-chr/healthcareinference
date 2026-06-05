from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class StrokeEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Acute Ischemic Stroke"
    ICD10_PRIMARY = "I63.9"

    MORTALITY_BY_SEVERITY = {"mild": 0.04, "moderate": 0.12, "severe": 0.30, "critical": 0.60}
    READMISSION_BY_SEVERITY = {"mild": 0.12, "moderate": 0.22, "severe": 0.35, "critical": 0.50}
    DISPOSITION_WEIGHTS = {
        "mild": [0.70, 0.28, 0.02],
        "moderate": [0.45, 0.44, 0.11],
        "severe": [0.22, 0.48, 0.30],
        "critical": [0.08, 0.40, 0.52],
    }

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cbc = self._base_cbc(rng)
        bmp = self._base_bmp(rng)

        # Stress hyperglycemia common in stroke
        glucose_delta = {"mild": 10, "moderate": 20, "severe": 35, "critical": 55}
        bmp = BMPPanel(
            sodium=bmp.sodium,
            potassium=bmp.potassium,
            chloride=bmp.chloride,
            bicarbonate=bmp.bicarbonate,
            bun=bmp.bun,
            creatinine=bmp.creatinine,
            glucose=self._normal(rng, 95 + glucose_delta[severity], 15, low=50.0),
            calcium=bmp.calcium,
        )

        inflammatory = InflammatoryPanel(
            crp=self._normal(rng, 8, 4, low=1.0),
            esr=self._normal(rng, 18, 6, low=2.0),
        )

        # INR elevated if on anticoagulation (simulate 40% of stroke patients)
        inr_elevated = rng.random() < 0.40
        inr_val = self._normal(rng, 2.5, 0.5) if inr_elevated else self._normal(rng, 1.1, 0.1)
        disease_specific = DiseaseSpecificPanel(inr=inr_val)

        flags: list[str] = []
        if severity == "critical":
            flags.append("CRITICAL: Large vessel occlusion — thrombectomy window")
        if bmp.glucose > 180:
            flags.append("Hyperglycemia — neurologic outcome risk")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 70)
        sex = demographics.get("sex", "male")
        pronoun = "He" if sex == "male" else "She"

        nihss_map = {"mild": "3", "moderate": "10", "severe": "18", "critical": "25+"}
        nihss = nihss_map[severity]

        hpi_map = {
            "mild": (
                f"{pronoun} is a {age}-year-old with sudden onset right arm weakness and slurred "
                f"speech 2 hours prior to presentation. NIHSS {nihss}. CT head: no hemorrhage. "
                "Last known well 2 hours ago. Hypertension history."
            ),
            "moderate": (
                f"{pronoun} is a {age}-year-old presenting with sudden left-sided hemiplegia, "
                f"facial droop, and aphasia. NIHSS {nihss}. BP 188/106 on arrival. "
                "CT head negative for hemorrhage. CTA: M1 occlusion suspected."
            ),
            "severe": (
                f"{pronoun} is a {age}-year-old found unresponsive with right gaze deviation and "
                f"left hemiplegia. NIHSS {nihss}. BP 210/118. CT head: early ischemic changes "
                "right MCA territory. CTA: ICA occlusion. Outside tPA window."
            ),
            "critical": (
                f"{pronoun} is a {age}-year-old in coma with fixed right pupil, absent corneal "
                f"reflex, bilateral Babinski. NIHSS {nihss}. BP 220/130. CT: large right MCA "
                "infarct with midline shift 8mm. Herniation imminent."
            ),
        }

        assessment_map = {
            "mild": f"Acute ischemic stroke, mild (NIHSS {nihss}). tPA considered — thrombolytics criteria review pending.",
            "moderate": f"Acute ischemic stroke, moderate (NIHSS {nihss}). IV tPA administered. Thrombectomy evaluation underway.",
            "severe": f"Acute ischemic stroke, severe (NIHSS {nihss}). Large vessel occlusion. Thrombectomy performed.",
            "critical": f"Malignant MCA infarction (NIHSS {nihss}). Herniation. Goals-of-care discussion with family.",
        }

        plan_map = {
            "mild": ["IV tPA if within window and no contraindications", "Aspirin 325mg once tPA cleared", "BP management: target < 180/105", "Neurology consultation", "MRI brain within 24h", "Echo and telemetry for cardioembolic source"],
            "moderate": ["IV tPA 0.9 mg/kg over 60min (max 90mg)", "Thrombectomy evaluation: CTA head/neck", "BP permissive hypertension post-tPA: < 180/105", "Neurology stroke team", "NPO until swallow evaluation"],
            "severe": ["Mechanical thrombectomy if within 24h and salvageable penumbra", "BP target < 185/110 pre-intervention", "Anticoagulation hold", "Stroke unit admission", "PT/OT/speech therapy consult"],
            "critical": ["Neurosurgery consult for hemicraniectomy consideration", "ICP monitoring", "Elevate HOB 30°", "Mannitol or hypertonic saline for herniation", "Family meeting — goals of care", "Comfort measures if appropriate"],
        }

        meds_map = {
            "mild": ["Aspirin 325mg PO daily (after tPA window)", "Atorvastatin 80mg PO nightly", "Lisinopril 5mg PO daily"],
            "moderate": ["Alteplase (tPA) 0.9 mg/kg IV", "Aspirin 81mg PO daily post-tPA", "Atorvastatin 80mg PO"],
            "severe": ["Aspirin 325mg PO (if no hemorrhagic transformation)", "Atorvastatin 80mg PO", "Levetiracetam 500mg IV BID (seizure prophylaxis)", "Heparin DVT prophylaxis"],
            "critical": ["Mannitol 1g/kg IV (herniation)", "Hypertonic saline 3% NaCl", "Labetalol IV PRN BP > 220", "Levetiracetam 1g IV load", "Fentanyl PRN comfort"],
        }

        secondary = ["Hypertension", "Atrial fibrillation"]
        if severity in ("severe", "critical"):
            secondary += ["Cerebral edema", "Dysphagia", "Aspiration pneumonia risk"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed",
                "Continuous cardiac telemetry initiated",
                "Neurological checks q1h initiated",
                "BP monitoring q15min",
                "NPO pending formal swallow evaluation",
                "Non-contrast CT head completed — no hemorrhage",
            ],
            "moderate": [
                "2 peripheral IVs placed",
                "IV tPA administered: 0.9 mg/kg over 60 min (max 90mg)",
                "BP monitoring q15min during tPA and 2h post-infusion",
                "Neurological checks q1h",
                "NPO — swallow screen deferred until stable",
                "Continuous telemetry for atrial fibrillation monitoring",
            ],
            "severe": [
                "2 peripheral IVs placed",
                "Emergent CTA head and neck completed",
                "Mechanical thrombectomy performed — TICI 2b reperfusion achieved",
                "Labetalol IV administered for BP > 185/110",
                "Foley catheter placed",
                "Head of bed elevated 30° for ICP management",
                "Neurology stroke team activated",
            ],
            "critical": [
                "Central line placed",
                "ICP monitor placed by neurosurgery",
                "Emergent orotracheal intubation for airway protection",
                "Mannitol 1g/kg IV administered for herniation",
                "Neurosurgery consultation for hemicraniectomy evaluation",
                "Continuous EEG monitoring initiated",
                "Foley catheter placed",
                "Head of bed elevated 30°",
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
