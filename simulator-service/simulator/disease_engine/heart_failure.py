from __future__ import annotations

import numpy as np

from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel


class HeartFailureEngine(BaseDiseaseEngine):
    DISEASE_NAME = "Acute Decompensated Heart Failure"
    ICD10_PRIMARY = "I50.9"

    MORTALITY_BY_SEVERITY = {"mild": 0.03, "moderate": 0.08, "severe": 0.22, "critical": 0.50}
    READMISSION_BY_SEVERITY = {"mild": 0.20, "moderate": 0.28, "severe": 0.38, "critical": 0.50}
    DISPOSITION_WEIGHTS = {
        "mild": [0.88, 0.10, 0.02],
        "moderate": [0.72, 0.22, 0.06],
        "severe": [0.48, 0.36, 0.16],
        "critical": [0.22, 0.42, 0.36],
    }

    # BNP: lognormal(log_mean, sigma) — convert to pg/mL
    _BNP_LOG = {"mild": (5.5, 0.5), "moderate": (7.2, 0.5), "severe": (8.1, 0.6), "critical": (9.0, 0.7)}
    _SODIUM = {"mild": (137, 3), "moderate": (133, 4), "severe": (129, 5), "critical": (125, 6)}
    _CREATININE_DELTA = {"mild": 0.1, "moderate": 0.35, "severe": 0.80, "critical": 1.60}

    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet:
        cbc = self._base_cbc(rng)
        # Dilutional anemia common
        hgb = cbc.hemoglobin - (0 if severity == "mild" else 1.5)
        cbc = CBCPanel(
            wbc=cbc.wbc,
            rbc=cbc.rbc,
            hemoglobin=max(7.0, hgb),
            hematocrit=max(21.0, cbc.hematocrit - (0 if severity == "mild" else 4.5)),
            platelets=cbc.platelets,
        )

        bmp = self._base_bmp(rng)
        na_mean, na_std = self._SODIUM[severity]
        cr_delta = self._CREATININE_DELTA[severity]
        bmp = BMPPanel(
            sodium=self._normal(rng, na_mean, na_std),
            potassium=self._normal(rng, 4.2, 0.5),
            chloride=self._normal(rng, 98, 3),
            bicarbonate=self._normal(rng, 24, 3),
            bun=self._normal(rng, 26, 8, low=5.0),
            creatinine=self._normal(rng, 0.9 + cr_delta, 0.3, low=0.3),
            glucose=bmp.glucose,
            calcium=bmp.calcium,
        )

        inflammatory = InflammatoryPanel(
            crp=self._normal(rng, 12, 6, low=1.0),
            esr=self._normal(rng, 22, 8, low=2.0),
        )

        log_mean, sigma = self._BNP_LOG[severity]
        bnp_val = self._lognormal(rng, log_mean, sigma)
        disease_specific = DiseaseSpecificPanel(bnp=bnp_val)

        flags: list[str] = []
        if bnp_val > 5000:
            flags.append("CRITICAL: BNP > 5000 pg/mL")
        if bmp.sodium < 125:
            flags.append("CRITICAL: Severe hyponatremia")

        return LabValueSet(cbc, bmp, inflammatory, disease_specific, flags)

    def get_followup_labs(
        self, rng: np.random.Generator, admission: LabValueSet, trajectory: str
    ) -> LabValueSet:
        base = super().get_followup_labs(rng, admission, trajectory)
        ds = admission.disease_specific
        if ds.bnp is not None:
            factor = {"improving": 0.55, "partial": 0.80, "declining": 1.25}[trajectory]
            new_bnp = float(max(50, ds.bnp * factor + rng.normal(0, ds.bnp * 0.05)))
        else:
            new_bnp = None
        new_ds = DiseaseSpecificPanel(bnp=new_bnp)
        return LabValueSet(base.cbc, base.bmp, base.inflammatory, new_ds, [])

    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments:
        age = demographics.get("age", 68)
        sex = demographics.get("sex", "male")
        pronoun = "He" if sex == "male" else "She"

        hpi_map = {
            "mild": (
                f"{pronoun} is a {age}-year-old with known HFrEF (EF 35%) presenting with 3 days "
                "of worsening lower extremity edema and mild dyspnea on exertion. "
                "Weight gain of 4 lbs over 2 days. HR 82, BP 142/88, SpO2 96% on RA."
            ),
            "moderate": (
                f"{pronoun} is a {age}-year-old with HFrEF (EF 30%) presenting with 2 days of "
                "worsening dyspnea, orthopnea (2-pillow), and bilateral lower extremity edema 2+. "
                "HR 96, BP 152/94, RR 22, SpO2 92% on RA. JVD present. Bibasilar crackles."
            ),
            "severe": (
                f"{pronoun} is a {age}-year-old with severe HFrEF (EF 20%) in acute decompensation. "
                "Severe dyspnea at rest, unable to lay flat. HR 118, BP 88/62, RR 26, "
                "SpO2 88% on 4L NC. S3 gallop. Marked JVD. Anasarca. BNP markedly elevated."
            ),
            "critical": (
                f"{pronoun} is a {age}-year-old with end-stage HFrEF (EF 15%) in cardiogenic shock. "
                "HR 135, BP 76/48 despite fluids, RR 30, SpO2 84% on 15L NRB. "
                "Cool, clammy extremities. Lactate 4.2. Requires inotropic support."
            ),
        }

        assessment_map = {
            "mild": "ADHF, NYHA class II. Compensated. Respond to outpatient diuresis.",
            "moderate": "ADHF, NYHA class III. Moderate decompensation. IV diuresis needed.",
            "severe": "ADHF, NYHA class IV. Severe decompensation with hypoxia. ICU monitoring.",
            "critical": "Cardiogenic shock. SCAI stage C–D. Inotropes required. Guarded prognosis.",
        }

        plan_map = {
            "mild": ["Furosemide 40mg PO daily (increase dose if inadequate response)", "Fluid restriction 1.5L/day", "Daily weights", "Follow-up in 1 week"],
            "moderate": ["IV furosemide 80mg bolus then 20mg/hr infusion", "Strict I&O", "Echocardiogram", "Cardiomegaly noted on CXR"],
            "severe": ["IV furosemide drip titrated to UO 0.5–1 mL/kg/hr", "Dobutamine 2.5 mcg/kg/min", "ICU transfer", "Cardiology consultation", "Consider BiPAP"],
            "critical": ["Inotropes: dobutamine 5–10 mcg/kg/min", "Vasopressors if MAP < 65", "IABP or Impella consideration", "Cardiology STAT consult", "Intubation PRN"],
        }

        meds_map = {
            "mild": ["Furosemide 40mg PO daily", "Lisinopril 5mg PO daily", "Carvedilol 6.25mg PO BID", "Spironolactone 25mg PO daily"],
            "moderate": ["Furosemide 80mg IV bolus", "Metolazone 5mg PO daily", "Lisinopril held if BP low"],
            "severe": ["Furosemide 20mg/hr IV infusion", "Dobutamine 2.5 mcg/kg/min IV", "Heparin 5000 units SQ BID (DVT prophylaxis)"],
            "critical": ["Dobutamine 7.5 mcg/kg/min IV", "Norepinephrine 0.1 mcg/kg/min IV", "Furosemide 40mg IV q6h", "Morphine 2mg IV PRN dyspnea"],
        }

        secondary = ["Chronic kidney disease", "Hypertension"]
        if severity in ("severe", "critical"):
            secondary += ["Cardiogenic pulmonary edema", "Acute kidney injury"]

        interventions_map = {
            "mild": [
                "Peripheral IV placed",
                "Daily weight monitoring initiated",
                "Telemetry monitoring",
                "Fluid restriction 1.5L/day counseled",
                "O2 via nasal cannula at 2L/min to maintain SpO2 > 94%",
            ],
            "moderate": [
                "Peripheral IV placed",
                "Foley catheter placed for strict I&Os",
                "Daily weight monitoring",
                "Continuous telemetry",
                "O2 titrated to SpO2 > 94%",
                "IV furosemide 80mg bolus administered",
            ],
            "severe": [
                "Peripheral IV placed",
                "Foley catheter placed for strict fluid balance",
                "Continuous cardiac monitoring",
                "Arterial line placed — right radial",
                "BiPAP initiated: IPAP 10 / EPAP 5, FiO2 40%",
                "IV furosemide drip initiated at 20mg/hr",
            ],
            "critical": [
                "Central venous catheter placed — right internal jugular",
                "Arterial line placed — right radial for continuous BP monitoring",
                "Foley catheter",
                "Emergent orotracheal intubation for respiratory failure",
                "Swan-Ganz catheter placed for hemodynamic monitoring",
                "Dobutamine infusion initiated at 2.5 mcg/kg/min",
                "Continuous cardiac monitoring and 12-lead EKG obtained",
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
