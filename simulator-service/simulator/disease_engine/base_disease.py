from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulator.models.labs import BMPPanel, CBCPanel, DiseaseSpecificPanel, InflammatoryPanel
from simulator.models.patient import DischargeDisposition, Severity


@dataclass
class LabValueSet:
    cbc: CBCPanel
    bmp: BMPPanel
    inflammatory: InflammatoryPanel
    disease_specific: DiseaseSpecificPanel
    critical_flags: list[str]


@dataclass
class NoteFragments:
    hpi: str
    assessment: str
    plan: list[str]
    medications: list[str]
    secondary_diagnoses: list[str]
    interventions: list[str]


class BaseDiseaseEngine(ABC):
    DISEASE_NAME: str = ""
    ICD10_PRIMARY: str = ""

    # Mortality probability by severity
    MORTALITY_BY_SEVERITY: dict[str, float] = {
        "mild": 0.02,
        "moderate": 0.08,
        "severe": 0.20,
        "critical": 0.45,
    }

    # Readmission probability by severity
    READMISSION_BY_SEVERITY: dict[str, float] = {
        "mild": 0.10,
        "moderate": 0.18,
        "severe": 0.28,
        "critical": 0.40,
    }

    # Discharge disposition weights [home, snf, death] by severity
    DISPOSITION_WEIGHTS: dict[str, list[float]] = {
        "mild": [0.92, 0.07, 0.01],
        "moderate": [0.78, 0.18, 0.04],
        "severe": [0.55, 0.32, 0.13],
        "critical": [0.30, 0.40, 0.30],
    }

    @abstractmethod
    def get_lab_values(self, rng: np.random.Generator, severity: str) -> LabValueSet: ...

    @abstractmethod
    def get_note_fragments(self, severity: str, demographics: dict) -> NoteFragments: ...

    def get_followup_labs(
        self, rng: np.random.Generator, admission: LabValueSet, trajectory: str
    ) -> LabValueSet:
        """
        Generate follow-up labs (~24-48h after admission).
        trajectory: "improving" | "partial" | "declining"
        Default: nudge CBC/BMP inflammatory markers toward or away from normal.
        Disease engines override this for disease-specific trending.
        """
        factor = {"improving": -0.35, "partial": -0.10, "declining": 0.20}[trajectory]

        def _trend(val: float, target: float, noise_std: float) -> float:
            delta = (target - val) * abs(factor) * (1 if factor < 0 else -1)
            return float(max(0.1, val + delta + rng.normal(0, noise_std)))

        cbc = CBCPanel(
            wbc=_trend(admission.cbc.wbc, 7.5, 0.5),
            rbc=admission.cbc.rbc,
            hemoglobin=admission.cbc.hemoglobin,
            hematocrit=admission.cbc.hematocrit,
            platelets=_trend(admission.cbc.platelets, 250, 10),
        )
        bmp = BMPPanel(
            sodium=_trend(admission.bmp.sodium, 140, 1.0),
            potassium=admission.bmp.potassium,
            chloride=admission.bmp.chloride,
            bicarbonate=_trend(admission.bmp.bicarbonate, 24, 0.5),
            bun=_trend(admission.bmp.bun, 14, 1.5),
            creatinine=_trend(admission.bmp.creatinine, 0.9, 0.05),
            glucose=_trend(admission.bmp.glucose, 90, 5),
            calcium=admission.bmp.calcium,
        )
        inflammatory = InflammatoryPanel(
            crp=_trend(admission.inflammatory.crp, 5, 2),
            esr=_trend(admission.inflammatory.esr, 15, 2),
        )
        return LabValueSet(cbc, bmp, inflammatory, admission.disease_specific, [])

    def get_discharge_disposition(
        self, severity: str, rng: np.random.Generator
    ) -> DischargeDisposition:
        weights = self.DISPOSITION_WEIGHTS[severity]
        choice = str(rng.choice(["home", "skilled_nursing_facility", "death"], p=weights))
        return DischargeDisposition(choice)

    def get_mortality_risk(self, severity: str) -> float:
        return self.MORTALITY_BY_SEVERITY[severity]

    def get_readmission_risk(self, severity: str) -> float:
        return self.READMISSION_BY_SEVERITY[severity]

    @staticmethod
    def _normal(rng: np.random.Generator, mean: float, std: float, low: float = 0.0) -> float:
        return float(max(low, rng.normal(mean, std)))

    @staticmethod
    def _lognormal(rng: np.random.Generator, mean: float, sigma: float) -> float:
        return float(rng.lognormal(mean, sigma))

    def _base_cbc(self, rng: np.random.Generator) -> CBCPanel:
        """Normal-range CBC as baseline."""
        return CBCPanel(
            wbc=self._normal(rng, 7.5, 1.5, low=1.0),
            rbc=self._normal(rng, 4.8, 0.4, low=2.0),
            hemoglobin=self._normal(rng, 14.5, 1.5, low=5.0),
            hematocrit=self._normal(rng, 44.0, 4.0, low=15.0),
            platelets=self._normal(rng, 250.0, 60.0, low=10.0),
        )

    def _base_bmp(self, rng: np.random.Generator) -> BMPPanel:
        """Normal-range BMP as baseline."""
        return BMPPanel(
            sodium=self._normal(rng, 140.0, 2.5),
            potassium=self._normal(rng, 4.0, 0.4),
            chloride=self._normal(rng, 102.0, 2.5),
            bicarbonate=self._normal(rng, 25.0, 2.0),
            bun=self._normal(rng, 14.0, 4.0, low=2.0),
            creatinine=self._normal(rng, 0.9, 0.2, low=0.3),
            glucose=self._normal(rng, 90.0, 10.0, low=40.0),
            calcium=self._normal(rng, 9.5, 0.5),
        )

    def _base_inflammatory(self, rng: np.random.Generator) -> InflammatoryPanel:
        return InflammatoryPanel(
            crp=self._normal(rng, 4.0, 2.0, low=0.1),
            esr=self._normal(rng, 15.0, 5.0, low=1.0),
        )
