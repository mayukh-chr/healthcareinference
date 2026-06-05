from simulator.disease_engine.base_disease import BaseDiseaseEngine, LabValueSet, NoteFragments
from simulator.disease_engine.ckd import CKDEngine
from simulator.disease_engine.diabetes import DiabetesEngine
from simulator.disease_engine.heart_failure import HeartFailureEngine
from simulator.disease_engine.pneumonia import PneumoniaEngine
from simulator.disease_engine.sepsis import SepsisEngine
from simulator.disease_engine.stroke import StrokeEngine

DISEASE_REGISTRY: dict[str, type[BaseDiseaseEngine]] = {
    "sepsis": SepsisEngine,
    "pneumonia": PneumoniaEngine,
    "heart_failure": HeartFailureEngine,
    "diabetes": DiabetesEngine,
    "ckd": CKDEngine,
    "stroke": StrokeEngine,
}

__all__ = [
    "BaseDiseaseEngine",
    "LabValueSet",
    "NoteFragments",
    "SepsisEngine",
    "PneumoniaEngine",
    "HeartFailureEngine",
    "DiabetesEngine",
    "CKDEngine",
    "StrokeEngine",
    "DISEASE_REGISTRY",
]
