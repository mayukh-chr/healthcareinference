from __future__ import annotations

import numpy as np


class SeedManager:
    """
    Hierarchical deterministic seed derivation using PCG64.

    Given (patient_id, global_seed) → reproducible, byte-identical encounter.

    Each module gets an independent RNG namespace so adding a call in one
    module never shifts another module's output.
    """

    NAMESPACES: dict[str, int] = {
        "demographics": 0x0001,
        "disease_selection": 0x0002,
        "lab_values": 0x0003,
        "documents": 0x0004,
        "timeline": 0x0005,
        "ground_truth": 0x0006,
        "history": 0x0007,
        "validation": 0x0008,
        "care_team": 0x0009,
        "lab_followup": 0x000A,
    }

    def __init__(self, patient_id: str, global_seed: int) -> None:
        patient_int = int(patient_id.replace("-", ""), 16) & 0xFFFF_FFFF_FFFF_FFFF
        self._base = patient_int ^ (global_seed & 0xFFFF_FFFF_FFFF_FFFF)

    def get_rng(self, namespace: str) -> np.random.Generator:
        ns = self.NAMESPACES.get(namespace, hash(namespace) & 0xFFFF)
        seed = (self._base ^ (ns << 32)) & 0xFFFF_FFFF_FFFF_FFFF
        return np.random.default_rng(np.random.PCG64(seed))

