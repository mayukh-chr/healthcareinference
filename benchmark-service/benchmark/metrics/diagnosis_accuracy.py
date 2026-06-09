from __future__ import annotations


def evaluate(inferred: dict, ground_truth: dict) -> dict[str, bool | None]:
    inferred_icd10 = (inferred.get("primary_icd10") or "").strip().upper()
    gt_icd10 = ground_truth.get("primary_icd10", "").strip().upper()

    if not inferred_icd10 or not gt_icd10:
        return {"diagnosis_exact": None, "diagnosis_chapter": None}

    exact = inferred_icd10 == gt_icd10
    # ICD-10 chapter = first 3 characters (e.g., A41, J18, I50)
    chapter_match = inferred_icd10[:3] == gt_icd10[:3]

    return {"diagnosis_exact": exact, "diagnosis_chapter": chapter_match}
