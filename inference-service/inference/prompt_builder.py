from __future__ import annotations

_SYSTEM = (
    "You are a clinical AI assistant. Analyze the provided hospital encounter documents and respond "
    "with a valid JSON object that matches the requested schema. Be precise and clinically accurate."
)


def build_prompt(task: str, documents: list[str]) -> list[dict]:
    combined = "\n\n---\n\n".join(documents)

    task_instructions = {
        "diagnosis": (
            "Extract the primary diagnosis and any secondary diagnoses from these clinical documents. "
            "Return primary_diagnosis (full name), primary_icd10 (ICD-10 code), "
            "secondary_diagnoses (list of names), and confidence (0.0-1.0)."
        ),
        "entity_extraction": (
            "Extract all clinical entities from these documents. "
            "Return: diagnoses (list), medications (list with dosages), lab_findings (list of abnormal values)."
        ),
        "structured_json": (
            "Generate a complete structured clinical summary. "
            "Return: primary_diagnosis, primary_icd10, secondary_diagnoses, disease_severity "
            "(mild/moderate/severe/critical), risk_level (low/medium/high/critical), "
            "medications (list), key_lab_findings (dict of lab: status), discharge_disposition."
        ),
        "summarization": (
            "Write a concise 2-3 sentence clinical summary of this hospital encounter. "
            "Include the primary diagnosis, key treatments, and outcome. Return as {\"summary\": \"...\"}."
        ),
        "risk_classification": (
            "Assess the clinical risk for this patient. "
            "Return: mortality_risk (0.0-1.0), readmission_risk (0.0-1.0), "
            "risk_level (low/medium/high/critical)."
        ),
    }

    instruction = task_instructions.get(task, "Analyze the clinical documents and respond with JSON.")

    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"{instruction}\n\nCLINICAL DOCUMENTS:\n\n{combined}",
        },
    ]
