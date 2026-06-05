from __future__ import annotations

from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_SYNONYM_MAP: dict[str, list[str]] = {
    "shortness of breath": ["dyspnea", "shortness of breath", "SOB", "difficulty breathing"],
    "chest pain": ["chest pain", "chest discomfort", "precordial pain", "chest tightness"],
    "fever": ["fever", "febrile", "pyrexia", "elevated temperature"],
    "nausea": ["nausea", "nausea and vomiting", "N/V", "emesis"],
    "altered mental status": ["altered mental status", "confusion", "AMS", "encephalopathy"],
    "elevated": ["elevated", "increased", "high", "above normal"],
    "decreased": ["decreased", "low", "below normal", "reduced"],
}

_QUALIFIER_MAP: dict[str, str] = {
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
    "critical": "critical",
}


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def synonym_filter(term: str, seed: int = 0) -> str:
        options = _SYNONYM_MAP.get(term.lower(), [term])
        rng = np.random.default_rng(seed)
        return str(options[int(rng.integers(0, len(options)))])

    def qualifier_filter(severity: str) -> str:
        return _QUALIFIER_MAP.get(severity, severity)

    def join_list(items: list[str], sep: str = ", ") -> str:
        return sep.join(items)

    def numbered_list(items: list[str]) -> str:
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))

    env.filters["synonym"] = synonym_filter
    env.filters["qualifier"] = qualifier_filter
    env.filters["join_list"] = join_list
    env.filters["numbered_list"] = numbered_list
    return env


_ENV = _make_env()


def render(template_name: str, context: dict) -> str:
    tmpl = _ENV.get_template(template_name)
    return tmpl.render(**context)
