"""SYNTH-PROC-v1 procedure ontology loader.

SYNTH-PROC-v1 is an original synthetic procedure ontology. It is NOT derived
from and does NOT map to AMA CPT. See `synth_proc_v1.json` for full license
terms and disclaimer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ONTOLOGY_PATH = Path(__file__).resolve().parent / "synth_proc_v1.json"

_cached: dict[str, Any] | None = None


def load_ontology() -> dict[str, Any]:
    """Load SYNTH-PROC-v1 from disk. Cached after first call."""
    global _cached
    if _cached is None:
        with _ONTOLOGY_PATH.open("r", encoding="utf-8") as fh:
            _cached = json.load(fh)
    return _cached


def list_codes() -> list[dict[str, Any]]:
    """Return the flat list of procedure-code records."""
    return list(load_ontology()["codes"])


def get_code(code: str) -> dict[str, Any]:
    """Retrieve a single procedure record by code. Raises KeyError if missing."""
    for record in list_codes():
        if record["code"] == code:
            return record
    raise KeyError(f"SYNTH-PROC-v1 code '{code}' not found.")


def codes_for_specialty(specialty: str) -> list[dict[str, Any]]:
    """Return every procedure record belonging to a specialty (CARD / ORTH / ...)."""
    return [r for r in list_codes() if r["specialty"] == specialty.upper()]
