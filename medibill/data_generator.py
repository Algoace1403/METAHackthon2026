"""Deterministic synthetic-claim generator for MediBill-Env.

Produces CLEAN, policy-compliant medical claims given a provider, a policy
version, and a seed. The Round 1 ``DataCorruptor`` pipeline is then applied
on top to inject noise (null values, format drift, duplicate claims) just
like the Round 1 environment.

Policy-sensitive fields (diagnosis_code, procedure_code, pre_auth_flag,
required_signatures, provider, policy_version) are generated to be VALID
under the policy version specified at call time. Mid-episode policy drift
is handled by the environment's runtime mutation of the active policy —
not by this generator.

All data is synthetic. See medibill/__init__.py for licensing details.
"""

from __future__ import annotations

import random
from typing import Any

from medibill.data.icd10_pool import all_codes as all_icd10
from medibill.data.insurers import (
    PolicyRules,
    ProviderPack,
    get_provider,
    rules_as_dict,
)
from medibill.data.name_pool import all_first_names, all_hospitals, all_last_names
from medibill.ontology import codes_for_specialty as synth_codes_for_specialty
from medibill.ontology import list_codes as all_synth_codes


# Schema tags: every claim field must be exactly one of these two sets.
# Disjoint by construction — the grader's final_correctness and
# policy_compliance axes use this partition (see spec v3 §5.1).
IDENTITY_FIELDS: tuple[str, ...] = (
    "claim_id",
    "patient_id",
    "patient_name",
    "dob",
    "gender",
    "hospital_id",
    "admission_date",
    "discharge_date",
    "amount_billed_inr",
    "amount_paid_inr",
    "line_items",
)

POLICY_SENSITIVE_FIELDS: tuple[str, ...] = (
    "provider",
    "policy_version",
    "diagnosis_code",
    "procedure_code",
    "pre_auth_flag",
    "pre_auth_number",
    "required_signatures",
    "discharge_summary_attached",
    "diagnosis_narrative",
)

# Of the policy-sensitive fields, these are the ones the AGENT fills in — the
# grader's ``policy_compliance`` axis scores correctness on this subset only.
#
# The other policy-sensitive fields (provider, diagnosis_code, procedure_code)
# are visible in the dirty state because the treating physician already coded
# them in the EHR; the billing agent is not expected to re-derive them. That
# means no_op does NOT get free policy credit for carry-through values.
FILLIN_POLICY_FIELDS: tuple[str, ...] = (
    "policy_version",
    "pre_auth_flag",
    "pre_auth_number",
    "required_signatures",
    "discharge_summary_attached",
    "diagnosis_narrative",
)

# Internal/hidden fields (used only by the grader, never shown to the agent)
HIDDEN_FIELDS: tuple[str, ...] = ("_entity_id",)

# Defensive import-time assertion: identity / policy sets MUST be disjoint.
# The grader relies on this for its anti-double-counting guarantee (v3 §5.1).
assert not (set(IDENTITY_FIELDS) & set(POLICY_SENSITIVE_FIELDS)), (
    "IDENTITY_FIELDS and POLICY_SENSITIVE_FIELDS must be disjoint: "
    f"overlap = {set(IDENTITY_FIELDS) & set(POLICY_SENSITIVE_FIELDS)}"
)
assert set(FILLIN_POLICY_FIELDS).issubset(set(POLICY_SENSITIVE_FIELDS)), (
    "FILLIN_POLICY_FIELDS must be a subset of POLICY_SENSITIVE_FIELDS"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_patient(rng: random.Random, entity_idx: int) -> dict[str, Any]:
    """Generate deterministic patient demographics."""
    gender = rng.choice(("M", "F"))
    first = rng.choice(all_first_names(gender))
    last = rng.choice(all_last_names())
    year = rng.randint(1940, 2010)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return {
        "patient_id": f"PAT-{entity_idx:04d}",
        "patient_name": f"{first} {last}",
        "dob": f"{year:04d}-{month:02d}-{day:02d}",
        "gender": gender,
    }


def _pick_admission_window(rng: random.Random) -> tuple[str, str]:
    """Pick an admission/discharge date pair within FY25 (2024-04-01 to 2025-03-31)."""
    month = rng.randint(4, 15)  # will wrap via %12 below
    m = ((month - 1) % 12) + 1
    year = 2024 if month <= 12 else 2025
    day = rng.randint(1, 28)
    stay_days = rng.randint(1, 7)
    admission = f"{year:04d}-{m:02d}-{day:02d}"
    # Keep within same month for simplicity in easy task; v2 tasks may extend
    discharge_day = min(28, day + stay_days)
    discharge = f"{year:04d}-{m:02d}-{discharge_day:02d}"
    return admission, discharge


def _pick_icd_procedure_pair(rng: random.Random, rules: PolicyRules) -> tuple[str, str]:
    """Pick an ICD-10 code + SYNTH-PROC-v1 code consistent with each other and
    covered under the given policy version."""
    # Filter ICD-10 codes by the policy's covered prefixes
    eligible_icds = [c for c in all_icd10() if c.prefix in rules.covered_icd10_prefixes]
    if not eligible_icds:
        raise RuntimeError(
            f"No ICD-10 codes are covered under policy "
            f"{rules.provider}/{rules.policy_version}. Expand covered_icd10_prefixes."
        )
    icd = rng.choice(eligible_icds)

    # Find a SYNTH-PROC that lists this ICD prefix in its typical_icd10_ranges
    matching_procs: list[dict[str, Any]] = []
    for rec in synth_codes_for_specialty(icd.specialty):
        for rng_prefix in rec["typical_icd10_ranges"]:
            if icd.code.startswith(rng_prefix):
                matching_procs.append(rec)
                break

    # Fallback: any procedure in the specialty
    candidates = matching_procs or synth_codes_for_specialty(icd.specialty)
    # Final fallback: any procedure at all (GEN specialty has untyped codes)
    if not candidates:
        candidates = all_synth_codes()

    proc = rng.choice(candidates)
    return icd.code, proc["code"]


def _billing_amount(rng: random.Random, proc_code: str) -> int:
    """Return a billed amount perturbed around the SYNTH-PROC CGHS rate."""
    for rec in all_synth_codes():
        if rec["code"] == proc_code:
            base = int(rec["cghs_rate_inr"])
            break
    else:
        base = 1000
    # ±20% variance
    return max(100, int(base * rng.uniform(0.8, 1.2)))


def _narrative_for(icd_code: str) -> str:
    """Minimal free-text narrative keyed on ICD code. Used by rule: narrative must
    be non-empty when ``requires_diagnosis_narrative`` is True."""
    return (
        f"Patient presented with clinical findings consistent with {icd_code}. "
        f"Inpatient evaluation and management performed per standard protocol."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_clean_medical_records(
    n_claims: int,
    provider: str,
    policy_version: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate *n_claims* clean, policy-compliant medical claims deterministically.

    Args:
        n_claims: number of claims to generate.
        provider: provider name registered in insurers.PROVIDER_REGISTRY.
        policy_version: the policy version the claims should be compliant with.
        seed: RNG seed — same seed + args always produces identical output.

    Returns:
        A list of claim dicts, each containing:
          - hidden field: ``_entity_id``
          - identity fields (see ``IDENTITY_FIELDS``)
          - policy-sensitive fields (see ``POLICY_SENSITIVE_FIELDS``)

        Field-set membership is the grader's partition between
        ``final_correctness`` and ``policy_compliance`` axes.
    """
    if n_claims <= 0:
        raise ValueError("n_claims must be positive")

    pack: ProviderPack = get_provider(provider)
    rules: PolicyRules = pack.at(policy_version)
    rng = random.Random(seed)
    hospitals = all_hospitals()

    claims: list[dict[str, Any]] = []
    for i in range(n_claims):
        entity_id = f"CLAIM-{provider}-{i:04d}"
        patient = _pick_patient(rng, i)
        hospital = rng.choice(hospitals)
        admission, discharge = _pick_admission_window(rng)
        icd_code, proc_code = _pick_icd_procedure_pair(rng, rules)
        amount_billed = _billing_amount(rng, proc_code)
        coinsurance = amount_billed * (rules.coinsurance_percent / 100.0)
        amount_paid = max(0, min(rules.max_claim_inr, int(amount_billed - coinsurance)))
        pre_auth_required = amount_billed >= rules.pre_auth_threshold_inr
        narrative = _narrative_for(icd_code) if rules.requires_diagnosis_narrative else ""

        claim: dict[str, Any] = {
            # --- hidden ---
            "_entity_id": entity_id,
            # --- identity fields ---
            "claim_id": f"CLM-{provider}-{2024000 + i}",
            "patient_id": patient["patient_id"],
            "patient_name": patient["patient_name"],
            "dob": patient["dob"],
            "gender": patient["gender"],
            "hospital_id": hospital["hospital_id"],
            "admission_date": admission,
            "discharge_date": discharge,
            "amount_billed_inr": amount_billed,
            "amount_paid_inr": amount_paid,
            "line_items": f"{proc_code}; inpatient stay at {hospital['name']}",
            # --- policy-sensitive fields ---
            "provider": provider,
            "policy_version": policy_version,
            "diagnosis_code": icd_code,
            "procedure_code": proc_code,
            "pre_auth_flag": pre_auth_required,
            "pre_auth_number": f"PA-{provider}-{2024000 + i}" if pre_auth_required else None,
            "required_signatures": list(rules.required_signatures),
            "discharge_summary_attached": rules.requires_discharge_summary,
            "diagnosis_narrative": narrative,
        }
        claims.append(claim)

    return claims


def serialise_policy(rules: PolicyRules) -> dict[str, Any]:
    """Re-export of rules_as_dict for convenience at the top of the module."""
    return rules_as_dict(rules)


def claim_schema() -> dict[str, Any]:
    """Return the full schema describing claim fields.

    Each field tag is either ``identity`` or ``policy_sensitive``. The grader
    uses this mapping to partition cells into the two disjoint scoring axes.
    """
    expected_types: dict[str, str] = {}
    for f in IDENTITY_FIELDS:
        if f in ("amount_billed_inr", "amount_paid_inr"):
            expected_types[f] = "int"
        elif f in ("dob", "admission_date", "discharge_date"):
            expected_types[f] = "date"
        else:
            expected_types[f] = "str"
    for f in POLICY_SENSITIVE_FIELDS:
        if f == "pre_auth_flag" or f == "discharge_summary_attached":
            expected_types[f] = "bool"
        elif f == "required_signatures":
            expected_types[f] = "list"
        else:
            expected_types[f] = "str"

    field_tags: dict[str, str] = {}
    for f in IDENTITY_FIELDS:
        field_tags[f] = "identity"
    for f in POLICY_SENSITIVE_FIELDS:
        field_tags[f] = "policy_sensitive"

    return {
        "primary_key": "claim_id",
        "expected_types": expected_types,
        "field_tags": field_tags,
        "identity_fields": list(IDENTITY_FIELDS),
        "policy_sensitive_fields": list(POLICY_SENSITIVE_FIELDS),
    }
