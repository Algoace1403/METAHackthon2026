"""Insurance provider policy pack for MediBill-Env.

Each provider exposes one or more POLICY VERSIONS. At reset, an episode picks
a starting version. On medium/hard tasks the environment may silently mutate
the active policy to a later version mid-episode (see environment.py). The
agent's only way to detect the mutation is to re-call ``insurance_lookup``
and compare the returned ``policy_version`` against what it saw before.

All providers are fictional or generic (CGHS / PMJAY are Government of India
public schemes; Star, HDFC ERGO stand-ins mirror public policy structure but
contain no proprietary payer content). Rates and rules are illustrative only.

Nothing in this file constitutes medical or insurance advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyRules:
    """Rules active under a specific policy version.

    Fields are tagged in the environment's schema as ``policy_sensitive`` so
    that the grader's ``policy_compliance`` axis evaluates them against the
    policy active AT SUBMIT TIME, not whatever rules the agent cached.
    """

    provider: str
    policy_version: str
    covered_icd10_prefixes: frozenset[str]
    pre_auth_threshold_inr: int
    max_claim_inr: int
    requires_discharge_summary: bool
    requires_diagnosis_narrative: bool
    required_signatures: tuple[str, ...]
    coinsurance_percent: float
    notes: str = ""


@dataclass(frozen=True)
class ProviderPack:
    """All known policy versions for a single provider, ordered earliest-first."""

    provider: str
    display_name: str
    versions: tuple[PolicyRules, ...]

    def at(self, version: str) -> PolicyRules:
        for rules in self.versions:
            if rules.policy_version == version:
                return rules
        raise KeyError(
            f"Policy version '{version}' not found for provider "
            f"'{self.provider}'. Known: {[r.policy_version for r in self.versions]}"
        )

    @property
    def latest_version(self) -> str:
        return self.versions[-1].policy_version


# ---------------------------------------------------------------------------
# Central Government Health Scheme (CGHS, Govt. of India)
# Public scheme. Broad ICD coverage. No pre-auth for routine OPD-equivalent.
# ---------------------------------------------------------------------------

CGHS = ProviderPack(
    provider="CGHS",
    display_name="Central Government Health Scheme",
    versions=(
        PolicyRules(
            provider="CGHS",
            policy_version="v2024.1",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "Z", "D5", "D6",
            }),
            pre_auth_threshold_inr=25000,
            max_claim_inr=500000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=False,
            required_signatures=("treating_physician", "hospital_authorized_signatory"),
            coinsurance_percent=0.0,
            notes="CGHS FY24-25 reference policy. No cost-sharing for beneficiaries.",
        ),
        PolicyRules(
            provider="CGHS",
            policy_version="v2024.2",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "Z", "D5", "D6",
            }),
            pre_auth_threshold_inr=15000,
            max_claim_inr=500000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=True,
            required_signatures=(
                "treating_physician",
                "hospital_authorized_signatory",
                "pharmacy_in_charge",
            ),
            coinsurance_percent=0.0,
            notes=(
                "DRIFT v2024.1 -> v2024.2: pre-auth threshold reduced 25000 -> 15000; "
                "diagnosis narrative now required; additional pharmacy signature required."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pradhan Mantri Jan Arogya Yojana (PMJAY / Ayushman Bharat)
# Public scheme. Capped claims. Pre-auth for all inpatient.
# ---------------------------------------------------------------------------

PMJAY = ProviderPack(
    provider="PMJAY",
    display_name="Ayushman Bharat - Pradhan Mantri Jan Arogya Yojana",
    versions=(
        PolicyRules(
            provider="PMJAY",
            policy_version="v2024.0",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "C", "D", "O", "P",
            }),
            pre_auth_threshold_inr=0,
            max_claim_inr=500000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=True,
            required_signatures=("treating_physician", "hospital_empanelment_officer"),
            coinsurance_percent=0.0,
            notes="All inpatient claims require pre-auth under PMJAY.",
        ),
        PolicyRules(
            provider="PMJAY",
            policy_version="v2025.1",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "C", "D", "O", "P",
            }),
            pre_auth_threshold_inr=0,
            max_claim_inr=500000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=True,
            required_signatures=(
                "treating_physician",
                "hospital_empanelment_officer",
                "district_coordinator",
            ),
            coinsurance_percent=0.0,
            notes=(
                "DRIFT v2024.0 -> v2025.1: district coordinator counter-signature required; "
                "appeal window reduced from 30 to 15 days."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Star Health (private, stand-in policy structure)
# Ranges mirror typical Indian private cashless policies.
# ---------------------------------------------------------------------------

STAR = ProviderPack(
    provider="Star",
    display_name="Star Health (illustrative private policy)",
    versions=(
        PolicyRules(
            provider="Star",
            policy_version="v1.3",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "D5", "D6", "O",
            }),
            pre_auth_threshold_inr=10000,
            max_claim_inr=1000000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=False,
            required_signatures=("treating_physician",),
            coinsurance_percent=10.0,
            notes="Cataract, hernia, and knee replacement subject to 2-year waiting period (not modeled at row level in v3).",
        ),
        PolicyRules(
            provider="Star",
            policy_version="v1.4",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "D5", "D6", "O",
            }),
            pre_auth_threshold_inr=5000,
            max_claim_inr=1000000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=True,
            required_signatures=("treating_physician", "hospital_authorized_signatory"),
            coinsurance_percent=10.0,
            notes=(
                "DRIFT v1.3 -> v1.4: pre-auth threshold reduced 10000 -> 5000; "
                "diagnosis narrative now required; hospital authorized signatory added."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# HDFC ERGO (private, stand-in policy structure)
# ---------------------------------------------------------------------------

HDFC_ERGO = ProviderPack(
    provider="HDFC_ERGO",
    display_name="HDFC ERGO (illustrative private policy)",
    versions=(
        PolicyRules(
            provider="HDFC_ERGO",
            policy_version="v3.0",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "D5", "D6", "O", "P",
            }),
            pre_auth_threshold_inr=15000,
            max_claim_inr=2000000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=False,
            required_signatures=("treating_physician", "hospital_authorized_signatory"),
            coinsurance_percent=5.0,
        ),
        PolicyRules(
            provider="HDFC_ERGO",
            policy_version="v3.1",
            covered_icd10_prefixes=frozenset({
                "I", "J", "K", "M", "E", "N", "R", "S", "T", "D5", "D6", "O", "P", "C",
            }),
            pre_auth_threshold_inr=15000,
            max_claim_inr=2000000,
            requires_discharge_summary=True,
            requires_diagnosis_narrative=True,
            required_signatures=("treating_physician", "hospital_authorized_signatory"),
            coinsurance_percent=5.0,
            notes=(
                "DRIFT v3.0 -> v3.1: oncology (C-prefix) ICD codes now covered; "
                "diagnosis narrative required on all claims."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, ProviderPack] = {
    "CGHS": CGHS,
    "PMJAY": PMJAY,
    "Star": STAR,
    "HDFC_ERGO": HDFC_ERGO,
}


def get_provider(name: str) -> ProviderPack:
    """Retrieve a ProviderPack by name. Raises KeyError if not found."""
    if name not in PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown insurance provider '{name}'. "
            f"Known: {list(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[name]


def list_providers() -> list[str]:
    """Return the list of known provider names."""
    return list(PROVIDER_REGISTRY.keys())


def rules_as_dict(rules: PolicyRules) -> dict[str, Any]:
    """Serialise PolicyRules to a JSON-safe dict for observation payloads.

    frozenset and tuple are converted to sorted list / list so the dict is
    deterministically serialisable by the OpenEnv HTTP layer.
    """
    return {
        "provider": rules.provider,
        "policy_version": rules.policy_version,
        "covered_icd10_prefixes": sorted(rules.covered_icd10_prefixes),
        "pre_auth_threshold_inr": rules.pre_auth_threshold_inr,
        "max_claim_inr": rules.max_claim_inr,
        "requires_discharge_summary": rules.requires_discharge_summary,
        "requires_diagnosis_narrative": rules.requires_diagnosis_narrative,
        "required_signatures": list(rules.required_signatures),
        "coinsurance_percent": rules.coinsurance_percent,
        "notes": rules.notes,
    }
