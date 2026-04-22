"""Task definitions for MediBill-Env.

Each task is a fully self-contained bundle: provider, initial policy version,
clean ground-truth claims, corruption specs that produce the agent-visible
dirty state at reset, and optional drift specs that mutate the active policy
mid-episode on medium/hard tasks.

Schema carries ``field_tags`` so the grader can partition cells between
``final_correctness`` (identity) and ``policy_compliance`` (policy-sensitive)
axes — disjoint by construction. See spec v3 §5.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medibill.data.insurers import get_provider
from medibill.data_generator import (
    claim_schema,
    generate_clean_medical_records,
)


@dataclass
class DriftEvent:
    """A scripted mid-episode policy mutation.

    When ``step`` is reached inside an episode, the environment swaps the
    active ``policy_version`` to ``to_version``. Ground-truth claim cells
    (policy-sensitive fields only) are re-scored against the new rules.

    The environment emits no public announcement of the change. The agent
    must call ``insurance_lookup`` again to observe the new ``policy_version``.
    """

    step: int
    to_version: str


@dataclass
class MediBillTask:
    task_id: str
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    description: str
    provider: str            # provider key in insurers.PROVIDER_REGISTRY
    initial_policy_version: str
    n_claims: int
    seed: int
    drift_events: list[DriftEvent] = field(default_factory=list)
    max_steps: int = 30
    budget: float = 40.0
    ambiguous_cells: list[tuple[str, str]] = field(default_factory=list)

    def build(self) -> dict[str, Any]:
        """Materialise ground-truth claims + schema for this task."""
        pack = get_provider(self.provider)
        # Validate the declared version exists before generation
        pack.at(self.initial_policy_version)
        gt = generate_clean_medical_records(
            n_claims=self.n_claims,
            provider=self.provider,
            policy_version=self.initial_policy_version,
            seed=self.seed,
        )
        # Validate any drift event refers to a real version
        for event in self.drift_events:
            pack.at(event.to_version)
        return {
            "ground_truth": gt,
            "schema": claim_schema(),
            "initial_policy_version": self.initial_policy_version,
            "provider": self.provider,
            "drift_events": [
                {"step": e.step, "to_version": e.to_version}
                for e in self.drift_events
            ],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, MediBillTask] = {}


def register_task(task: MediBillTask) -> None:
    _TASK_REGISTRY[task.task_id] = task


def get_task(task_id: str) -> MediBillTask:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(
            f"MediBill task '{task_id}' not found. Available: {list(_TASK_REGISTRY)}"
        )
    return _TASK_REGISTRY[task_id]


def list_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "provider": t.provider,
            "initial_policy_version": t.initial_policy_version,
            "n_claims": t.n_claims,
            "drift_events": len(t.drift_events),
            "max_steps": t.max_steps,
            "budget": t.budget,
        }
        for t in _TASK_REGISTRY.values()
    ]


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# EASY — single CGHS provider, no drift, 6 claims
EASY_CASHLESS = MediBillTask(
    task_id="easy_cashless",
    name="CGHS Cashless Claim Reconciliation",
    difficulty="easy",
    description=(
        "Six CGHS cashless claims under policy v2024.1. No policy drift. "
        "Agent must look up the policy once, then fill in pre-auth, "
        "signatures, discharge-summary flag, narrative, and policy_version "
        "fields to match the current ruleset before submitting each claim."
    ),
    provider="CGHS",
    initial_policy_version="v2024.1",
    n_claims=6,
    seed=42,
    drift_events=[],
    max_steps=60,
    budget=80.0,
)
register_task(EASY_CASHLESS)


# MEDIUM — single payer (Star) at latest version. Tests volume + policy
# interpretation under a stable ruleset. True multi-provider mixing is a
# Day-2 extension (would need data_generator to accept a provider-mix spec).
MEDIUM_MULTI_PAYER = MediBillTask(
    task_id="medium_multi_payer",
    name="High-Volume Cashless Reconciliation",
    difficulty="medium",
    description=(
        "Ten Star Health claims under policy v1.4 at the provider's latest "
        "version. No mid-episode drift. Stress-tests policy interpretation "
        "at scale: pre-auth threshold of 5000 INR catches most claims, "
        "narrative is required, and the signature list was expanded in v1.4."
    ),
    provider="Star",
    initial_policy_version="v1.4",
    n_claims=10,
    seed=43,
    drift_events=[],
    max_steps=100,
    budget=150.0,
)
register_task(MEDIUM_MULTI_PAYER)


# HARD — single provider, silent mid-episode drift at a seed-dependent step.
# This is the hero-mechanic task: policy_version changes without announcement.
#
# Drift step is NOT fixed: the environment picks a step uniformly from
# DRIFT_STEP_CHOICES per seed. The environment also picks the destination
# version from the provider's remaining newer versions. This closes the
# "learn to refresh at step 20" shortcut flagged in Codex review of v3.
DRIFT_STEP_CHOICES: tuple[int, ...] = tuple(range(10, 40))

HARD_DRIFT = MediBillTask(
    task_id="hard_drift",
    name="Silent Mid-Episode Policy Drift",
    difficulty="hard",
    description=(
        "Twelve claims under Star policy v1.3 at reset. At a seed-dependent "
        "step between 10 and 39 the active policy silently bumps to v1.4 "
        "(lower pre-auth threshold, diagnosis narrative now required, "
        "additional signature). The step is NOT fixed — the agent cannot "
        "memorise a schedule; it must notice the drift by re-querying "
        "insurance_lookup. Submissions after drift are graded against v1.4."
    ),
    provider="Star",
    initial_policy_version="v1.3",
    n_claims=12,
    seed=44,
    # Placeholder drift event; environment replaces the step at reset time
    # using the task seed and DRIFT_STEP_CHOICES. See environment.reset().
    drift_events=[DriftEvent(step=20, to_version="v1.4")],
    max_steps=140,
    budget=200.0,
    # Two cells flagged as legitimately ambiguous (correct answer is
    # escalate_to_human). Hand-curated per spec §5.4.
    ambiguous_cells=[
        ("CLAIM-Star-0003", "diagnosis_code"),
        ("CLAIM-Star-0008", "pre_auth_flag"),
    ],
)
register_task(HARD_DRIFT)
