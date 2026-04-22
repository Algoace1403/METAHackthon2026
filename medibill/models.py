"""Pydantic models for MediBill-Env.

The action schema exposes FIVE agent-visible operations; Round 1 primitives
(fix_value, fill_missing, merge_duplicates, ...) remain internal-only. Agents
interact with a deliberate three-tool surface (ehr_query, insurance_lookup,
coding_engine) plus two meta-actions (escalate_to_human, submit_claim). This
is the novelty claim discussed in spec v3 §3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Allowed action-type constants (source of truth)
# ---------------------------------------------------------------------------

AGENT_ACTION_TYPES: tuple[str, ...] = (
    "ehr_query",
    "insurance_lookup",
    "coding_engine",
    "escalate_to_human",
    "submit_claim",
)


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Envelope returned by every tool call."""

    tool: str
    status: str = Field(description='One of: "success", "error", "no_effect"')
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    cost: float = 0.0


class ClaimPreview(BaseModel):
    """Agent-visible slice of a claim under construction. Hidden fields (_entity_id)
    and ground-truth-only fields are never included here.

    Mutable fields are typed as ``Any`` so the preview faithfully reports
    whatever the agent wrote, even when the agent writes a type-invalid value.
    The grader compares against the authoritative dict in ``state.claims`` —
    this preview is purely presentational.
    """

    model_config = {"arbitrary_types_allowed": True}

    claim_id: str
    patient_id: Any | None = None
    patient_name: Any | None = None
    dob: Any | None = None
    gender: Any | None = None
    hospital_id: Any | None = None
    admission_date: Any | None = None
    discharge_date: Any | None = None
    amount_billed_inr: Any | None = None
    amount_paid_inr: Any | None = None
    line_items: Any | None = None
    provider: Any | None = None
    policy_version: Any | None = None
    diagnosis_code: Any | None = None
    procedure_code: Any | None = None
    pre_auth_flag: Any | None = None
    pre_auth_number: Any | None = None
    required_signatures: Any = Field(default_factory=list)
    discharge_summary_attached: Any | None = None
    diagnosis_narrative: Any = ""
    submitted: bool = False


# ---------------------------------------------------------------------------
# Core action / observation / state
# ---------------------------------------------------------------------------


class MediBillAction(Action):
    """The single action type the agent emits.

    Each ``action_type`` maps to exactly one handler in the environment. Only
    the action types listed in ``AGENT_ACTION_TYPES`` are accepted; anything
    else returns an error result (never a crash).

    Params per action_type:
        ``ehr_query``:        {"patient_id": str} OR {"claim_id": str}
        ``insurance_lookup``: {"provider": str}
        ``coding_engine``:    {"claim_id": str, "field": str, "value": Any}
        ``escalate_to_human``:{"claim_id": str, "field": str, "reason": str}
        ``submit_claim``:     {"claim_id": str}
    """

    action_type: str = Field(
        ...,
        description=f"One of: {', '.join(AGENT_ACTION_TYPES)}",
    )
    params: dict[str, Any] = Field(default_factory=dict)


class MediBillObservation(Observation):
    """Observation returned after each step.

    The agent sees a list of claims under construction (preview only), the
    most recent tool call result, step and budget context, and how many
    claims are still unsubmitted.

    The active ``policy_version`` is deliberately NOT exposed here. The only
    way the agent can learn the currently-active policy is to call
    ``insurance_lookup``. When drift occurs mid-episode, the agent's last
    cached lookup is now stale; there is no announcement.
    """

    claims: list[ClaimPreview] = Field(default_factory=list)
    claims_remaining: int = 0
    last_tool_result: ToolResult | None = None
    recent_tool_results: list[ToolResult] = Field(default_factory=list)
    step_number: int = 0
    max_steps: int = 30
    steps_remaining: int = 30
    budget_spent: float = 0.0
    budget_remaining: float = 40.0
    tool_costs: dict[str, float] = Field(default_factory=dict)
    task_id: str = ""
    task_name: str = ""
    difficulty: str = ""


class DriftRecord(BaseModel):
    """Internal record of a policy-version mutation event that has already fired."""

    step: int
    from_version: str
    to_version: str


class MediBillState(State):
    """Internal environment state. Not exposed to the agent directly."""

    # Inherited from State: episode_id, step_count
    task_id: str = ""
    provider: str = ""
    active_policy_version: str = ""
    initial_policy_version: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)  # dirty/in-progress (agent-mutable)
    ground_truth: list[dict[str, Any]] = Field(default_factory=list)  # clean targets (hidden)
    submitted_claim_ids: list[str] = Field(default_factory=list)
    tool_log: list[dict[str, Any]] = Field(default_factory=list)
    lookup_history: list[dict[str, Any]] = Field(default_factory=list)  # (step, provider, observed_version)
    drift_events_pending: list[dict[str, Any]] = Field(default_factory=list)  # future events
    drift_events_fired: list[DriftRecord] = Field(default_factory=list)
    max_steps: int = 30
    is_complete: bool = False
    budget: float = 40.0
    budget_spent: float = 0.0
    budget_remaining: float = 40.0
    previous_score: float = 0.0
    initial_raw_score: float = 0.0
    ambiguous_cells: list[tuple[str, str]] = Field(default_factory=list)
