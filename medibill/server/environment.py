"""Core MediBill-Env environment.

Implements the OpenEnv Environment contract (reset / step / state) with a
three-tool agent surface, policy-version drift mechanics, and a cost budget.

This file intentionally does NOT implement composite scoring — that lives in
`medibill.server.grader`. Step-level reward in v3-Day1 is a simple
bookkeeping signal; terminal composite scoring is plugged in Day 3 once the
grader lands and the baseline validation gate has passed.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server import Environment

from medibill.data.insurers import PROVIDER_REGISTRY, get_provider, rules_as_dict
from medibill.data_generator import (
    IDENTITY_FIELDS,
    POLICY_SENSITIVE_FIELDS,
    HIDDEN_FIELDS,
)
from medibill.models import (
    AGENT_ACTION_TYPES,
    ClaimPreview,
    DriftRecord,
    MediBillAction,
    MediBillObservation,
    MediBillState,
    ToolResult,
)
from medibill.tasks import get_task as get_medibill_task

logger = logging.getLogger(__name__)


# Per-tool costs (charged against the task's action budget)
TOOL_COSTS: dict[str, float] = {
    "ehr_query": 0.5,
    "insurance_lookup": 1.0,
    "coding_engine": 2.0,
    "escalate_to_human": 0.3,
    "submit_claim": 0.0,
}

# Step-level penalty to discourage stalling
STEP_COST: float = 0.002

# Default seed when none provided
DEFAULT_SEED: int = 42


class MediBillEnvironment(Environment[MediBillAction, MediBillObservation, MediBillState]):
    """MediBill-Env OpenEnv environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._state = MediBillState()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> MediBillObservation:
        task_id: str = kwargs.get("task_id", "easy_cashless")
        task = get_medibill_task(task_id)
        built = task.build()
        actual_seed = seed if seed is not None else DEFAULT_SEED

        # Randomise drift timing per task so an agent cannot memorise a
        # fixed schedule. Uses the episode seed so grading stays reproducible
        # per seed. hard_drift gets one randomised step; hard_silent_revert
        # gets two (drift forward at step S1, revert at step S1 + gap).
        drift_events = list(built["drift_events"])
        if task_id == "hard_drift" and drift_events:
            from medibill.tasks import DRIFT_STEP_CHOICES
            rng = random.Random(actual_seed)
            new_step = rng.choice(DRIFT_STEP_CHOICES)
            drift_events[0] = {**drift_events[0], "step": new_step}
        elif task_id == "hard_silent_revert" and len(drift_events) >= 2:
            from medibill.tasks import REVERT_FIRST_DRIFT_CHOICES, REVERT_GAP_CHOICES
            rng = random.Random(actual_seed)
            first_step = rng.choice(REVERT_FIRST_DRIFT_CHOICES)
            gap = rng.choice(REVERT_GAP_CHOICES)
            drift_events[0] = {**drift_events[0], "step": first_step}
            drift_events[1] = {**drift_events[1], "step": first_step + gap}

        # The agent's starting dirty state: identity fields intact from the
        # ground truth; policy-sensitive fields cleared so the agent must call
        # insurance_lookup and coding_engine to fill them in correctly.
        dirty = _build_dirty_from_ground_truth(built["ground_truth"])

        self._state = MediBillState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            provider=built["provider"],
            active_policy_version=built["initial_policy_version"],
            initial_policy_version=built["initial_policy_version"],
            claims=copy.deepcopy(dirty),
            ground_truth=copy.deepcopy(built["ground_truth"]),
            submitted_claim_ids=[],
            tool_log=[],
            lookup_history=[],
            drift_events_pending=drift_events,
            drift_events_fired=[],
            max_steps=task.max_steps,
            is_complete=False,
            budget=task.budget,
            budget_spent=0.0,
            budget_remaining=task.budget,
            previous_score=0.0,
            initial_raw_score=0.0,
            ambiguous_cells=list(task.ambiguous_cells),
        )
        self._task_name = task.name
        self._difficulty = task.difficulty

        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: MediBillAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> MediBillObservation:
        if self._state.is_complete:
            return self._build_observation(reward=0.0, done=True)

        self._state.step_count += 1
        self._maybe_fire_drift()

        # Validate action_type
        if action.action_type not in AGENT_ACTION_TYPES:
            result = ToolResult(
                tool=action.action_type,
                status="error",
                message=(
                    f"Unknown action_type '{action.action_type}'. "
                    f"Allowed: {list(AGENT_ACTION_TYPES)}"
                ),
                cost=0.0,
            )
            self._state.tool_log.append(
                _log_entry(result, step=self._state.step_count, params=action.params)
            )
            return self._build_observation(reward=-0.02, done=False)

        # Budget gate
        cost = TOOL_COSTS.get(action.action_type, 1.0)
        if cost > 0 and self._state.budget_remaining < cost:
            result = ToolResult(
                tool=action.action_type,
                status="error",
                message=(
                    f"Budget exhausted: {self._state.budget_remaining:.2f} remaining, "
                    f"tool costs {cost:.2f}"
                ),
                cost=0.0,
            )
            self._state.tool_log.append(
                _log_entry(result, step=self._state.step_count, params=action.params)
            )
            return self._build_observation(reward=-0.02, done=False)

        # Dispatch
        handler_name = f"_tool_{action.action_type}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            result = ToolResult(
                tool=action.action_type,
                status="error",
                message=f"Handler '{handler_name}' missing (should not happen).",
                cost=0.0,
            )
        else:
            try:
                result = handler(action.params)
            except (KeyError, TypeError, ValueError) as exc:
                result = ToolResult(
                    tool=action.action_type,
                    status="error",
                    message=f"Invalid params: {exc}",
                    cost=0.0,
                )
            except Exception as exc:  # safety net; never crash the server
                logger.exception("Unexpected error in %s handler", action.action_type)
                result = ToolResult(
                    tool=action.action_type,
                    status="error",
                    message=str(exc),
                    cost=0.0,
                )

        # Deduct cost on success and no_effect (not on error)
        if result.status != "error":
            self._state.budget_spent += cost
            self._state.budget_remaining = self._state.budget - self._state.budget_spent
            result = result.model_copy(update={"cost": cost})

        self._state.tool_log.append(
            _log_entry(result, step=self._state.step_count, params=action.params)
        )

        # Termination: all claims submitted OR max steps reached
        all_submitted = len(self._state.submitted_claim_ids) >= len(self._state.claims)
        step_limit = self._state.step_count >= self._state.max_steps
        is_done = all_submitted or step_limit
        self._state.is_complete = is_done

        reward = self._compute_reward(result, is_done)

        return self._build_observation(reward=reward, done=is_done)

    @property
    def state(self) -> MediBillState:  # type: ignore[override]
        return self._state

    # ------------------------------------------------------------------
    # Drift firing
    # ------------------------------------------------------------------

    def _maybe_fire_drift(self) -> None:
        """Fire any scripted drift events whose step has arrived.

        The environment deliberately does NOT announce the event to the agent
        in the observation stream — that would defeat the hero mechanic. The
        only way the agent can detect drift is to call ``insurance_lookup``
        and compare the returned ``policy_version`` to what it saw before.
        """
        step = self._state.step_count
        still_pending: list[dict[str, Any]] = []
        for event in self._state.drift_events_pending:
            if step >= event["step"]:
                prev_version = self._state.active_policy_version
                new_version = event["to_version"]
                self._state.active_policy_version = new_version
                self._state.drift_events_fired.append(
                    DriftRecord(step=step, from_version=prev_version, to_version=new_version)
                )
                # Ground-truth policy-sensitive fields are re-computed against the
                # new policy so that the grader can evaluate compliance at submit
                # time. Only cells NOT yet submitted are updated.
                _recompute_policy_fields_inplace(
                    self._state.ground_truth,
                    new_provider=self._state.provider,
                    new_version=new_version,
                )
            else:
                still_pending.append(event)
        self._state.drift_events_pending = still_pending

    # ------------------------------------------------------------------
    # Reward (step-level only; terminal composite lands with the grader)
    # ------------------------------------------------------------------

    def _compute_reward(self, result: ToolResult, is_done: bool) -> float:
        if result.status == "error":
            return -0.02
        if result.status == "no_effect":
            return -0.005
        # Small positive for forward motion; final composite scoring happens
        # in the grader once it lands on Day 3.
        return -STEP_COST if not is_done else 0.0

    # ------------------------------------------------------------------
    # Claim lookup
    # ------------------------------------------------------------------

    def _find_claim(self, claim_id: str) -> dict[str, Any] | None:
        for c in self._state.claims:
            if c.get("claim_id") == claim_id:
                return c
        return None

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _tool_ehr_query(self, params: dict[str, Any]) -> ToolResult:
        pid = params.get("patient_id")
        cid = params.get("claim_id")
        if not pid and not cid:
            return ToolResult(
                tool="ehr_query",
                status="error",
                message="ehr_query requires either 'patient_id' or 'claim_id'.",
            )
        matches: list[dict[str, Any]] = []
        for claim in self._state.claims:
            if pid and claim.get("patient_id") == pid:
                matches.append(claim)
            elif cid and claim.get("claim_id") == cid:
                matches.append(claim)
        if not matches:
            return ToolResult(
                tool="ehr_query",
                status="no_effect",
                message="No matching records in the EHR.",
            )
        payload = [_public_claim_view(m) for m in matches]
        return ToolResult(
            tool="ehr_query",
            status="success",
            message=f"Found {len(matches)} EHR record(s).",
            payload={"records": payload},
        )

    def _tool_insurance_lookup(self, params: dict[str, Any]) -> ToolResult:
        provider = params.get("provider", self._state.provider)
        if provider not in PROVIDER_REGISTRY:
            return ToolResult(
                tool="insurance_lookup",
                status="error",
                message=f"Unknown provider '{provider}'.",
            )
        # The CURRENT active version governs the payload — this is how the
        # agent discovers drift after it has happened.
        rules = get_provider(provider).at(self._state.active_policy_version)
        self._state.lookup_history.append(
            {
                "step": self._state.step_count,
                "provider": provider,
                "observed_version": rules.policy_version,
            }
        )
        return ToolResult(
            tool="insurance_lookup",
            status="success",
            message=(
                f"Current policy for {provider}: {rules.policy_version}."
            ),
            payload={"rules": rules_as_dict(rules)},
        )

    def _tool_coding_engine(self, params: dict[str, Any]) -> ToolResult:
        claim_id = params.get("claim_id")
        field = params.get("field")
        value = params.get("value")
        if not claim_id or not field:
            return ToolResult(
                tool="coding_engine",
                status="error",
                message="coding_engine requires 'claim_id', 'field', 'value'.",
            )
        claim = self._find_claim(claim_id)
        if claim is None:
            return ToolResult(
                tool="coding_engine",
                status="error",
                message=f"Unknown claim_id '{claim_id}'.",
            )
        if field in HIDDEN_FIELDS:
            return ToolResult(
                tool="coding_engine",
                status="error",
                message=f"Field '{field}' is hidden; not agent-modifiable.",
            )
        if field not in IDENTITY_FIELDS and field not in POLICY_SENSITIVE_FIELDS:
            return ToolResult(
                tool="coding_engine",
                status="error",
                message=f"Unknown field '{field}'.",
            )
        if claim_id in self._state.submitted_claim_ids:
            return ToolResult(
                tool="coding_engine",
                status="error",
                message=f"Claim '{claim_id}' is already submitted and locked.",
            )
        prev = claim.get(field)
        claim[field] = value
        return ToolResult(
            tool="coding_engine",
            status="success",
            message=f"Set '{field}' on '{claim_id}'.",
            payload={"previous": prev, "current": value},
        )

    def _tool_escalate_to_human(self, params: dict[str, Any]) -> ToolResult:
        claim_id = params.get("claim_id")
        field = params.get("field")
        reason = params.get("reason", "")
        if not claim_id or not field:
            return ToolResult(
                tool="escalate_to_human",
                status="error",
                message="escalate_to_human requires 'claim_id', 'field', 'reason'.",
            )
        claim = self._find_claim(claim_id)
        if claim is None:
            return ToolResult(
                tool="escalate_to_human",
                status="error",
                message=f"Unknown claim_id '{claim_id}'.",
            )
        claim.setdefault("_escalations", []).append({"field": field, "reason": reason})
        return ToolResult(
            tool="escalate_to_human",
            status="success",
            message=f"Escalated '{field}' on '{claim_id}' for human review.",
        )

    def _tool_submit_claim(self, params: dict[str, Any]) -> ToolResult:
        claim_id = params.get("claim_id")
        if not claim_id:
            return ToolResult(
                tool="submit_claim",
                status="error",
                message="submit_claim requires 'claim_id'.",
            )
        claim = self._find_claim(claim_id)
        if claim is None:
            return ToolResult(
                tool="submit_claim",
                status="error",
                message=f"Unknown claim_id '{claim_id}'.",
            )
        if claim_id in self._state.submitted_claim_ids:
            return ToolResult(
                tool="submit_claim",
                status="no_effect",
                message=f"Claim '{claim_id}' was already submitted.",
            )
        self._state.submitted_claim_ids.append(claim_id)
        claim["_submitted_at_step"] = self._state.step_count
        claim["_submitted_under_version"] = self._state.active_policy_version
        return ToolResult(
            tool="submit_claim",
            status="success",
            message=(
                f"Submitted '{claim_id}' at step {self._state.step_count} "
                f"under policy {self._state.active_policy_version}."
            ),
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float | None, done: bool) -> MediBillObservation:
        previews: list[ClaimPreview] = []
        for c in self._state.claims:
            view = _public_claim_view(c)
            view["submitted"] = c.get("claim_id") in self._state.submitted_claim_ids
            previews.append(ClaimPreview(**view))

        recent_raw = self._state.tool_log[-5:]
        recent = [ToolResult(**r) for r in recent_raw]
        last = recent[-1] if recent else None

        unsubmitted = [c for c in self._state.claims if c.get("claim_id") not in self._state.submitted_claim_ids]

        return MediBillObservation(
            done=done,
            reward=reward,
            metadata={},
            claims=previews,
            claims_remaining=len(unsubmitted),
            last_tool_result=last,
            recent_tool_results=recent,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            steps_remaining=self._state.max_steps - self._state.step_count,
            budget_spent=self._state.budget_spent,
            budget_remaining=self._state.budget_remaining,
            tool_costs=TOOL_COSTS,
            task_id=self._state.task_id,
            task_name=getattr(self, "_task_name", self._state.task_id),
            difficulty=getattr(self, "_difficulty", ""),
        )

    # ------------------------------------------------------------------
    # OpenEnv metadata
    # ------------------------------------------------------------------

    def get_metadata(self):  # type: ignore[override]
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="medibill",
            description=(
                "MediBill-Env: Indian cashless health-insurance claim "
                "reconciliation under IRDAI regulatory clock, with silent "
                "mid-episode policy-version drift as hero mechanic."
            ),
            version="0.1.0",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_entry(
    result: ToolResult,
    step: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Serialise a tool result into the persistent tool_log format.

    The grader consumes extra bookkeeping fields (``_step``, ``_params_cache``)
    that are not part of the public ToolResult model. Underscore-prefixed
    keys are never exposed to the agent (observations only show ToolResult).
    """
    entry = result.model_dump()
    entry["_step"] = step
    entry["_params_cache"] = dict(params)
    return entry


def _build_dirty_from_ground_truth(gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce the agent's starting claim state.

    Simulates a real discharge-summary arriving at the billing desk: identity
    fields and the raw medical codes (diagnosis_code, procedure_code) are
    already filled in by the treating physician. What remains for the billing
    agent is the **policy-compliance layer**: which version of the insurer
    rulebook applies right now, whether pre-auth is required under that
    version, which signatures are mandatory, whether a narrative is needed.

    This focuses the agent's task on the hero mechanic (policy-version-
    correct submission) rather than generic text-to-code mapping.
    """
    # Fields that always remain visible — part of the EHR record the agent
    # receives at the start of the episode.
    KEEP_VISIBLE = {"provider", "diagnosis_code", "procedure_code"}
    dirty: list[dict[str, Any]] = []
    for claim in gt:
        d = dict(claim)
        for f in POLICY_SENSITIVE_FIELDS:
            if f in KEEP_VISIBLE:
                continue
            if f == "policy_version":
                # Hide the current version — only insurance_lookup reveals it.
                d[f] = None
            elif f == "pre_auth_flag":
                d[f] = None
            elif f == "pre_auth_number":
                d[f] = None
            elif f == "required_signatures":
                d[f] = []
            elif f == "discharge_summary_attached":
                d[f] = None
            elif f == "diagnosis_narrative":
                d[f] = ""
            else:
                d[f] = None
        dirty.append(d)
    return dirty


def _public_claim_view(claim: dict[str, Any]) -> dict[str, Any]:
    """Strip hidden fields from a claim for agent-visible observations."""
    return {k: v for k, v in claim.items() if not k.startswith("_")}


def _recompute_policy_fields_inplace(
    ground_truth: list[dict[str, Any]],
    new_provider: str,
    new_version: str,
) -> None:
    """Mutate ground-truth policy-sensitive fields to match the new policy version.

    Only cells NOT yet locked are touched. Because submissions are locked in
    the agent-visible ``claims`` list (not ground truth), we update the full
    ground-truth list — the grader only compares unsubmitted claims against
    the fresh ground truth; submitted claims retain the version they were
    stamped with at submit time.
    """
    rules = get_provider(new_provider).at(new_version)
    for claim in ground_truth:
        amount = int(claim.get("amount_billed_inr") or 0)
        claim["policy_version"] = new_version
        claim["pre_auth_flag"] = amount >= rules.pre_auth_threshold_inr
        claim["pre_auth_number"] = (
            claim.get("pre_auth_number")
            if claim["pre_auth_flag"]
            else None
        )
        claim["required_signatures"] = list(rules.required_signatures)
        claim["discharge_summary_attached"] = rules.requires_discharge_summary
        if rules.requires_diagnosis_narrative and not claim.get("diagnosis_narrative"):
            claim["diagnosis_narrative"] = (
                f"Patient presented with clinical findings consistent with "
                f"{claim.get('diagnosis_code','unknown')}. Inpatient evaluation "
                f"and management performed per standard protocol."
            )
