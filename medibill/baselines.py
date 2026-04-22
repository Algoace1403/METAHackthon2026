"""Baseline policies for MediBill-Env grader validation.

Three reference agents with different skill levels. Running them on the same
task should produce clearly separated score distributions — that is the
grader's score-separation gate (spec v3 §6).

    random_agent:        Uniform-random tool selection.
    no_op_agent:         Submit every claim immediately without any other work.
    scripted_heuristic:  Query EHR, look up insurance, fill in policy fields
                         using the returned rules, then submit.

These are intentionally simple — the validation gate is about whether the
GRADER separates obviously-different behaviours, not about building a strong
policy.
"""

from __future__ import annotations

import random
from typing import Callable

from medibill.data_generator import POLICY_SENSITIVE_FIELDS
from medibill.models import AGENT_ACTION_TYPES, MediBillAction, MediBillObservation
from medibill.server.environment import MediBillEnvironment

Agent = Callable[[MediBillObservation, MediBillEnvironment, random.Random], MediBillAction]


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------


def _provider_from_obs(obs: MediBillObservation) -> str:
    """Read provider from the agent-visible claim previews.

    Baselines must be tool-faithful: they are only allowed to use observations
    and tool-result payloads, never the internal ``env.state``. Provider is
    part of the dirty claim so it is agent-visible.
    """
    for c in obs.claims or []:
        if c.provider:
            return str(c.provider)
    return "CGHS"  # conservative fallback; hits when obs.claims is empty


def random_agent(
    obs: MediBillObservation,
    env: MediBillEnvironment,  # unused; kept to match the Agent signature
    rng: random.Random,
) -> MediBillAction:
    action_type = rng.choice(AGENT_ACTION_TYPES)
    params: dict = {}
    claims = obs.claims or []
    if action_type == "ehr_query":
        if claims:
            params = {"claim_id": rng.choice(claims).claim_id}
    elif action_type == "insurance_lookup":
        params = {"provider": _provider_from_obs(obs)}
    elif action_type == "coding_engine":
        if claims:
            params = {
                "claim_id": rng.choice(claims).claim_id,
                "field": rng.choice(POLICY_SENSITIVE_FIELDS),
                "value": rng.choice(["X", "Y", None, "PROC-GEN-001", "I10"]),
            }
    elif action_type == "escalate_to_human":
        if claims:
            params = {
                "claim_id": rng.choice(claims).claim_id,
                "field": rng.choice(POLICY_SENSITIVE_FIELDS),
                "reason": "random",
            }
    elif action_type == "submit_claim":
        unsub = [c for c in claims if not c.submitted]
        if unsub:
            params = {"claim_id": rng.choice(unsub).claim_id}
    return MediBillAction(action_type=action_type, params=params)


# ---------------------------------------------------------------------------
# No-op (immediate submit everything, skip all work)
# ---------------------------------------------------------------------------


def no_op_agent(
    obs: MediBillObservation,
    env: MediBillEnvironment,  # unused
    rng: random.Random,
) -> MediBillAction:
    unsub = [c for c in (obs.claims or []) if not c.submitted]
    if unsub:
        return MediBillAction(
            action_type="submit_claim",
            params={"claim_id": unsub[0].claim_id},
        )
    # Nothing left — any idle call; env will treat as no-op.
    return MediBillAction(
        action_type="insurance_lookup",
        params={"provider": _provider_from_obs(obs)},
    )


# ---------------------------------------------------------------------------
# Scripted heuristic (the floor-of-competence reference)
# ---------------------------------------------------------------------------


class ScriptedHeuristicPolicy:
    """Stateful rule-based policy — strictly tool-faithful.

    Only uses the agent-visible observation and the payload of the most recent
    ``insurance_lookup`` tool call. Never reaches into ``env.state`` or
    directly imports provider packs. This is what a weak SFT checkpoint
    should plausibly learn to do.

    Does one insurance_lookup up front, caches the returned rules, then for
    each claim: runs ``ehr_query`` to pull the record, fills in the six
    policy-sensitive fields that the dirty state leaves blank (policy_version,
    pre_auth_flag, pre_auth_number, required_signatures,
    discharge_summary_attached, diagnosis_narrative), then submits.

    The explicit ``ehr_query`` phase is present so scripted trajectories
    cover all five agent-visible tools. Without it, SFT data generated from
    this policy would never teach the model to use ``ehr_query``.
    """

    def __init__(self) -> None:
        self._looked_up: bool = False
        self._rules_cache: dict | None = None
        self._phase: dict[str, str] = {}

    _PHASES = (
        "ehr",
        "set_version",
        "set_preauth_flag",
        "set_preauth_number",
        "set_signatures",
        "set_summary",
        "set_narrative",
        "submit",
    )

    def _next_phase(self, current: str) -> str:
        idx = self._PHASES.index(current)
        return self._PHASES[min(idx + 1, len(self._PHASES) - 1)]

    def __call__(
        self,
        obs: MediBillObservation,
        env: MediBillEnvironment,  # unused; tool-faithful policies ignore env.state
        rng: random.Random,
    ) -> MediBillAction:
        # Absorb any insurance_lookup payload that has just arrived
        if (
            obs.last_tool_result
            and obs.last_tool_result.tool == "insurance_lookup"
            and obs.last_tool_result.status == "success"
        ):
            rules = obs.last_tool_result.payload.get("rules")
            if isinstance(rules, dict):
                self._rules_cache = rules

        if not self._looked_up:
            self._looked_up = True
            return MediBillAction(
                action_type="insurance_lookup",
                params={"provider": _provider_from_obs(obs)},
            )

        if self._rules_cache is None:
            # Lookup result somehow missing — try again rather than cheat.
            return MediBillAction(
                action_type="insurance_lookup",
                params={"provider": _provider_from_obs(obs)},
            )

        remaining = [c for c in obs.claims if not c.submitted]
        if not remaining:
            return MediBillAction(
                action_type="insurance_lookup",
                params={"provider": _provider_from_obs(obs)},
            )

        claim = remaining[0]
        cid = claim.claim_id
        phase = self._phase.get(cid, "ehr")

        if phase == "ehr":
            self._phase[cid] = self._next_phase(phase)
            return MediBillAction(
                action_type="ehr_query",
                params={"claim_id": cid},
            )

        if phase == "set_version":
            self._phase[cid] = self._next_phase(phase)
            return MediBillAction(
                action_type="coding_engine",
                params={
                    "claim_id": cid,
                    "field": "policy_version",
                    "value": self._rules_cache.get("policy_version", ""),
                },
            )
        amount = claim.amount_billed_inr or 0
        threshold = self._rules_cache.get("pre_auth_threshold_inr", 0)
        flag = amount >= threshold
        if phase == "set_preauth_flag":
            self._phase[cid] = self._next_phase(phase)
            return MediBillAction(
                action_type="coding_engine",
                params={"claim_id": cid, "field": "pre_auth_flag", "value": flag},
            )
        if phase == "set_preauth_number":
            self._phase[cid] = self._next_phase(phase)
            provider = _provider_from_obs(obs)
            val = f"PA-{provider}-{cid.rsplit('-', 1)[-1]}" if flag else None
            return MediBillAction(
                action_type="coding_engine",
                params={"claim_id": cid, "field": "pre_auth_number", "value": val},
            )
        if phase == "set_signatures":
            self._phase[cid] = self._next_phase(phase)
            return MediBillAction(
                action_type="coding_engine",
                params={
                    "claim_id": cid,
                    "field": "required_signatures",
                    "value": list(self._rules_cache.get("required_signatures", [])),
                },
            )
        if phase == "set_summary":
            self._phase[cid] = self._next_phase(phase)
            return MediBillAction(
                action_type="coding_engine",
                params={
                    "claim_id": cid,
                    "field": "discharge_summary_attached",
                    "value": bool(self._rules_cache.get("requires_discharge_summary", False)),
                },
            )
        if phase == "set_narrative":
            self._phase[cid] = self._next_phase(phase)
            narrative = (
                f"Patient presented with clinical findings consistent with "
                f"{claim.diagnosis_code or 'UNKNOWN'}. Inpatient evaluation "
                f"and management performed per standard protocol."
                if self._rules_cache.get("requires_diagnosis_narrative") else ""
            )
            return MediBillAction(
                action_type="coding_engine",
                params={
                    "claim_id": cid,
                    "field": "diagnosis_narrative",
                    "value": narrative,
                },
            )
        # submit
        self._phase[cid] = "done"
        return MediBillAction(action_type="submit_claim", params={"claim_id": cid})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_episode(
    agent: Agent,
    task_id: str,
    seed: int,
) -> tuple[MediBillEnvironment, MediBillObservation]:
    """Run one episode of *agent* on *task_id*. Returns the final env + obs."""
    env = MediBillEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    rng = random.Random(seed)
    done = False
    safety = env.state.max_steps + 5
    while not done and safety > 0:
        action = agent(obs, env, rng)
        obs = env.step(action)
        done = bool(obs.done)
        safety -= 1
    return env, obs
