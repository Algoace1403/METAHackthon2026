"""Typed WebSocket client for MediBill-Env.

This module is the SUPPORTED entry point for any multi-step interaction with a
running MediBill-Env server — inference, trajectory generation, rollout
harnesses, training loops.

Why not just call the REST endpoints with ``requests`` or ``curl``?
    OpenEnv's REST ``/step`` handler is STATELESS: every HTTP request
    constructs a fresh ``MediBillEnvironment`` via the env factory and closes
    it after the request returns. ``/step`` called without first running
    ``/reset`` on the same instance will land on an un-reset environment and
    fail with errors like ``"Policy version '' not found"``. Multi-step
    trajectories only compose when the client maintains a stateful session
    over WebSocket.

    ``MediBillEnv`` wraps the stateful WebSocket path behind four methods —
    ``reset``, ``step``, ``state``, ``close`` — so downstream code never has
    to think about the transport.

Usage:
    from medibill.client import MediBillEnv
    from medibill.models import MediBillAction

    with MediBillEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(seed=42, task_id="easy_cashless")
        obs = result.observation

        while not result.done:
            result = env.step(MediBillAction(
                action_type="insurance_lookup",
                params={"provider": obs.claims[0].provider},
            ))
            obs = result.observation

The class inherits ``reset``, ``step``, ``state``, ``close``, and context-
manager behaviour from :class:`openenv.core.env_client.EnvClient`. Subclassing
only supplies the three serialisation hooks the base class needs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from medibill.models import (
    ClaimPreview,
    MediBillAction,
    MediBillObservation,
    MediBillState,
    ToolResult,
)


class MediBillEnv(EnvClient[MediBillAction, MediBillObservation, MediBillState]):
    """Stateful client for a running MediBill-Env server.

    Parameters
    ----------
    base_url:
        Root URL of the server (e.g. ``"http://localhost:8000"``).

    Notes
    -----
    All multi-step work MUST go through this client (or any other class that
    inherits from :class:`EnvClient`). Ad-hoc ``requests.post(".../step")``
    will run against a fresh environment per call and produce nonsense.
    """

    # ------------------------------------------------------------------
    # EnvClient serialisation hooks
    # ------------------------------------------------------------------

    def _step_payload(self, action: MediBillAction) -> Dict[str, Any]:
        """Serialise a :class:`MediBillAction` to the wire format."""
        return {
            "action_type": action.action_type,
            "params": action.params,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[MediBillObservation]:
        """Parse a step / reset response into a typed :class:`StepResult`."""
        obs_data: Dict[str, Any] = payload.get("observation", {})
        observation = _build_observation(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MediBillState:
        """Parse a ``/state`` response into a typed :class:`MediBillState`."""
        return MediBillState(**payload)


# ---------------------------------------------------------------------------
# Private helpers — kept outside the class to stay under 50 lines each.
# ---------------------------------------------------------------------------


def _parse_tool_result(raw: Optional[Dict[str, Any]]) -> Optional[ToolResult]:
    """Convert a raw tool-result dict into a :class:`ToolResult`, or ``None``."""
    if raw is None:
        return None
    return ToolResult(**raw)


def _parse_tool_results(
    raw_list: List[Dict[str, Any]],
) -> List[ToolResult]:
    """Convert a list of raw tool-result dicts into :class:`ToolResult` models."""
    return [ToolResult(**r) for r in raw_list]


def _parse_claims(raw_list: List[Dict[str, Any]]) -> List[ClaimPreview]:
    """Convert a list of raw claim-preview dicts into :class:`ClaimPreview`."""
    return [ClaimPreview(**c) for c in raw_list]


def _build_observation(obs_data: Dict[str, Any]) -> MediBillObservation:
    """Construct a fully-typed :class:`MediBillObservation` from raw JSON.

    Nested models (:class:`ClaimPreview`, :class:`ToolResult`) are parsed
    explicitly so that callers always receive validated Pydantic objects.
    """
    return MediBillObservation(
        # Claim state
        claims=_parse_claims(obs_data.get("claims", []) or []),
        claims_remaining=obs_data.get("claims_remaining", 0),
        # Tool-call history
        last_tool_result=_parse_tool_result(obs_data.get("last_tool_result")),
        recent_tool_results=_parse_tool_results(
            obs_data.get("recent_tool_results", []) or []
        ),
        # Step context
        step_number=obs_data.get("step_number", 0),
        max_steps=obs_data.get("max_steps", 30),
        steps_remaining=obs_data.get("steps_remaining", 30),
        # Budget
        budget_spent=obs_data.get("budget_spent", 0.0),
        budget_remaining=obs_data.get("budget_remaining", 40.0),
        tool_costs=obs_data.get("tool_costs", {}),
        # Task metadata
        task_id=obs_data.get("task_id", ""),
        task_name=obs_data.get("task_name", ""),
        difficulty=obs_data.get("difficulty", ""),
        # Inherited Observation fields
        done=obs_data.get("done", False),
        reward=obs_data.get("reward"),
        metadata=obs_data.get("metadata", {}),
    )
