"""Prompt formatting for MediBill-Env.

Turns a :class:`MediBillObservation` into the text an LLM agent sees, and
produces the system prompt that scopes the agent's job. Shared by
trajectory generation, SFT data preparation, and inference.

Design goals:
    * Deterministic — same observation produces the same text every call.
    * Compact — fits a 3B-model context window with room for ~30 turns of
      history (ballpark 2048 prompt tokens per step).
    * No provider/version leakage — the agent only sees what it would see
      in a live episode. Ground-truth fields are never surfaced here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List

from medibill.models import (
    AGENT_ACTION_TYPES,
    ClaimPreview,
    MediBillObservation,
    ToolResult,
)


SYSTEM_PROMPT: str = (
    "You are a medical billing coder at an Indian hospital working under "
    "IRDAI's 1-hour pre-authorization and 3-hour discharge clock. Your job "
    "is to finalize cashless insurance claims so they pass the insurer's "
    "current policy, then submit each claim.\n"
    "\n"
    "You have five tools. Call exactly one per step, emitting a single JSON "
    "object of the form:\n"
    '    {"action_type": "<tool>", "params": { ... }}\n'
    "\n"
    "Tools and their params:\n"
    '  * ehr_query        {"claim_id": <str>} OR {"patient_id": <str>}\n'
    '  * insurance_lookup {"provider": <str>}\n'
    '  * coding_engine    {"claim_id": <str>, "field": <str>, "value": <any>}\n'
    '  * escalate_to_human{"claim_id": <str>, "field": <str>, "reason": <str>}\n'
    '  * submit_claim     {"claim_id": <str>}\n'
    "\n"
    "Policy drift: the active insurance policy can change mid-episode without "
    "announcement. If a previous insurance_lookup response may be stale, call "
    "insurance_lookup again before submitting. submit_claim is graded against "
    "the policy active at submit time, not the one you remember.\n"
    "\n"
    "Budget: every tool call costs a small amount. Spend it on the work that "
    "actually improves the claim. Finish all claims within the step budget."
)


# ---------------------------------------------------------------------------
# Prompt version handshake
# ---------------------------------------------------------------------------
# Hash-derived identifier that changes whenever SYSTEM_PROMPT changes.
# Every trajectory generator records this in each trajectory; downstream
# trainers refuse to load traces whose ``prompt_version`` does not match
# the installed package's ``PROMPT_VERSION``. This prevents silent drift
# between the prompt used to generate SFT data and the prompt shown to the
# model at inference time.

PROMPT_VERSION: str = (
    "sp-" + hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
)


# ---------------------------------------------------------------------------
# Observation -> text
# ---------------------------------------------------------------------------


def format_observation(obs: MediBillObservation, *, claim_cap: int = 6) -> str:
    """Render a MediBillObservation as the text the agent sees at this step.

    Parameters
    ----------
    obs:
        The current observation.
    claim_cap:
        Maximum number of unsubmitted claims to display in the claim table.
        Keeps the prompt bounded on hard tasks with many claims.
    """
    lines: List[str] = []
    lines.append(
        f"[Task] {obs.task_name or obs.task_id} "
        f"(difficulty={obs.difficulty})"
    )
    lines.append(
        f"[Step] {obs.step_number}/{obs.max_steps}  "
        f"remaining={obs.steps_remaining}  "
        f"budget={obs.budget_remaining:.1f}/"
        f"{obs.budget_remaining + obs.budget_spent:.1f}  "
        f"claims_unsubmitted={obs.claims_remaining}"
    )

    if obs.last_tool_result is not None:
        lines.append("")
        lines.extend(_format_tool_result(obs.last_tool_result))

    lines.append("")
    lines.append("[Claims]")
    lines.extend(_format_claim_table(obs.claims, claim_cap=claim_cap))

    lines.append("")
    lines.append(
        f"[Your turn] Emit a single JSON action "
        f"(allowed: {', '.join(AGENT_ACTION_TYPES)})."
    )
    return "\n".join(lines)


def _format_tool_result(result: ToolResult) -> List[str]:
    lines = [f"[Last tool] {result.tool} → {result.status}"]
    if result.message:
        lines.append(f"  msg: {result.message}")
    if result.payload:
        payload_str = _compact_json(result.payload, max_len=600)
        lines.append(f"  payload: {payload_str}")
    return lines


def _format_claim_table(claims: List[ClaimPreview], *, claim_cap: int) -> List[str]:
    unsubmitted = [c for c in claims if not c.submitted]
    shown = unsubmitted[:claim_cap]
    lines: List[str] = []
    for c in shown:
        lines.append(
            f"  - {c.claim_id}  provider={c.provider}  "
            f"amount_inr={c.amount_billed_inr}  "
            f"dx={c.diagnosis_code}  proc={c.procedure_code}  "
            f"policy_version={c.policy_version}  "
            f"pre_auth_flag={c.pre_auth_flag}  "
            f"sigs={len(c.required_signatures) if isinstance(c.required_signatures, list) else '?'}  "
            f"summary={c.discharge_summary_attached}  "
            f"narrative={'set' if c.diagnosis_narrative else 'empty'}"
        )
    if len(unsubmitted) > claim_cap:
        lines.append(
            f"  ... and {len(unsubmitted) - claim_cap} more unsubmitted claim(s)"
        )
    if not shown:
        lines.append("  (all claims submitted)")
    return lines


def _compact_json(obj: Any, *, max_len: int) -> str:
    """Serialise an object to JSON truncated to ``max_len`` chars.

    Truncation is character-based, not token-based — good enough for
    prompt budgeting and cheaper than calling a tokenizer.
    """
    try:
        s = json.dumps(obj, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"...(+{len(s) - max_len}ch)"
    return s
