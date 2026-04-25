"""Baseline LLM inference script for MediBill-Env.

Runs an LLM-based medical-billing agent through the three MediBill tasks
(easy_cashless, medium_multi_payer, hard_drift), collects per-task scores,
and emits [START]/[STEP]/[END] lines to stdout per OpenEnv convention.

Environment variables:
    API_BASE_URL  — LLM endpoint URL (default: https://api.openai.com/v1)
    MODEL_NAME    — model identifier (default: gpt-4.1-mini)
    HF_TOKEN      — bearer token forwarded as the API key (mandatory)
    ENV_BASE_URL  — running MediBill-Env server (default: http://localhost:8000)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from medibill.client import MediBillEnv
from medibill.models import (
    AGENT_ACTION_TYPES,
    MediBillAction,
    MediBillObservation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS: List[str] = ["easy_cashless", "medium_multi_payer", "hard_drift"]
BENCHMARK: str = "medibill"
SEED: int = 42
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 1024
GLOBAL_TIMEOUT_SECONDS: int = 1100  # 18.3 min safety margin
SUCCESS_SCORE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------


class GlobalTimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise GlobalTimeoutError("Global timeout reached")


# ---------------------------------------------------------------------------
# System prompt — 5 MediBill actions with exact param schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are a medical billing agent reconciling cashless insurance claims under IRDAI deadlines.

GOAL: For each claim, look up the currently-active insurance policy, then fill in the policy-sensitive fields using the rules returned by the lookup. Then submit each claim.

CRITICAL: On hard tasks the active policy may silently change mid-episode. There is NO announcement. If you suspect drift (your last lookup looks stale, or new submissions start failing), call insurance_lookup again. Claims are graded against whatever policy is active at submit_claim time.

RESPOND WITH ONLY A JSON OBJECT (no markdown, no prose):
{"action_type": "<one of 5 below>", "params": {...}}

ACTIONS:

1. ehr_query — Read the patient record for a claim.
   params: {"claim_id": "CLAIM-PROVIDER-0001"}  OR  {"patient_id": "PAT-0001"}

2. insurance_lookup — Fetch the currently-active policy rules. Returns: pre_auth_threshold_inr, required_signatures (list), discharge_summary_required (bool), narrative_required (bool), policy_version (str). CALL THIS BEFORE ANY coding_engine.
   params: {"provider": "<provider_name from claim>"}

3. coding_engine — Write one field on one claim.
   params: {"claim_id": "CLAIM-PROVIDER-0001", "field": "<field_name>", "value": <any>}
   Editable fields: policy_version, pre_auth_flag, pre_auth_number, required_signatures, discharge_summary_attached, diagnosis_narrative, diagnosis_code, procedure_code.

4. escalate_to_human — Use when a cell is genuinely ambiguous and you cannot decide.
   params: {"claim_id": "CLAIM-PROVIDER-0001", "field": "diagnosis_code", "reason": "ambiguous chart"}

5. submit_claim — Lock and grade one claim. The claim is frozen after this.
   params: {"claim_id": "CLAIM-PROVIDER-0001"}

WORKFLOW (recommended):
- Step 1: insurance_lookup with the provider from the first claim.
- For each claim: (optional) ehr_query → coding_engine for each blank policy field → submit_claim.
- If your last insurance_lookup was many steps ago AND you are about to submit, lookup again — it might have drifted.

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(obs: MediBillObservation) -> str:
    """Build the per-step user prompt from a MediBillObservation."""
    parts: List[str] = []

    parts.append(
        f"## Step {obs.step_number} / {obs.max_steps} "
        f"({obs.steps_remaining} remaining)"
    )
    parts.append(f"Task: {obs.task_id} ({obs.difficulty})")
    total_budget = obs.budget_remaining + obs.budget_spent
    parts.append(
        f"Budget: {obs.budget_remaining:.1f} / {total_budget:.1f} remaining "
        f"(tool costs: {json.dumps(obs.tool_costs)})"
    )

    # Last tool result — esp. insurance_lookup payload (the rules!)
    if obs.last_tool_result is not None:
        tr = obs.last_tool_result
        parts.append(f"\n## Last tool: {tr.tool} → {tr.status}")
        if tr.message:
            parts.append(f"Message: {tr.message}")
        if tr.payload:
            payload_str = json.dumps(tr.payload, default=str)
            if len(payload_str) > 800:
                payload_str = payload_str[:800] + "...(truncated)"
            parts.append(f"Payload: {payload_str}")

    # Recent tool history (last 3, excluding the most recent which is above)
    if len(obs.recent_tool_results) > 1:
        parts.append("\n## Recent tool history (older first)")
        for tr in obs.recent_tool_results[-4:-1]:
            parts.append(f"- {tr.tool} → {tr.status}: {tr.message[:120]}")

    # Claims under construction
    parts.append(
        f"\n## Claims ({obs.claims_remaining} unsubmitted of {len(obs.claims)})"
    )
    for c in obs.claims:
        status = "SUBMITTED" if c.submitted else "open"
        narrative = "set" if c.diagnosis_narrative else "BLANK"
        sigs = c.required_signatures if c.required_signatures else "BLANK"
        parts.append(
            f"- {c.claim_id} [{status}] provider={c.provider} "
            f"amount_inr={c.amount_billed_inr}\n"
            f"   policy_version={c.policy_version} pre_auth_flag={c.pre_auth_flag} "
            f"pre_auth_number={c.pre_auth_number}\n"
            f"   required_signatures={sigs} discharge_summary={c.discharge_summary_attached} "
            f"narrative={narrative}\n"
            f"   diagnosis_code={c.diagnosis_code} procedure_code={c.procedure_code}"
        )

    parts.append(
        '\n## Respond with ONLY a JSON object: {"action_type": "...", "params": {...}}'
    )
    return "\n".join(parts)


def _normalize_params(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common LLM param-name aliases for MediBill actions."""
    p = dict(params)

    # Universal aliases
    if "id" in p and "claim_id" not in p and action_type != "ehr_query":
        p["claim_id"] = p.pop("id")
    if "patient" in p and "patient_id" not in p:
        p["patient_id"] = p.pop("patient")

    if action_type == "coding_engine":
        if "field_name" in p and "field" not in p:
            p["field"] = p.pop("field_name")
        if "new_value" in p and "value" not in p:
            p["value"] = p.pop("new_value")

    if action_type == "insurance_lookup":
        if "provider_name" in p and "provider" not in p:
            p["provider"] = p.pop("provider_name")

    return p


def _fallback_action(obs: Optional[MediBillObservation]) -> MediBillAction:
    """Forward-progress fallback when LM output cannot be parsed.

    Submits the first unsubmitted claim if any exist; otherwise calls
    insurance_lookup. Never gets stuck in a no-op loop.
    """
    if obs is not None and obs.claims:
        for c in obs.claims:
            if not c.submitted:
                return MediBillAction(
                    action_type="submit_claim",
                    params={"claim_id": c.claim_id},
                )
        provider = next(
            (c.provider for c in obs.claims if c.provider), "CGHS"
        )
        return MediBillAction(
            action_type="insurance_lookup",
            params={"provider": str(provider)},
        )
    return MediBillAction(
        action_type="insurance_lookup",
        params={"provider": "CGHS"},
    )


def _parse_action(
    response_text: str, obs: Optional[MediBillObservation]
) -> MediBillAction:
    """Parse the LLM response into a MediBillAction.

    Tries to extract a JSON object with an ``action_type`` key. Falls back
    to a forward-progress submit if parsing fails or the action_type is not
    one of the 5 valid AGENT_ACTION_TYPES.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    candidates: List[str] = [text]
    # Try to find a JSON object embedded in prose
    m = re.search(r"\{.*\"action_type\".*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "action_type" not in data:
            continue
        action_type = str(data["action_type"]).strip()
        if action_type not in AGENT_ACTION_TYPES:
            continue
        params = _normalize_params(action_type, data.get("params") or {})
        return MediBillAction(action_type=action_type, params=params)

    print(
        f"  [WARN] Could not parse LLM response, using forward-progress fallback",
        file=sys.stderr,
    )
    print(f"  [WARN] Raw: {text[:200]}", file=sys.stderr)
    return _fallback_action(obs)


def _call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    retry: bool = True,
) -> str:
    """Call the LLM and return the assistant message content."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}", file=sys.stderr)
        if retry:
            print("  [INFO] Retrying once...", file=sys.stderr)
            time.sleep(1)
            return _call_llm(client, messages, retry=False)
        print("  [WARN] Retry failed; emitting fallback action", file=sys.stderr)
        return '{"action_type": "submit_claim", "params": {}}'


# ---------------------------------------------------------------------------
# Stdout logging (mandatory format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(client: OpenAI, env_base_url: str, task_id: str) -> float:
    """Run a single MediBill task and return the final composite score."""
    print(f"\n  Task: {task_id}", file=sys.stderr)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with MediBillEnv(base_url=env_base_url).sync() as env:
            result = env.reset(seed=SEED, task_id=task_id)
            obs: MediBillObservation = result.observation

            print(f"  Claims: {len(obs.claims)}", file=sys.stderr)
            print(f"  Max steps: {obs.max_steps}", file=sys.stderr)
            print(f"  Budget: {obs.budget_remaining}", file=sys.stderr)

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            step = 0
            done = False
            while not done:
                step += 1
                user_msg = _build_user_prompt(obs)
                messages.append({"role": "user", "content": user_msg})

                # Keep conversation manageable: system + last 12 messages
                if len(messages) > 13:
                    messages = [messages[0]] + messages[-12:]

                llm_response = _call_llm(client, messages)
                messages.append({"role": "assistant", "content": llm_response})

                action = _parse_action(llm_response, obs)
                action_str = (
                    f"{action.action_type}("
                    f"{json.dumps(action.params, default=str) if action.params else ''})"
                )

                error: Optional[str] = None
                try:
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    reward = (
                        result.reward if result.reward is not None else 0.0
                    )
                except Exception as e:
                    print(
                        f"  [ERROR] env.step failed: {e}", file=sys.stderr
                    )
                    traceback.print_exc(file=sys.stderr)
                    reward = 0.0
                    error = str(e)
                    done = True

                rewards.append(float(reward))
                steps_taken = step

                log_step(
                    step=step,
                    action=action_str,
                    reward=float(reward),
                    done=done,
                    error=error,
                )

            score = (
                float(result.reward)
                if result is not None and result.reward is not None
                else 0.0
            )
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"  [ERROR] Task failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    log_end(success=success, steps=steps_taken, rewards=rewards)
    print(f"  Final score: {score:.4f}", file=sys.stderr)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GLOBAL_TIMEOUT_SECONDS)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"LLM endpoint: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Environment: {ENV_BASE_URL}", file=sys.stderr)
    print(f"Tasks: {TASKS}", file=sys.stderr)

    scores: Dict[str, float] = {}

    for task_id in TASKS:
        try:
            score = run_task(client, ENV_BASE_URL, task_id)
            scores[task_id] = score
        except GlobalTimeoutError:
            print(
                f"\n[TIMEOUT] Global timeout reached at task '{task_id}'",
                file=sys.stderr,
            )
            scores[task_id] = 0.0
            break
        except Exception as e:
            print(
                f"\n[ERROR] Task '{task_id}' failed: {e}", file=sys.stderr
            )
            traceback.print_exc(file=sys.stderr)
            scores[task_id] = 0.0

    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  RESULTS", file=sys.stderr)
    for task_id, s in scores.items():
        print(f"  {task_id}: {s:.4f}", file=sys.stderr)
    print(f"  Average: {avg_score:.4f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
