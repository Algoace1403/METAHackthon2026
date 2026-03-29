"""Baseline inference script for DataClean-Env hackathon.

Runs an LLM-based data-cleaning agent through all three tasks
(easy_contacts, medium_employees, hard_patients), collects scores,
and prints a JSON results block for the validator.

Environment variables required:
    API_BASE_URL  - LLM endpoint URL
    MODEL_NAME    - model identifier (e.g. "meta-llama/Llama-3-8B-Instruct")
    HF_TOKEN      - API key for the LLM endpoint

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

from dataclean_env import DataCleanEnv, DataCleanAction
from dataclean_env.models import DataCleanObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# URL of the DataClean-Env environment server.
# The validator may set ENV_BASE_URL, or we fall back to localhost.
# IMPORTANT: Update the fallback URL to your deployed HF Space before submission.
ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL",
    "http://localhost:8000",
)

TASKS: List[str] = ["easy_contacts", "medium_employees", "hard_patients"]
SEED: int = 42
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 1024
GLOBAL_TIMEOUT_SECONDS: int = 1100  # 18.3 min safety margin

# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Global timeout reached")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are a data-cleaning agent. Your job is to fix quality issues in a tabular dataset.

## Available Actions

Respond with ONLY a JSON object (no markdown, no explanation) containing:
- "action_type": one of the types below
- "params": a dict of parameters for that action

### Action Types

1. **fix_value** - Fix an incorrect value in a cell.
   params: {"row_id": <int>, "column": "<col_name>", "new_value": "<corrected_value>"}

2. **delete_row** - Delete a row (e.g. junk/duplicate that cannot be merged).
   params: {"row_id": <int>}

3. **fill_missing** - Fill a missing (null) value.
   params: {"row_id": <int>, "column": "<col_name>", "value": "<fill_value>"}

4. **standardize_format** - Standardize the format of ALL values in a column (column-level, not row-level).
   params: {"column": "<col_name>", "format_type": "<format>"}
   format_type options: "date:YYYY-MM-DD", "phone:US", "phone:E164", "name:title_case",
                        "email:lowercase", "zip:5digit", "currency:float", "state:abbreviation"

5. **merge_duplicates** - Merge two duplicate rows (keeps the first, deletes the second).
   params: {"row_id1": <int>, "row_id2": <int>, "strategy": "<merge_strategy>"}
   strategy options: "keep_first", "keep_second", "merge_prefer_nonnull", "merge_prefer_row1", "merge_prefer_row2"

6. **flag_anomaly** - Flag a suspicious value for review.
   params: {"row_id": <int>, "column": "<col_name>", "reason": "<why>"}

7. **split_column** - Split a column into multiple columns.
   params: {"column": "<col_name>", "delimiter": "<delim>", "new_names": ["<name1>", "<name2>"]}

8. **rename_column** - Rename a column.
   params: {"old_name": "<old>", "new_name": "<new>"}

9. **cast_type** - Cast a column to a different type.
   params: {"column": "<col_name>", "target_type": "<type>"}
   target_type options: "int", "float", "str", "bool", "date"

10. **escalate_to_human** - Escalate an ambiguous cell to human review when you are uncertain.
   params: {"row_id": <int>, "column": "<col_name>", "confidence": <float 0-1>, "reason": "<why>"}

11. **mark_complete** - Signal that you believe the dataset is clean.
   params: {}

## Important Rules

- Always reference rows by their **row_id** (the first column shown), NOT by row index.
- Examine the quality issues carefully and fix the most impactful ones first.
- When all issues are resolved (or you cannot fix more), use mark_complete.
- Respond with ONLY the JSON object. No extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(obs: DataCleanObservation) -> str:
    """Build the per-step user prompt from the observation."""
    parts: List[str] = []

    # Step info
    parts.append(
        f"## Step {obs.step_number} / {obs.max_steps} "
        f"({obs.steps_remaining} remaining)"
    )

    # Data summary
    ds = obs.data_summary
    parts.append(
        f"\n## Data Summary\n"
        f"- Rows: {ds.row_count}, Columns: {ds.column_count}\n"
        f"- Total cells: {ds.total_cells}, Null cells: {ds.null_count}\n"
        f"- Quality issues: {ds.issue_count}\n"
        f"- Columns: {', '.join(ds.columns)}"
    )

    # Quality issues (grouped, max 15)
    if obs.issue_groups:
        parts.append("\n## Quality Issues (grouped)")
        shown = 0
        for group in obs.issue_groups:
            parts.append(f"\n### {group.issue_type} ({group.count} issues)")
            for ex in group.examples:
                if shown >= 15:
                    break
                parts.append(
                    f"  - Row {ex.row_id}, col '{ex.column}': "
                    f"{ex.description}"
                    + (f" -> suggestion: {ex.suggestion}" if ex.suggestion else "")
                )
                shown += 1
            if shown >= 15:
                remaining = obs.issues_remaining - shown
                if remaining > 0:
                    parts.append(f"  ... and {remaining} more issues not shown")
                break

    # Last action result
    if obs.last_action_result is not None:
        ar = obs.last_action_result
        parts.append(
            f"\n## Last Action Result\n"
            f"- Action: {ar.action}, Status: {ar.status}\n"
            f"- Message: {ar.message}\n"
            f"- Cells modified: {ar.cells_modified}"
        )

    # Current data table
    if obs.rows:
        parts.append("\n## Current Data (row_id is first column)")
        header_cols = ["row_id"] + obs.columns
        parts.append("| " + " | ".join(str(c) for c in header_cols) + " |")
        parts.append("| " + " | ".join("---" for _ in header_cols) + " |")
        for row in obs.rows:
            parts.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(parts)


def _normalize_params(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize param aliases to canonical names expected by environment.py.

    Handles common LLM mistakes like using 'value' instead of 'new_value'
    for fix_value, or 'row_id_1'/'row_id_2' instead of 'row_id1'/'row_id2'.
    """
    p = dict(params)

    if action_type == "fix_value":
        # Map 'value' -> 'new_value' (only when 'new_value' is absent)
        if "value" in p and "new_value" not in p:
            p["new_value"] = p.pop("value")

    elif action_type == "merge_duplicates":
        # Map 'row_id_1' -> 'row_id1', 'row_id_2' -> 'row_id2'
        if "row_id_1" in p and "row_id1" not in p:
            p["row_id1"] = p.pop("row_id_1")
        if "row_id_2" in p and "row_id2" not in p:
            p["row_id2"] = p.pop("row_id_2")

    return p


def _parse_action(response_text: str) -> DataCleanAction:
    """Parse the LLM response into a DataCleanAction.

    Tries to extract a JSON object with an "action_type" key.
    Falls back to mark_complete if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            action_type = data["action_type"]
            params = _normalize_params(action_type, data.get("params", {}))
            return DataCleanAction(
                action_type=action_type,
                params=params,
            )
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]*\"action_type\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            action_type = data["action_type"]
            params = _normalize_params(action_type, data.get("params", {}))
            return DataCleanAction(
                action_type=action_type,
                params=params,
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Try nested braces (for params containing dicts)
    match = re.search(r"\{.*\"action_type\".*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "action_type" in data:
                action_type = data["action_type"]
                params = _normalize_params(action_type, data.get("params", {}))
                return DataCleanAction(
                    action_type=action_type,
                    params=params,
                )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback
    print(f"  [WARN] Could not parse LLM response, falling back to mark_complete")
    print(f"  [WARN] Raw response: {text[:200]}")
    return DataCleanAction(action_type="mark_complete", params={})


def _call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    retry: bool = True,
) -> str:
    """Call the LLM and return the assistant message content.

    Retries once on failure, then returns a fallback mark_complete JSON.
    """
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
        print(f"  [ERROR] LLM call failed: {e}")
        if retry:
            print("  [INFO] Retrying once...")
            time.sleep(1)
            return _call_llm(client, messages, retry=False)
        print("  [WARN] Retry failed, using fallback action")
        return '{"action_type": "mark_complete", "params": {}}'


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    env_base_url: str,
    task_id: str,
) -> float:
    """Run a single data-cleaning task and return the final score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    with DataCleanEnv(base_url=env_base_url).sync() as env:
        result = env.reset(seed=SEED, task_id=task_id)
        obs: DataCleanObservation = result.observation

        print(f"  Initial issues: {obs.data_summary.issue_count}")
        print(f"  Max steps: {obs.max_steps}")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        step = 0
        while not obs.done:
            step += 1
            user_msg = _build_user_prompt(obs)
            messages.append({"role": "user", "content": user_msg})

            # Keep conversation manageable: only system + last 6 exchanges
            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            llm_response = _call_llm(client, messages)
            messages.append({"role": "assistant", "content": llm_response})

            action = _parse_action(llm_response)
            print(
                f"  Step {step}: {action.action_type} "
                f"{json.dumps(action.params) if action.params else ''}"
            )

            try:
                result = env.step(action)
                obs = result.observation
            except Exception as e:
                print(f"  [ERROR] Environment step failed: {e}")
                traceback.print_exc()
                continue

        # Extract final score from reward or metadata
        score = 0.0
        if result.reward is not None:
            score = float(result.reward)
        elif obs.metadata and "score" in obs.metadata:
            score = float(obs.metadata["score"])

        print(f"  Final score: {score:.4f}")
        return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all tasks and print results."""
    # Validate environment variables
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL environment variable is not set")
        return 1
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is not set")
        return 1
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set")
        return 1

    print(f"Environment URL: {ENV_BASE_URL}")

    # Set global timeout (Unix only; no-op on Windows)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GLOBAL_TIMEOUT_SECONDS)

    # Initialize OpenAI client pointed at the provided endpoint
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    print(f"LLM endpoint: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")
    print(f"Tasks: {TASKS}")
    print(f"Seed: {SEED}, Temperature: {TEMPERATURE}")

    scores: Dict[str, float] = {}

    for task_id in TASKS:
        try:
            score = run_task(client, ENV_BASE_URL, task_id)
            scores[task_id] = score
        except TimeoutError:
            print(f"\n[TIMEOUT] Global timeout reached during task '{task_id}'")
            scores[task_id] = 0.0
            break
        except Exception as e:
            print(f"\n[ERROR] Task '{task_id}' failed: {e}")
            traceback.print_exc()
            scores[task_id] = 0.0

    # Cancel alarm if still active
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)

    # Print results
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    print(f"  ---")
    print(f"  Average: {avg_score:.4f}")
    print(f"{'='*60}")

    # JSON output block for validator parsing
    output = {
        "scores": scores,
        "average_score": round(avg_score, 4),
        "model": MODEL_NAME,
        "seed": SEED,
        "temperature": TEMPERATURE,
    }
    print(f"\n--- JSON OUTPUT ---")
    print(json.dumps(output, indent=2))
    print(f"--- END JSON OUTPUT ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
