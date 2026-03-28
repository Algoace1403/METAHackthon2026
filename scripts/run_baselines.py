"""Run LLM baselines against the DataClean-Env server.

Requires:
  1. A running DataClean-Env server (e.g. `python -m dataclean_env.server`)
  2. An LLM inference endpoint (vLLM, TGI, OpenAI-compatible, etc.)

Environment variables:
  API_BASE_URL  - DataClean-Env server URL (default: http://localhost:8000)
  MODEL_NAME    - Model identifier for the LLM endpoint (e.g. "meta-llama/Llama-3-8B")
  HF_TOKEN      - HuggingFace token (if needed for gated models)
  LLM_BASE_URL  - LLM inference endpoint (default: http://localhost:8001/v1)

Usage:
    API_BASE_URL=http://localhost:8000 MODEL_NAME=gpt-4 python3 scripts/run_baselines.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Bootstrap: install openenv mock if the real package is absent.
# Note: the DataCleanEnv *client* needs a real running server at runtime,
# but the mock lets the module import succeed for validation and --help.
# ---------------------------------------------------------------------------


def _ensure_openenv_mock() -> None:
    """Install a lightweight openenv mock into sys.modules if needed."""
    try:
        import openenv.core.env_server  # noqa: F401
        return
    except ImportError:
        pass

    from types import ModuleType

    class _Base:
        def __init__(self, **kw: object) -> None:
            for k, v in kw.items():
                setattr(self, k, v)

    class _Environment:
        def __init__(self) -> None:
            pass

        def __class_getitem__(cls, item):  # type: ignore[override]
            return cls

    class _EnvClient:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def __class_getitem__(cls, item):  # type: ignore[override]
            return cls

    names = [
        "openenv", "openenv.core", "openenv.core.env_server",
        "openenv.core.env_server.types", "openenv.core.env_client",
        "openenv.core.client_types",
    ]
    mods = {n: ModuleType(n) for n in names}
    for n, m in mods.items():
        sys.modules[n] = m

    mods["openenv"].core = mods["openenv.core"]  # type: ignore[attr-defined]
    mods["openenv.core"].env_server = mods["openenv.core.env_server"]  # type: ignore[attr-defined]
    mods["openenv.core"].env_client = mods["openenv.core.env_client"]  # type: ignore[attr-defined]
    mods["openenv.core"].client_types = mods["openenv.core.client_types"]  # type: ignore[attr-defined]

    for attr in ("Action", "Observation", "State"):
        setattr(mods["openenv.core.env_server"], attr, type(attr, (_Base,), {}))
    setattr(mods["openenv.core.env_server"], "Environment", _Environment)
    setattr(mods["openenv.core.env_server.types"], "EnvironmentMetadata", _Base)
    setattr(mods["openenv.core.env_client"], "EnvClient", _EnvClient)
    setattr(mods["openenv.core.client_types"], "StepResult", _Base)


_ensure_openenv_mock()

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TASK_IDS = ["easy_contacts", "medium_employees", "hard_patients"]


# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------

def _check_prerequisites() -> bool:
    """Check that required config is available. Returns True if OK."""
    ok = True

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME env var is not set.")
        print("  Example: MODEL_NAME=gpt-4 python3 scripts/run_baselines.py")
        ok = False

    try:
        from dataclean_env.client import DataCleanEnv  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: Cannot import DataCleanEnv client: {exc}")
        print("  Install the package: pip install -e .")
        ok = False

    try:
        import httpx  # noqa: F401
    except ImportError:
        print("WARNING: httpx not installed. Install with: pip install httpx")
        print("  The client depends on httpx for HTTP transport.")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# LLM interaction (stub -- replace with your inference logic)
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Call the LLM endpoint and return the completion text.

    This is a stub. Replace the body with your preferred inference method:
      - OpenAI-compatible: POST to LLM_BASE_URL/chat/completions
      - HuggingFace TGI: POST to LLM_BASE_URL/generate
      - vLLM: POST to LLM_BASE_URL/chat/completions

    The prompt contains the observation as JSON. The LLM should return a
    JSON object with keys "action_type" and "params".
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required. Install with: pip install httpx")

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a data cleaning agent. Given a dataset observation, "
                    "return a JSON action with keys 'action_type' and 'params'. "
                    "Available actions: fix_value, delete_row, fill_missing, "
                    "standardize_format, merge_duplicates, flag_anomaly, "
                    "split_column, rename_column, cast_type, escalate_to_human, "
                    "mark_complete."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    resp = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_action(text: str) -> Dict[str, Any]:
    """Parse an LLM response into an action dict.

    Expects JSON with "action_type" and "params" keys.
    Falls back to mark_complete if parsing fails.
    """
    # Try to extract JSON from the response (handle markdown code blocks)
    cleaned = text.strip()
    if "```" in cleaned:
        # Extract content between first pair of triple backticks
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            # Remove optional language tag on first line
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if "action_type" in parsed:
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: mark_complete
    print(f"  WARNING: Could not parse LLM response, falling back to mark_complete")
    return {"action_type": "mark_complete", "params": {}}


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """Run one LLM-driven episode. Returns the final score."""
    from dataclean_env.client import DataCleanEnv
    from dataclean_env.models import DataCleanAction

    with DataCleanEnv(base_url=API_BASE_URL).sync() as env:
        result = env.reset(task_id=task_id)
        obs = result.observation

        step = 0
        while not obs.done:
            # Build prompt from observation
            prompt_data = {
                "task_id": obs.task_id,
                "step": obs.step_number,
                "steps_remaining": obs.steps_remaining,
                "row_count": obs.row_count,
                "columns": obs.columns,
                "issues_remaining": obs.issues_remaining,
                "quality_issues": [
                    {
                        "row_id": qi.row_id,
                        "column": qi.column,
                        "issue_type": qi.issue_type,
                        "description": qi.description,
                        "suggestion": qi.suggestion,
                    }
                    for qi in obs.quality_issues[:20]  # Cap for context length
                ],
                "rows": obs.rows[:15],  # Cap for context length
            }
            prompt = json.dumps(prompt_data, indent=2, default=str)

            # Get LLM action
            llm_text = call_llm(prompt)
            action_dict = parse_llm_action(llm_text)

            action = DataCleanAction(
                action_type=action_dict["action_type"],
                params=action_dict.get("params", {}),
            )

            result = env.step(action)
            obs = result.observation
            step += 1
            print(f"    Step {step}: {action_dict['action_type']} -> reward={obs.reward}")

    return obs.reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_prerequisites():
        print("\nFix the issues above and try again.")
        sys.exit(1)

    print(f"Model:      {MODEL_NAME}")
    print(f"Server:     {API_BASE_URL}")
    print(f"LLM API:    {LLM_BASE_URL}")
    print()

    results: Dict[str, float] = {}

    for task_id in TASK_IDS:
        print(f"--- {task_id} ---")
        try:
            score = run_episode(task_id)
            results[task_id] = score
            print(f"  Final score: {score:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[task_id] = -1.0

    # Print summary table
    print("\n" + "=" * 50)
    print(f"Baseline Results: {MODEL_NAME}")
    print("=" * 50)
    print(f"{'Task':<25} {'Score':>10}")
    print("-" * 50)
    for task_id in TASK_IDS:
        s = results.get(task_id, -1.0)
        score_str = f"{s:.4f}" if s >= 0 else "ERROR"
        print(f"{task_id:<25} {score_str:>10}")

    valid = [s for s in results.values() if s >= 0]
    if valid:
        mean = sum(valid) / len(valid)
        print("-" * 50)
        print(f"{'Mean':<25} {mean:>10.4f}")
    print()


if __name__ == "__main__":
    main()
