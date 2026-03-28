"""Gradio web dashboard for manual testing of the DataClean-Env environment.

Provides interactive controls for task selection, action execution,
dataset inspection, quality issue review, and reward tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from dataclean_env.models import DataCleanAction
from dataclean_env.server.environment import DataCleanEnvironment
from dataclean_env.server.tasks import list_tasks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_TYPES: list[str] = [
    "fix_value",
    "delete_row",
    "fill_missing",
    "standardize_format",
    "merge_duplicates",
    "flag_anomaly",
    "split_column",
    "rename_column",
    "cast_type",
    "mark_complete",
]

TASK_CHOICES: list[str] = [t["task_id"] for t in list_tasks()]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600;700&display=swap');

:root {
    --primary: #2563EB;
    --cta: #F97316;
    --bg: #F8FAFC;
    --text: #1E293B;
}

body, .gradio-container {
    font-family: 'Fira Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.dark body, .dark .gradio-container {
    background: #0F172A !important;
    color: #E2E8F0 !important;
}

code, .mono, .dataframe td, .dataframe th {
    font-family: 'Fira Code', monospace !important;
}

.stat-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}

.dark .stat-card {
    background: #1E293B;
    border-color: #334155;
}

.stat-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
}

.stat-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    font-family: 'Fira Code', monospace;
}

button.primary {
    background: var(--primary) !important;
}

button.secondary, button.stop {
    background: var(--cta) !important;
}

.reward-display {
    font-family: 'Fira Code', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 6px;
    text-align: center;
}
"""

# ---------------------------------------------------------------------------
# Environment wrapper (single shared instance)
# ---------------------------------------------------------------------------

_env = DataCleanEnvironment()
_last_obs: Optional[Any] = None
_action_history: list[dict[str, str]] = []


def _obs_to_dataframe(obs: Any) -> pd.DataFrame:
    """Convert observation rows into a pandas DataFrame."""
    if not obs.rows:
        return pd.DataFrame()
    return pd.DataFrame(obs.rows, columns=obs.columns)


def _issue_table(obs: Any) -> pd.DataFrame:
    """Build a DataFrame of quality issues grouped by type."""
    if not obs.issue_groups:
        return pd.DataFrame(columns=["Type", "Count", "Example"])
    rows = []
    for group in obs.issue_groups:
        example = group.examples[0].description if group.examples else ""
        rows.append({
            "Type": group.issue_type,
            "Count": group.count,
            "Example": example,
        })
    return pd.DataFrame(rows)


def _history_table() -> pd.DataFrame:
    """Return last 10 actions as a DataFrame."""
    if not _action_history:
        return pd.DataFrame(columns=["#", "Action", "Status", "Message"])
    recent = _action_history[-10:]
    return pd.DataFrame(recent)


def _stat_html(label: str, value: Any) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>'
    )


def _format_reward(reward: Any) -> str:
    if reward is None:
        return "---"
    return f"{float(reward):.4f}"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def reset_env(
    task_id: str, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, str, str]:
    """Reset the environment with the selected task and seed."""
    global _last_obs, _action_history
    _action_history = []

    obs = _env.reset(seed=int(seed), task_id=task_id)
    _last_obs = obs

    data_df = _obs_to_dataframe(obs)
    issues_df = _issue_table(obs)
    history_df = _history_table()

    rows_html = _stat_html("Rows", obs.data_summary.row_count)
    nulls_html = _stat_html("Nulls", obs.data_summary.null_count)
    issues_html = _stat_html("Issues", obs.data_summary.issue_count)
    score_html = _stat_html("Score", _format_reward(obs.reward))
    reward_text = f"Reward: {_format_reward(obs.reward)}  |  Step: {obs.step_number}/{obs.max_steps}"

    return data_df, issues_df, history_df, rows_html, nulls_html, issues_html, score_html, reward_text


def execute_action(
    action_type: str,
    row_id: str,
    column: str,
    value: str,
    extra_json: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, str, str]:
    """Execute an action on the environment and return updated state."""
    global _last_obs

    if _last_obs is None:
        raise gr.Error("Reset the environment first.")

    if _last_obs.done:
        raise gr.Error("Episode is done. Reset to start a new one.")

    params: Dict[str, Any] = {}
    if row_id.strip():
        params["row_id"] = int(row_id.strip())
    if column.strip():
        params["column"] = column.strip()
    if value.strip():
        # Map the generic "value" form field to the correct param name
        if action_type == "fix_value":
            params["new_value"] = value.strip()
        else:
            params["value"] = value.strip()

    if extra_json.strip():
        import json
        try:
            extra = json.loads(extra_json.strip())
            if isinstance(extra, dict):
                # Normalize merge_duplicates aliases
                if action_type == "merge_duplicates":
                    if "row_id_1" in extra and "row_id1" not in extra:
                        extra["row_id1"] = extra.pop("row_id_1")
                    if "row_id_2" in extra and "row_id2" not in extra:
                        extra["row_id2"] = extra.pop("row_id_2")
                params.update(extra)
        except json.JSONDecodeError:
            raise gr.Error("Extra params must be valid JSON object.")

    action = DataCleanAction(action_type=action_type, params=params)
    obs = _env.step(action)
    _last_obs = obs

    status = obs.last_action_result.status if obs.last_action_result else "unknown"
    message = obs.last_action_result.message if obs.last_action_result else ""
    _action_history.append({
        "#": str(len(_action_history) + 1),
        "Action": action_type,
        "Status": status,
        "Message": message[:80],
    })

    data_df = _obs_to_dataframe(obs)
    issues_df = _issue_table(obs)
    history_df = _history_table()

    rows_html = _stat_html("Rows", obs.data_summary.row_count)
    nulls_html = _stat_html("Nulls", obs.data_summary.null_count)
    issues_html = _stat_html("Issues", obs.data_summary.issue_count)
    score_html = _stat_html("Score", _format_reward(obs.reward))
    reward_text = f"Reward: {_format_reward(obs.reward)}  |  Step: {obs.step_number}/{obs.max_steps}"

    return data_df, issues_df, history_df, rows_html, nulls_html, issues_html, score_html, reward_text


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        title="DataClean-Env Dashboard",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
            font=["Fira Sans", "sans-serif"],
            font_mono=["Fira Code", "monospace"],
        ),
    ) as app:
        gr.Markdown("## DataClean-Env  /  Manual Testing Dashboard")

        with gr.Row():
            # ---- LEFT PANEL (30%) ----
            with gr.Column(scale=3, min_width=280):
                gr.Markdown("### Task Configuration")
                task_dd = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value=TASK_CHOICES[0] if TASK_CHOICES else "easy_contacts",
                    label="Task",
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("Reset Environment", variant="primary")

                gr.Markdown("### Data Summary")
                with gr.Row():
                    rows_stat = gr.HTML(_stat_html("Rows", "---"))
                    nulls_stat = gr.HTML(_stat_html("Nulls", "---"))
                with gr.Row():
                    issues_stat = gr.HTML(_stat_html("Issues", "---"))
                    score_stat = gr.HTML(_stat_html("Score", "---"))

                gr.Markdown("### Execute Action")
                action_dd = gr.Dropdown(
                    choices=ACTION_TYPES,
                    value=ACTION_TYPES[0],
                    label="Action Type",
                )
                row_id_input = gr.Textbox(label="row_id", placeholder="e.g. 3")
                column_input = gr.Textbox(label="column", placeholder="e.g. email")
                value_input = gr.Textbox(label="value / new_value", placeholder="e.g. john@example.com")
                extra_input = gr.Textbox(
                    label="Extra params (JSON)",
                    placeholder='{"format_type": "date:YYYY-MM-DD"} or {"row_id1": 0, "row_id2": 3, "strategy": "merge_prefer_nonnull"}',
                )
                exec_btn = gr.Button("Execute", variant="secondary")

            # ---- RIGHT PANEL (70%) ----
            with gr.Column(scale=7):
                reward_display = gr.Markdown(
                    value="Reward: ---  |  Step: 0/0",
                    elem_classes=["reward-display"],
                )

                gr.Markdown("### Dataset")
                data_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    max_rows=30,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Quality Issues")
                        issues_table = gr.Dataframe(
                            interactive=False,
                            wrap=True,
                            max_rows=15,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Action History")
                        history_table = gr.Dataframe(
                            interactive=False,
                            wrap=True,
                            max_rows=10,
                        )

        # ---- Wiring ----
        all_outputs = [
            data_table,
            issues_table,
            history_table,
            rows_stat,
            nulls_stat,
            issues_stat,
            score_stat,
            reward_display,
        ]

        reset_btn.click(
            fn=reset_env,
            inputs=[task_dd, seed_input],
            outputs=all_outputs,
        )

        exec_btn.click(
            fn=execute_action,
            inputs=[action_dd, row_id_input, column_input, value_input, extra_input],
            outputs=all_outputs,
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the dashboard."""
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
