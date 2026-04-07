"""GRPO training script for DataClean-Env.

Trains a language model to clean data using Group Relative Policy
Optimization (GRPO) from the TRL library. Uses `environment_factory`
to create per-rollout environment instances, so reward functions can
read accumulated state (budget, action log, score) directly.

Requirements:
    pip install "openenv-dataclean-env[train]"
    # Or manually:
    pip install trl>=0.16.0 transformers>=5.2.0 torch accelerate datasets

Usage:
    # Start the DataClean-Env server first:
    uvicorn dataclean_env.server.app:app --host 0.0.0.0 --port 8000

    # Then run training:
    accelerate launch scripts/train_grpo.py \
        --model Qwen/Qwen3-0.6B \
        --task easy_contacts \
        --num-episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so reward_functions can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a data-cleaning agent with GRPO on DataClean-Env"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--task", default="easy_contacts",
        choices=["easy_contacts", "medium_employees", "hard_patients"],
    )
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--output-dir", default="./grpo-dataclean-agent")
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt (G in GRPO)")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Imports (fail fast if missing)
    # ---------------------------------------------------------------
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("ERROR: Required packages not installed.")
        print('  pip install "openenv-dataclean-env[train]"')
        print("  # Or: pip install trl>=0.16.0 transformers>=5.2.0 torch accelerate datasets")
        sys.exit(1)

    from reward_functions import (
        reward_valid_action,
        reward_episode_score,
        reward_efficiency,
        reward_no_destruction,
    )

    # ---------------------------------------------------------------
    # Environment Factory
    # ---------------------------------------------------------------
    # TRL's environment_factory creates one instance per rollout.
    # The environment exposes methods as tools that the model can call.
    # After the episode, reward functions read state from the instance.

    # We need to import the environment class. Use mock if openenv
    # isn't installed (the training script doesn't need the server).
    try:
        from dataclean_env.server.environment import DataCleanEnvironment
        from dataclean_env.models import DataCleanAction
    except ImportError:
        print("ERROR: Cannot import DataClean-Env. Install it first:")
        print("  pip install -e .")
        sys.exit(1)

    class DataCleanTrainingEnv:
        """Wrapper environment for GRPO training.

        TRL's environment_factory creates one instance per rollout.
        Public methods become tools the model can call. After the
        episode, reward functions access self.reward and self._state.
        """

        def __init__(self) -> None:
            self._env = DataCleanEnvironment()
            self.reward: float = 0.0
            self._state = None

        def reset(self) -> str:
            """Start a new data cleaning episode.

            Returns:
                A description of the dataset and quality issues to fix.
            """
            obs = self._env.reset(seed=42, task_id=args.task)
            self._state = self._env.state

            # Build a text description for the model
            parts = [f"Task: {obs.task_name} ({obs.difficulty})"]
            parts.append(f"Rows: {obs.data_summary.row_count}, Issues: {obs.data_summary.issue_count}")
            parts.append(f"Budget: {obs.budget_remaining:.0f}")

            if obs.issue_groups:
                parts.append("\nQuality Issues:")
                for g in obs.issue_groups:
                    parts.append(f"  {g.issue_type}: {g.count}")
                    for ex in g.examples[:3]:
                        parts.append(f"    row {ex.row_id} '{ex.column}': {ex.description}")

            if obs.rows:
                parts.append(f"\nData ({len(obs.rows)} rows):")
                parts.append("| " + " | ".join(str(c) for c in obs.columns) + " |")
                for row in obs.rows[:8]:
                    parts.append("| " + " | ".join(str(v) for v in row) + " |")

            return "\n".join(parts)

        def step(self, action_json: str) -> str:
            """Execute a data cleaning action.

            Args:
                action_json: JSON string with action_type and params.
                    Example: {"action_type": "fix_value", "params": {"row_id": 0, "column": "name", "new_value": "Alice"}}

            Returns:
                Result of the action and updated dataset state.
            """
            try:
                data = json.loads(action_json)
                action = DataCleanAction(
                    action_type=data.get("action_type", "mark_complete"),
                    params=data.get("params", {}),
                )
            except (json.JSONDecodeError, Exception):
                action = DataCleanAction(action_type="mark_complete", params={})

            obs = self._env.step(action)
            self._state = self._env.state
            self.reward = float(obs.reward) if obs.reward is not None else 0.0

            # Build response text
            parts = []
            if obs.last_action_result:
                ar = obs.last_action_result
                parts.append(f"Action: {ar.action} → {ar.status}: {ar.message}")

            parts.append(f"Step {obs.step_number}/{obs.max_steps} | "
                         f"Budget: {obs.budget_remaining:.0f} | "
                         f"Issues: {obs.issues_remaining}")

            if obs.done:
                parts.append(f"\nEpisode complete. Final score: {self.reward:.4f}")
            else:
                if obs.quality_issues:
                    parts.append("\nRemaining issues:")
                    for issue in obs.quality_issues[:5]:
                        parts.append(f"  row {issue.row_id} '{issue.column}': {issue.description}")

            return "\n".join(parts)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    # Each prompt instructs the model to clean data for the task.
    # The environment provides context via reset().

    instructions = [
        f"You are a data cleaning agent. Clean the dataset by fixing quality issues. "
        f"Use the available tools to fix values, fill missing data, merge duplicates, "
        f"and standardize formats. When done, call step with mark_complete."
    ] * args.num_episodes

    dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": instruction}]
            for instruction in instructions
        ],
    })

    # ---------------------------------------------------------------
    # GRPO Config
    # ---------------------------------------------------------------

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=min(8, args.num_episodes),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        bf16=True,
    )

    # ---------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------

    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Output: {args.output_dir}")

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[
            reward_valid_action,
            reward_episode_score,
            reward_efficiency,
            reward_no_destruction,
        ],
        environment_factory=DataCleanTrainingEnv,
    )

    print("\nStarting GRPO training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
