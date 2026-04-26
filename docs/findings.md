# MediBill-Env: Training Findings

> Three findings from training Qwen 2.5 3B on this environment. The third
> one, the GRPO reward-saturation result, is the scientifically interesting one.
> All three are reproducible from `notebooks/sft_quickstart.ipynb` and the
> result JSONs in `results/`.

---

## Finding 1 — Base → SFT closes the imitation gap completely on easy and medium, mostly on hard

| Task | Base Qwen 2.5 3B | SFT (LoRA r=32) | Lift | n |
|---|---|---|---|---|
| `easy_cashless` | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | **+1.000** | 5 |
| `medium_multi_payer` | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | **+1.000** | 5 |
| `hard_drift` | 0.0000 ± 0.0000 | 0.7573 ± 0.0040 | **+0.7573** | 5–10 |

**What this tells us about the env:**

1. The base model produces valid JSON tool calls (parse_failures = 0/15 across all base eval episodes) but scores literal zero. This means the env's *format* is learnable from the system prompt alone — but its *policy* is not. The reward signal lives entirely in the policy work, not the calling convention.
2. The lift on `easy` and `medium` is the maximum the composite scoring can express (composite is capped at 1.0). The lift on `hard_drift` is bounded by the scripted teacher (whose own ceiling is 0.7611 — see Finding 2).
3. **Implication for environment authors:** if you want a sharp imitation signal, partition the work between format (free) and policy (graded). Our disjoint identity/policy partition (asserted at module import) is the mechanism that makes this signal sharp.

Reproducibility: `/results/base_eval_n5.json`, `/results/sft_eval_n10.json`, both cross-verified via sha256 byte-match + fresh-subprocess re-eval × 2 ("Codex reproducibility protocol").

---

## Finding 2 — SFT matches the scripted teacher on every tier, statistical tie

| Task | SFT (n=10, 95% CI) | Scripted (n=10, 95% CI) | Δ |
|---|---|---|---|
| `easy_cashless` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 |
| `medium_multi_payer` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 |
| `hard_drift` | 0.7573 ± 0.0040 | 0.7611 ± 0.0049 | **−0.0037 (inside both 95% CIs)** |

**What this tells us:**

1. SFT-from-scripted-traces is a *complete* extraction of the scripted policy at the modelling capacity we used (Qwen 2.5 3B + rank-32 LoRA on 681 training steps). The hard_drift Δ of −0.0037 is statistically indistinguishable from zero given the CI widths.
2. SFT cannot exceed its teacher. To break past 0.7611 on `hard_drift`, we need either (a) a stronger teacher, (b) RL with a reward that escapes the teacher distribution, or (c) richer task tiers that expose new optimal trajectories.

The scripted teacher itself caps at 0.7611 because two grader axes are *gated* in ways the teacher does not satisfy on every seed:
- `drift_bonus` (weight 0.05) is gated on `final_correctness + policy_compliance ≥ 0.80`. Some seeds' drift-step placements push policy_compliance below the gate.
- `abstention_quality` (weight 0.15) requires correct escalations on the two ambiguous cells. The scripted policy never escalates.

These two gates together define a structural ceiling around 0.78 for *any* imitation policy that follows the scripted action distribution. Closing them requires an RL signal that learns to escalate ambiguous cells and re-query insurance after an unobserved drift step — the abstention and drift-bonus axes the spec marks as *RL-only* by design.

---

## Finding 3 — GRPO rewards saturate at step 1 on this env. **This is calibration data.**

We followed SFT with a 5-reward single-step GRPO run targeting the grader's penalty structure:

| Reward fn | Targets | Weight |
|---|---|---|
| `reward_valid_json` | `parse_action()` returns non-`None` | 0.20 |
| `reward_action_in_schema` | `action_type ∈ AGENT_ACTION_TYPES` | 0.20 |
| `reward_no_oscillation` | no `coding_engine` field receives 3+ writes | 0.20 |
| `reward_no_repeated_tool` | no (tool, params) signature repeats 3+ times in a row | 0.20 |
| `reward_submit_with_coding` | every `submit_claim` follows ≥1 `coding_engine` on that claim | 0.20 |

**Observed:**
- `Δ_score` (post-GRPO eval vs pre-GRPO SFT eval): **±0.0002 across all three task tiers**
- `grad_norm` throughout training: **~1e-7**
- Loss curve: flat from step 1 onward

**Diagnosis:**

The five reward functions encode soft properties of "good agent behaviour" that the scripted teacher already exhibits in 100% of its trajectories. The SFT student, having distilled the scripted policy, inherits this property. So at GRPO step 1, every sample in the batch already satisfies all five rewards — there is no advantage signal for the GRPO objective to follow. The policy gradient is null. Training cannot move the policy.

**This is not a bug. It is a calibration result about the environment.**

It says: *the env's tool space at the current task tiers is shallow enough that the scripted-teacher action distribution captures the entire reward signal we know how to express.* Phrased differently — there is no behavioural variance left in the imitation distribution that our reward functions discriminate on.

**Implication for the env design:**

To create RL headroom, the next task tier(s) need optimal trajectories that are *not* in the scripted-teacher distribution. Concretely:

- **Reward-engineered tiers** where the optimal action sequence requires inferring policy version from indirect cues (no `insurance_lookup` available, or rate-limited).
- **Multi-step adversaries** where the policy mutation pattern itself depends on the agent's prior actions.
- **Open-ended escalation rewards** where `escalate_to_human` is a graded judgement, not a binary.

These are env-design decisions, not RL-shaping decisions. The reward functions above are correct — they just have nothing to grip onto in the current tier set.

**Implication for downstream training:**

If a future researcher uses this environment for RL, they should expect saturation against any scripted-teacher SFT initialisation. The path forward is one of:
1. Initialise from a *weaker* SFT (e.g., trained on filtered scripted traces that contain mistakes), so the RL gradient has slack; or
2. Use a reward function that targets the abstention or drift-bonus axes directly (which the scripted teacher doesn't max out — see Finding 2); or
3. Add task tiers that exit the scripted-teacher's distribution.

---

## What this all says

We submit MediBill-Env as a *correctly specified* environment that has reached the imitation ceiling on its current task tiers. The training infrastructure (data generation, SFT pipeline, GRPO config, eval harness) is fully reproducible. The remaining headroom on the abstention and drift-bonus axes is by-design RL-only and is the responsibility of richer task tiers to surface — not of more reward-shaping on the existing tiers.

The GRPO saturation result is itself a contribution: **future practitioners who try to fine-tune this env with single-step reward functions over a scripted-SFT initialisation should expect the same flat gradient and budget compute accordingly.**

---

*Reproducibility: every number in this document is sourced from a result JSON in `/results/` and was verified via sha256 byte-match of the adapter weights + fresh-subprocess re-eval × 2. Eval harness: `medibill/evaluate_sft.py`. Grader constants: `medibill/server/grader.py` lines 29–63 — every weight, gate, penalty exported as a Python module constant.*
