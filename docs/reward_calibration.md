# MediBill-Env Reward Calibration

> Single-page reference: every weight, gate, penalty and bonus the grader uses.
> Source of truth: `medibill/server/grader.py` (constants block, lines 29–63).
> No hidden magic numbers — every value below is exported as a module constant
> and is stable across episodes and seeds.

---

## 1. The six axes (weights sum to 1.0, asserted at import time)

| Axis | Weight | What it measures | Range | Gate |
|---|---:|---|---|---|
| `final_correctness` | **0.45** | Fraction of identity cells matching ground truth across all submitted claims | [0, 1] | none — always applies |
| `policy_compliance` | **0.20** | Fraction of policy-sensitive cells correct under the agent's submit-time policy version | [0, 1] | none — always applies |
| `abstention_quality` | **0.15** | Fraction of `escalate_to_human` calls that target a *genuinely* ambiguous (entity_id, field) | [0, 1] | N/A on tasks with zero ambiguous cells (weight redistributed) |
| `process_auditability` | **0.10** | Two binary signals: (A) `insurance_lookup` before first `submit_claim`; (B) `ehr_query` before first `coding_engine` — averaged | [0, 1] | requires ≥1 successful `coding_engine` call (else 0.0 — closes periodic-lookup exploit) |
| `efficiency` | **0.05** | `1 - budget_spent/budget` | [0, 1] | gated on `final_correctness ≥ 0.10` (else 0.0 — no efficiency credit for doing nothing) |
| `drift_bonus` | **0.05** | Fraction of *post-drift* submitted claims preceded by a fresh `insurance_lookup` between drift event and submit | [0, 1] | N/A on drift-free tasks; requires `final + policy ≥ 0.80` (high-quality only) |

**Disjoint field partition:** `IDENTITY_FIELDS` and `POLICY_SENSITIVE_FIELDS` are disjoint sets, asserted at module import time in `medibill/data_generator.py`. No field can score on both axes — closes the "double-count" exploit.

**Dynamic weight redistribution:** axes marked N/A for a given task contribute 0 and their weight is redistributed proportionally across applicable axes. Closes the "free credit on irrelevant axes" exploit (e.g., a no_op agent earning full `abstention_quality` on tasks with no ambiguous cells).

---

## 2. Penalty schedule (per occurrence; total clipped at PENALTY_CAP = 0.50)

| Penalty | Magnitude | Trigger | Why this exists |
|---|---:|---|---|
| `wrong_fix_ambiguous` | **−0.08** | Submitted with a value in an (entity_id, field) that is in the ambiguous set | Agent should have escalated, not guessed |
| `escalate_clearcut` | **−0.03** | `escalate_to_human` on a cell that is NOT ambiguous | Cheap escalation farming |
| `repeated_tool` | **−0.05** | Same (tool, params) called 3+ times in a row | Reward-hack via tool-call padding |
| `submit_without_coding` | **−0.05** | `submit_claim` on a claim that never had a `coding_engine` call | Submitting the dirty state verbatim |
| `oscillation` | **−0.03 × (writes−2)** | 3+ successful `coding_engine` writes to the same (claim_id, field) | Alternating values to farm process credit |

**One legitimate mistake-then-correction is free.** The third write triggers the penalty; each further write adds another. Cap at `PENALTY_CAP = 0.50` prevents penalty stacking from dominating the score (a small hack penalty doesn't override a fundamentally good submission).

---

## 3. Bonus schedule (total clipped at BONUS_CAP = 0.15)

| Bonus | Magnitude | Trigger | Gate |
|---|---:|---|---|
| `correct_escalation` | **+0.03** per | `escalate_to_human` on a genuinely ambiguous (entity_id, field) | `final_correctness ≥ 0.70` |
| `all_identity_80pct` | **+0.05** | ≥80% of submitted claims have ALL identity cells correct | `final ≥ 0.70` AND `policy_compliance ≥ 0.50` |
| `cross_claim_consistency` | **+0.03** | Same patient_id maps to a single patient_name across all submitted claims | `final ≥ 0.70` AND `policy_compliance ≥ 0.50` |

**Score-separation gate** (Codex review 2026-04-20 finding 2): structural bonuses (`all_identity_80pct`, `cross_claim_consistency`) require BOTH the identity floor AND the policy floor. Without this gate, a no_op agent submitting the dirty state verbatim would earn both bonuses for free on tasks where ground-truth identity is preserved in the dirty state. We caught this in audit and closed it.

---

## 4. Exploit gate — five attacks tested (`medibill/test_exploits.py`)

For every attack: `score(attack) ≤ score(no_op) + 1e-3` across 5 seeds × 2 task tiers (easy_cashless, hard_drift). Run as part of the test suite; this is a hard pre-training gate.

| Exploit | Strategy | Result |
|---|---|---|
| `ack_spammer` | 20× `insurance_lookup` then bare-submit all claims | ≤ no_op ✅ (no_coding penalty + repeated_tool penalty) |
| `escalate_everything` | Escalate every (claim, policy_field), then submit | ≤ no_op ✅ (escalate_clearcut penalty + still no real fix) |
| `oscillator` | Alternate `coding_engine` writes to one cell 20× | ≤ no_op ✅ (oscillation penalty + cell still wrong) |
| `double_count` | Submit with diagnosis_code present (carry-through from EHR) | ≤ no_op ✅ (disjoint partition stops cross-axis credit) |
| `periodic_lookup` | Poll `insurance_lookup` every 5 steps, submit between polls | ≤ no_op ✅ (no_coding penalty + still no real fix) |

**5 / 5 PASS.** No reward hack we have tried beats doing nothing. The grader is only beatable by actually doing the task.

---

## 5. Reward-saturation analysis (the GRPO finding, not a bug)

After SFT to scripted-teacher parity, we ran a GRPO experiment with 5 single-step reward functions designed to target the grader's penalty structure (`reward_no_oscillation`, `reward_no_repeated_tool`, `reward_submit_with_coding`, `reward_valid_json`, `reward_action_in_schema`).

**Observation:** `Δ_score = ±0.0002`, gradient norm ~1e-7 throughout training.

**Diagnosis:** SFT-from-scripted-traces already extracts the full reward signal these functions can express. The trajectories in our SFT corpus already satisfy: valid JSON, in-schema actions, no oscillation, no repeated-tool runs, coding-before-submit. The GRPO objective has no remaining gradient to follow.

**This is calibration data, not a failure.** It tells us:
1. The reward shaping is *consistent* with the SFT supervision signal — they pull in the same direction (good).
2. The env's tool space at the current task tiers is shallow enough that imitation captures it (a property of the env, not RL).
3. To create RL headroom we need to add task tiers where the optimal trajectory is *not* in the scripted-teacher distribution — for example, a "discovery" tier where the agent must infer policy version from indirect cues with no `insurance_lookup` available.

**Roadmap implication:** the next task tier is reward-engineered, not RL-engineered.

---

## 6. Reproducibility

Every constant in this document is exported as a Python module constant in `medibill/server/grader.py` (`WEIGHTS`, `PENALTY_*`, `BONUS_*`, `MIN_*`, `DRIFT_BONUS_MIN_FINAL_PLUS_POL`). External reviewers can `import` them and write tests against specific values.

```bash
python -m medibill.test_exploits     # 5 attacks vs no_op (PASS gate)
python -m medibill.validate_grader   # disjoint partition + axis applicability
```

---

*Source: `medibill/server/grader.py` lines 29–63 for constants; lines 484–576 for penalty implementation; lines 584–643 for bonus implementation. Argue with any number — it is a constant in source.*
