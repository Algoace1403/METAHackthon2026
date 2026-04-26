# MediBill-Env: Teaching an LLM Agent That Its World Just Changed

*A Meta × Scaler OpenEnv Hackathon Round 2 submission. Theme 3.1 — Professional World Modeling. Environment + grader + 5-attack exploit gate + SFT v1 → GRPO calibration → SFT v2 (0.9996 on hard_drift) all shipped.*

---

India's insurance regulator gives hospitals one hour to authorize a cashless claim and three hours to discharge it. Miss the three-hour clock and the overrun comes out of the insurer's shareholder fund — not the hospital's, not the patient's. So a human medical coder sits at a terminal, racing 180 minutes against a discharge summary, an ICD-10 code, and a policy document that may have been rewritten since their last shift.

That last part is the trap. Insurance policies in India drift constantly: codes get renamed, pre-auth thresholds move, signature requirements appear without an announcement. A coder who memorised yesterday's rules submits under yesterday's rules and watches the claim bounce. In FY24, ₹26,000 crore of health claims were disallowed — up 19% year-on-year. The mistake isn't ignorance. It's *staleness*.

**MediBill-Env** is an OpenEnv environment that puts an LLM agent into exactly that seat. Five tools, three task tiers, a six-axis deterministic grader, and one mechanic that makes everything else interesting: on hard tasks, the active policy mutates mid-episode without a flag, an event, or a hint. The only way the agent ever learns the rules changed is to call `insurance_lookup` again. And submissions are graded against the policy at submit time — not against whatever the agent thinks is true.

This post is about what we built, what the baselines say, what GRPO told us about reward saturation, and how a smarter teacher took a Qwen 2.5 3B model from **0.0000 → 0.9996** on the hardest task tier.

> **Quick links:** Repo · [github.com/Algoace1403/METAHackthon2026](https://github.com/Algoace1403/METAHackthon2026) · HF Space (LIVE) · [huggingface.co/spaces/Anuj424614/medibill-env](https://huggingface.co/spaces/Anuj424614/medibill-env) · Colab · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Algoace1403/METAHackthon2026/blob/main/notebooks/sft_quickstart.ipynb) · Spec · [`docs/round2-spec-v3.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/round2-spec-v3.md)

---

## 1. The regulatory clock

IRDAI's Master Circular on Health Insurance, issued 29 May 2024 and effective 31 July 2024, mandates cashless-claim turnaround of **1 hour for pre-authorisation and 3 hours for final discharge**. Miss the discharge clock and the cost overrun is borne by the insurer from shareholder funds — a structural penalty, not a notional one.

Two facts about how this is going:

- **FY24:** ₹26,000 crore in health claims disallowed (+19% YoY) — IRDAI Annual Report.
- **FY25:** about 13% of pre-auths still miss the one-hour window, per LocalCircles' January 2025 survey of policyholders.

Star Health, ICICI Lombard, HDFC ERGO, Bajaj Allianz — every cashless desk runs this clock. With 50 crore Indians now covered under Ayushman Bharat / PM-JAY, this is the largest cashless-claim system in the world racing the strictest clock in the world. The bottleneck is a human coder racing that clock — and the policies keep changing on them.

## 2. Why "policy drift" is the real failure mode

Most agent benchmarks check whether the model can fill a form correctly. That's schema validation, and rules engines have done it for thirty years. The interesting failure mode in this domain is *staleness* — the policy changed, the agent didn't notice, the claim is wrong.

In real workflows, insurer policies get updated:

- Codes get renamed (`HCPCS G0463` → `Q3014`).
- Pre-authorisation thresholds shift.
- New signature requirements appear.
- Coverage of specific diagnosis ranges expands or contracts.

An LLM agent that learns to do the job by imitating one month of expert trajectories will reproduce one month's rules. Drop it into the next month and it fails quietly. We wanted an environment that tests whether the agent **knows to re-check** before submitting — and grades it accordingly.

## 3. The environment

MediBill-Env is an OpenEnv environment with five agent tools and a six-axis deterministic grader.

| Tool | Purpose |
|---|---|
| `ehr_query` | Read a patient record |
| `insurance_lookup` | Fetch the insurer's currently-active policy rules |
| `coding_engine` | Write a policy-sensitive field (diagnosis code, pre-auth flag, narrative…) |
| `escalate_to_human` | Calibrated abstention |
| `submit_claim` | Lock a claim for grading |

Three task tiers:

| Task | Claims | Provider | Drift |
|---|---|---|---|
| `easy_cashless` | 6 | CGHS v2024.1 | none |
| `medium_multi_payer` | 10 | Star v1.4 | none |
| `hard_drift` | 12 | Star v1.3 → v1.4 | silent, at a seed-selected step in `range(10, 40)` |

All data is synthetic. Diagnoses use **ICD-10-CM** (CMS public domain). Procedures use **SYNTH-PROC-v1**, a project-owned ontology with no mapping to AMA CPT. Pricing references the **CGHS package-rate list** (MoHFW). No proprietary medical coding content is ever loaded.

The grader has six axes and an import-time invariant that identity fields and policy-sensitive fields are formally disjoint:

```python
assert not (set(IDENTITY_FIELDS) & set(POLICY_SENSITIVE_FIELDS))
```

| Axis | Weight | What it measures |
|---|---|---|
| `final_correctness` | 45% | Identity fields match ground truth |
| `policy_compliance` | 20% | Policy-sensitive fields correct under the policy at submit time |
| `abstention_quality` | 15% | Calibrated escalation |
| `process_auditability` | 10% | Tool-call order patterns |
| `efficiency` | 5% | Budget used vs allocation, gated on correctness |
| `drift_bonus` | 5% | Gated on post-drift re-query + final correctness |

## 4. The hero mechanic — silent policy drift

On hard tasks, the active policy mutates partway through the episode. There is no announcement.

```python
# medibill/server/environment.py — drift firing (simplified)
def _maybe_fire_drift(self) -> None:
    if self._state.step_count >= self._pending_drift.step:
        new_policy = get_provider(self.provider).at(self._pending_drift.to_version)
        self._state.active_policy = new_policy
        # No observation field announces this. No flag. No metadata key.
        # The only way the agent will learn: call insurance_lookup again.
```

The grader closes the loop: `submit_claim` is graded against the policy active **at submit time**, and the `drift_bonus` axis only credits the agent if a fresh `insurance_lookup` call happens between the last drift event and the next submit.

```python
# medibill/server/grader.py — drift_bonus gating (simplified)
def _axis_drift_bonus(tool_log, drift_events, submitted_ids, per_claim) -> float:
    # 1.0 only if a fresh insurance_lookup call exists between the last
    # drift event and submit_claim, AND final_correctness is adequate.
    # Otherwise 0.0. Prevents polling-based memorisation of drift timing.
```

That second snippet is the one that earns the post its right to exist. It is easy to *say* an environment tests reasoning under shifting state. It is harder to write a grader that refuses to be gamed by the obvious shortcut — calling `insurance_lookup` every step, regardless of whether anything changed. The grader's drift axis only pays for a lookup that follows a drift it didn't know was coming. That's the behaviour we want a learned policy to acquire.

## 5. Baselines and the drift acceptance gap

Before any training, we measure three independent baselines on every task. The gap between the strongest baseline (tool-faithful scripted) on the no-drift task and on the drift task is the **drift acceptance gap** — it's the entire reason this environment is hard, and the behavioural target the training pipeline is designed to close.

![Three baselines (random / no_op / scripted) across the three task tiers, n=20 seeds each, with 0.25 drift acceptance gap on hard_drift annotated](https://raw.githubusercontent.com/Algoace1403/METAHackthon2026/main/docs/img/baselines.png)

| Task (n=20) | random | no_op | scripted |
|---|---:|---:|---:|
| `easy_cashless` | 0.36 | 0.39 | **1.000** |
| `medium_multi_payer` | 0.21 | 0.15 | **1.000** |
| `hard_drift` | 0.108 | 0.079 | **0.754** |

*That 0.246 drop on `hard_drift` is the cost of carrying a stale policy model into submit. It is not a recovery story — yet.*

Seed 44 is the run that taught me what this environment was actually testing. The scripted baseline opens normally — `ehr_query`, `insurance_lookup`, `coding_engine`, submit. Three claims in, it's holding 1.000. At step 23 the policy silently bumps from `v1.3` to `v1.4`. The trace shows nothing. No flag, no log line, no observation field. The script keeps running its loop. Nine more claims, all submitted under a policy that no longer exists. Final composite: 0.753. The agent didn't make a coding mistake — `coding_engine` is identical across tasks. It made a *trust* mistake. It trusted a policy snapshot it had cached fifteen steps ago. Watching that score crystallise on the terminal was the moment the spec stopped being a spec and started being the point.

**Reproducibility.** Across 20 seeds on `hard_drift`, scripted lands in a tight band of **0.748–0.765, mean 0.754, sd 0.006**. The drift step varies seed-to-seed across the full `range(10, 40)` candidate set; the score is stable because the loss mechanism (post-drift submission under stale policy) is deterministic. A separation gate on every commit asserts `scripted - no_op ≥ 0.5` on `easy_cashless` and `≥ 0.4` on `hard_drift`. Current margins: +0.61 and +0.68. Per-seed data: [`docs/baseline_reproducibility.csv`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/baseline_reproducibility.csv).

### 5.1 Reward is hard to game — the exploit gate

A composable rubric is only as strong as its resistance to gaming. We wrote five attack policies — each one specifically targeting a class of grader exploit — and ship them as a continuous-integration gate.

![Exploit gate: five attack policies all clamped at or below the no_op floor on hard_drift, n=20 seeds each](https://raw.githubusercontent.com/Algoace1403/METAHackthon2026/main/docs/img/exploits.png)

| Attack | Hypothesis it tests | Mean (n=20) | vs no_op |
|---|---|---:|:--:|
| `ack_spammer` | grader pays for ceremony / log volume | 0.041 | **<** |
| `escalate_everything` | calibrated abstention can be faked by always escalating | 0.071 | **<** |
| `oscillator` | grader rewards "looking busy" via repeat writes | 0.000 | **<** |
| `double_count` | submit twice to inflate process_auditability | 0.079 | **=** |
| `periodic_lookup` | spam `insurance_lookup` to fish for `drift_bonus` | 0.069 | **<** |

Every attack policy is at or below the no_op floor (0.079) on `hard_drift`. The CI gate fails the build if any attack scores `> no_op + 1e-3`. Source: [`medibill/test_exploits.py`](https://github.com/Algoace1403/METAHackthon2026/blob/main/medibill/test_exploits.py).

The `periodic_lookup` exploit is the one that nearly broke the environment. Early in development, an agent that called `insurance_lookup` on every step earned `drift_bonus` cheaply — it always had a fresh lookup before any submit. Fixing it required tightening the bonus gate so it only credits a lookup that *follows* a drift event the agent couldn't have predicted. The gate now passes scripted (0.754) and clamps the polling exploit at 0.069 — well below no_op.

## 6. Training pipeline — three checkpoints to 0.9996

We trained Qwen 2.5 3B Instruct against this environment with Unsloth + HF TRL on Colab. Here is what happened, in order. The headline is at the bottom.

![Training progression on hard_drift: Base 0.0000 → SFT v1 0.7573 → GRPO saturated → SFT v2 0.9996](https://raw.githubusercontent.com/Algoace1403/METAHackthon2026/main/docs/img/base_vs_sft.png)

| Checkpoint | hard_drift score | Source | What changed |
|---|---:|---|---|
| Base Qwen 2.5 3B | **0.0000** | untrained | parse_failures = 0/15. The model produces valid JSON tool calls. It just has no policy reasoning. |
| SFT v1 | **0.7573** | scripted teacher (`ScriptedHeuristicPolicy`) | Imitates the scripted baseline; matches its structural ceiling within statistical noise. |
| GRPO over SFT v1 | **0.7575** (Δ ±0.0002) | 5-reward single-step GRPO | Rewards saturated by SFT — calibration finding. |
| **SFT v2** | **0.9996** | drift-aware teacher (`ScriptedDriftAwarePolicy`) | Teacher escalates ambiguous cells + does fresh `insurance_lookup` before every submit. |

### 6.1 Base → SFT v1 — supervised fine-tuning to scripted parity

We generated 144 trajectories total (16 seeds × 3 tasks × {scripted, random, no_op}), filtered to 48 scripted-heuristic trajectories for SFT. Eval: 12 trajectories from seeds 16–19, never seen in training.

Training format: `{"messages": [system, user, assistant]}` with assistant content as canonical JSON (`sort_keys=True`). Loss masked to assistant tokens via Unsloth's `train_on_responses_only`. LoRA `r=32, alpha=64` over `q/k/v/o/gate/up/down_proj`. Capability-conditional precision (bf16 on Ampere+, fp16 on Turing). Every trajectory carries a SHA-derived `prompt_version`; the loader refuses skewed records.

Result on n=10 held-out seeds: easy_cashless **1.0000 ± 0.0000**, medium_multi_payer **1.0000 ± 0.0000**, hard_drift **0.7573 ± 0.0040**. The hard_drift Δ vs scripted (0.7611 ± 0.0049) is **−0.0037**, statistically indistinguishable from zero given both 95% CIs. SFT cannot exceed its teacher.

### 6.2 GRPO — and the calibration finding

We followed SFT v1 with a 5-reward single-step GRPO run targeting the grader's penalty structure:

| Reward fn | Targets | Weight |
|---|---|---:|
| `reward_valid_json` | `parse_action()` returns non-`None` | 0.20 |
| `reward_action_in_schema` | `action_type ∈ AGENT_ACTION_TYPES` | 0.20 |
| `reward_no_oscillation` | no `coding_engine` field receives 3+ writes | 0.20 |
| `reward_no_repeated_tool` | no (tool, params) signature repeats 3+ times in a row | 0.20 |
| `reward_submit_with_coding` | every `submit_claim` follows ≥1 `coding_engine` on that claim | 0.20 |

**Observed:** Δ_score = ±0.0002 across all three task tiers. Gradient norm ~1e-7 throughout. Loss curve flat from step 1.

**Diagnosis.** The five reward functions encode soft properties of "good agent behaviour" that the scripted teacher already exhibits in 100% of its trajectories. The SFT v1 student, having distilled the scripted policy, inherits this property. So at GRPO step 1, every sample in the batch already satisfies all five rewards — there is no advantage signal for the GRPO objective to follow. The policy gradient is null. Training cannot move the policy.

**This is not a training bug. It is calibration data about the environment.** It says: *the env's tool space at the current task tiers is shallow enough that the scripted-teacher action distribution captures the entire reward signal we know how to express.* Phrased differently — there is no behavioural variance left in the imitation distribution that our reward functions discriminate on.

Future RL practitioners who try to fine-tune this env with single-step reward functions over a scripted-SFT initialisation should expect the same flat gradient and budget compute accordingly. That's a contribution, not an embarrassment.

### 6.3 SFT v2 — teacher engineering, not reward shaping

The diagnosis above pointed at the fix. Instead of writing more reward functions over the same saturated distribution, we built a stronger teacher and re-distilled.

`ScriptedDriftAwarePolicy` adds three behaviours over the baseline scripted policy:

1. **Escalate ambiguous cells.** The grader rewards calibrated abstention on the two cells where the rubric encodes uncertainty.
2. **Fresh `insurance_lookup` before every `submit_claim`.** If rules just changed, this is the only way to find out — and it's exactly what the `drift_bonus` axis pays for.
3. **Drift detection via rule comparison.** If the lookup result differs from the cached view, re-code unsubmitted claims under the new policy.

Local n=5 verification: scripted++ scores **0.9983** on hard_drift versus the baseline scripted's 0.7568. We then generated 90 new trajectories from this teacher and re-distilled into a fresh LoRA adapter on top of the same Qwen 2.5 3B base.

**Cost:** 90 trajectories × 3 epochs = 1482 training steps, loss 0.42 → 0.011, **33.5 minutes** on a Colab L4 GPU. LoRA r=32. Per-seed hard_drift scores on n=5 held-out: 1.0000, 1.0000, 1.0000, 1.0000, 0.9979. Zero parse failures across 15 episodes.

| Task | Base Qwen 2.5 3B | SFT v2 (LoRA r=32) | Lift |
|---|---:|---:|---:|
| `easy_cashless` | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | **+1.000** |
| `medium_multi_payer` | 0.0000 ± 0.0000 | **1.0000 ± 0.0000** | **+1.000** |
| `hard_drift` | 0.0000 ± 0.0000 | **0.9996 ± 0.0008** | **+0.9996** |
| **average** | **0.0000** | **0.9999** | **+0.9999** |

Verified via Codex's reproducibility protocol: sha256 byte-match of adapter weights + fresh-subprocess re-eval × 2. Eval JSONs in `/results/` ([`base_eval_n5.json`](https://github.com/Algoace1403/METAHackthon2026/blob/main/results/base_eval_n5.json), [`sft_eval_n10.json`](https://github.com/Algoace1403/METAHackthon2026/blob/main/results/sft_eval_n10.json), `sft_v2_eval_n5.json`).

![Per-task lift: average +0.9999 from base Qwen to SFT v2 across all 3 tiers](https://raw.githubusercontent.com/Algoace1403/METAHackthon2026/main/docs/img/improvement_per_task.png)

### 6.4 The lesson

The actionable takeaway from this work is one sentence. *When RL gets stuck on a saturated reward surface, the cheap and effective move is to engineer a stronger teacher and re-distill, not to write more reward functions.* +0.2423 lift on the hardest tier in 33 minutes of retraining beat anything reward shaping could have produced over the same compute. Future teams using this environment should know that.

## 7. Limitations

- The scripted teacher is still a hand-coded heuristic. SFT v2 inherits its policy; it does not exceed it. A genuine RL improvement (one that finds behaviours the teacher does not) would need either a weaker SFT init (with mistakes RL can correct), or task tiers that fall outside the teacher's distribution.
- Held-out eval for SFT v2 is n=5 seeds. The hard_drift score (0.9996) has tight CI (±0.0008), but scaling to 30+ seeds is a post-hackathon priority.
- We trained on a single base model (Qwen 2.5 3B Instruct). Whether the GRPO saturation result generalises to larger bases or different families is open.

## 8. Try it

```bash
git clone https://github.com/Algoace1403/METAHackthon2026
cd METAHackthon2026
pip install -e .
python -m medibill.demo_runner --seed 44      # one narrated episode, ~30s
python -m medibill.test_exploits               # 5-exploit gate, ~5s
python -m medibill.validate_grader --task all  # 3-baseline separation gate
```

Or [open the Colab notebook](https://colab.research.google.com/github/Algoace1403/METAHackthon2026/blob/main/notebooks/sft_quickstart.ipynb) and re-run the full SFT pipeline against the live HF Space.

Three commands, three artefacts: a narrated episode that lands at 0.753, a gate that proves no attack policy beats no_op, and a baseline sweep that proves the 0.25 drift acceptance gap is real. If you can reproduce all three on your laptop, you have everything you need to take a shot at closing the gap further with RL — using either richer task tiers or a different SFT init that hasn't already saturated the reward.

## 9. Closing

The interesting question in agent evaluation isn't "can the model fill the form" — rules engines have done that for thirty years. It's "does the model know when its world has changed underneath it." MediBill-Env doesn't answer that question; it asks it cleanly enough that an answer becomes measurable.

The 0.25 drift acceptance gap on `hard_drift` was the target. SFT v1 closed half of it (to 0.7573). GRPO showed us why we couldn't close more by reward shaping alone. SFT v2 closed essentially all of it (0.9996) by engineering a smarter teacher. That sequence — base → imitation → reward saturation → teacher upgrade — is, in our view, the most useful thing this submission contributes back to the OpenEnv community: a worked example of *what to try when GRPO gives you a flat gradient.*

The repo is up. The Space is live. The grader is one `pip install` away. The notebook reproduces the full result on free-tier Colab. Break it, fork it, push past 0.9996.

---

**Links**

- **Repository:** [github.com/Algoace1403/METAHackthon2026](https://github.com/Algoace1403/METAHackthon2026)
- **HF Space (LIVE):** [huggingface.co/spaces/Anuj424614/medibill-env](https://huggingface.co/spaces/Anuj424614/medibill-env)
- **Colab notebook:** [Open in Colab](https://colab.research.google.com/github/Algoace1403/METAHackthon2026/blob/main/notebooks/sft_quickstart.ipynb)
- **Specification:** [`docs/round2-spec-v3.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/round2-spec-v3.md)
- **Findings:** [`docs/findings.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/findings.md)
- **Reward calibration:** [`docs/reward_calibration.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/reward_calibration.md)

*Reviews by Codex and ChatGPT caught 30+ substantive issues across spec v1 → v3 and three rounds of training calibration. The final design is stronger because of those reviews.*
