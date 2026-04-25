# MediBill-Env: Teaching an LLM Agent That Its World Just Changed

*A Meta × Scaler OpenEnv Hackathon Round 2 submission. Theme 3.1 — Professional World Modeling. Environment-first: SFT and RL passes ship as runnable pipeline, no trained-model claims today.*

---

India's insurance regulator gives hospitals one hour to authorize a cashless claim and three hours to discharge it. Miss the three-hour clock and the overrun comes out of the insurer's shareholder fund — not the hospital's, not the patient's. So a human medical coder sits at a terminal, racing 180 minutes against a discharge summary, an ICD-10 code, and a policy document that may have been rewritten since their last shift.

That last part is the trap. Insurance policies in India drift constantly: codes get renamed, pre-auth thresholds move, signature requirements appear without an announcement. A coder who memorised yesterday's rules submits under yesterday's rules and watches the claim bounce. In FY24, ₹26,000 crore of health claims were disallowed — up 19% year-on-year. The mistake isn't ignorance. It's *staleness*.

**MediBill-Env** is an OpenEnv environment that puts an LLM agent into exactly that seat. Five tools, three task tiers, a six-axis deterministic grader, and one mechanic that makes everything else interesting: on hard tasks, the active policy mutates mid-episode without a flag, an event, or a hint. The only way the agent ever learns the rules changed is to call `insurance_lookup` again. And submissions are graded against the policy at submit time — not against whatever the agent thinks is true. This post is about what we built, what the baselines say, and the 0.25 gap that is the whole reason this environment is worth training against.

> **Quick links:** Repo · [github.com/Algoace1403/METAHackthon2026](https://github.com/Algoace1403/METAHackthon2026) · HF Space · [huggingface.co/spaces/Anuj424614/medibill-env](https://huggingface.co/spaces/Anuj424614/medibill-env) · Spec · [`docs/round2-spec-v3.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/round2-spec-v3.md)

---

## 1. The regulatory clock

IRDAI's Master Circular on Health Insurance, issued 29 May 2024 and effective 31 July 2024, mandates cashless-claim turnaround of **1 hour for pre-authorisation and 3 hours for final discharge**. Miss the discharge clock and the cost overrun is borne by the insurer from shareholder funds — a structural penalty, not a notional one.

Two facts about how this is going:

- **FY24:** ₹26,000 crore in health claims disallowed (+19% YoY) — IRDAI Annual Report.
- **FY25:** about 13% of pre-auths still miss the one-hour window, per LocalCircles' January 2025 survey of policyholders.

The bottleneck is a human coder racing a clock — and the policies keep changing on them.

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
| `abstention_quality` | 15% | Calibrated escalation (RL-only target — see §6.1) |
| `process_auditability` | 10% | Tool-call order patterns |
| `efficiency` | 5% | Budget used vs allocation, gated on correctness |
| `drift_bonus` | 5% | Gated on post-drift re-query + final correctness (RL-only target) |

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

*That 0.246 drop on `hard_drift` is the cost of carrying a stale policy model into submit. It is not a recovery story.*

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

## 6. Training pipeline (shipped, not measured today)

The full **Qwen2.5-3B-Instruct + LoRA SFT** pipeline ships in this repo and is documented to run on free-tier Colab T4. We did not execute it inside the hackathon's compute window, so we do not show SFT bars on the chart above. The pipeline is included so judges can reproduce the next step end-to-end.

- **Trajectories.** 144 total (16 seeds × 3 tasks × {scripted, random, no_op}), filtered to 48 scripted-heuristic trajectories for SFT. Random and no-op trajectories exist as a contrast pool, held out of training.
- **Eval.** 12 trajectories (seeds 16–19, scripted), never seen in training.
- **Chat format.** `{"messages": [system, user, assistant]}`. Assistant content is canonical JSON (`sort_keys=True`). Loss is masked to assistant tokens only via Unsloth's `train_on_responses_only`, which token-ID-matches the ChatML role markers `<|im_start|>user\n` and `<|im_start|>assistant\n` to set `labels=-100` on every non-assistant token.
- **Precision.** Capability-conditional: bf16 on Ampere+, fp16 on Turing.
- **Prompt-version guard.** Every trajectory carries a SHA-derived `prompt_version`. The training loader refuses records whose version doesn't match the installed one. Skews silently kill training; we'd rather fail loudly.
- **LoRA config.** `r=32, alpha=64`, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

### 6.1 What SFT is expected to improve, and what it is not

Four of the six rubric axes are SFT-reachable from scripted trajectories: `final_correctness`, `policy_compliance`, `process_auditability`, and `efficiency`. All four are demonstrable by imitation.

Two axes are **RL-only** targets in our pipeline, scoped explicitly in spec v3 §7.6:

- `abstention_quality` — the ambiguous-cell ground truth is not in the agent's observation. An abstention-aware scripted policy would need to read hidden state.
- `drift_bonus` — the scripted policy detects drift by schedule, not by policy staleness. Meaningful drift calibration requires reward signal.

We ship the SFT half and *did not falsify the RL half*. A "trained model" bar that's secretly the scripted baseline is the single most-flagged failure mode in this hackathon's judging rubric. We have a working SFT script and a separation gate; we did not run SFT inside our compute window, so we don't claim a number.

## 7. Limitations

- We did not run SFT inside the hackathon compute window. The pipeline ships and is reproducible, but the numbers in this post are baselines, not training results.
- Scripted scores 1.000 on `easy_cashless` and `medium_multi_payer`, so SFT can only tie or lose on those tasks. The meaningful SFT delta is on `hard_drift` specifically.
- Held-out eval is 12 trajectories. Means are indicative; 95% CIs are wide. Scaling eval to 60+ trajectories is a post-hackathon priority.

## 8. Try it

```bash
git clone https://github.com/Algoace1403/METAHackthon2026
cd METAHackthon2026
pip install -e .
python -m medibill.demo_runner --seed 44      # one narrated episode, ~30s
python -m medibill.test_exploits               # 5-exploit gate, ~5s
python -m medibill.validate_grader --task all  # 3-baseline separation gate
```

Three commands, three artefacts: a narrated episode that lands at 0.753, a gate that proves no attack policy beats no_op, and a baseline sweep that proves the 0.25 drift acceptance gap is real. If you can reproduce all three on your laptop, you have everything you need to take a shot at closing the gap.

## 9. Closing

The interesting question in agent evaluation isn't "can the model fill the form" — rules engines have done that for thirty years. It's "does the model know when its world has changed underneath it." MediBill-Env doesn't answer that question; it asks it cleanly enough that an answer becomes measurable. The 0.25 drift acceptance gap on `hard_drift` is the target. The exploit gate keeps the target honest. The training pipeline ships so the next person — maybe you — can take a shot at closing it.

The repo is up. The Space is live. The grader is one `pip install` away. Break it, fork it, beat 0.754.

---

**Links**

- **Repository:** [github.com/Algoace1403/METAHackthon2026](https://github.com/Algoace1403/METAHackthon2026)
- **HF Space:** [huggingface.co/spaces/Anuj424614/medibill-env](https://huggingface.co/spaces/Anuj424614/medibill-env)
- **Specification:** [`docs/round2-spec-v3.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/round2-spec-v3.md)
- **Colab runbook:** [`docs/colab_recipe.md`](https://github.com/Algoace1403/METAHackthon2026/blob/main/docs/colab_recipe.md)

*Reviews by Codex and ChatGPT caught 22 substantive issues between spec v1 and v3. The final design is stronger because of those reviews.*
