# MediBill-Env v3 — Status Update (2026-04-20)

Three-part document for collaborator review, judge-safe summary, and open-risk tracking.

---

## 1. Technical status update (collaborators)

### Verified now (each claim below has a test that exits 0 today)

**Environment surface.** Five agent-visible actions: `ehr_query`, `insurance_lookup`, `coding_engine`, `escalate_to_human`, `submit_claim`. Round 1 primitives (`fix_value`, `fill_missing`, `merge_duplicates`, …) are not reachable from the agent.

**Silent policy drift.** On `hard_drift`, the environment replaces the active policy at a step drawn uniformly from `range(10, 40)` (30 candidates) using the episode seed. No observation field or flag announces the mutation. In the current design, the only supported way for an agent to observe the new policy is a fresh `insurance_lookup` call after the drift step; submissions are graded against the policy active at submit time. The environment enforces observability, not reasoning quality — a polling strategy can surface the new payload without having learned why or when to look.

**Grader partitioning.** Two disjoint field sets:

- `IDENTITY_FIELDS` (11 fields) scored by `final_correctness`.
- `POLICY_SENSITIVE_FIELDS` (9 fields), of which only the **fillin subset** of 6 is scored by `policy_compliance`:
  - `policy_version`
  - `pre_auth_flag`
  - `pre_auth_number`
  - `required_signatures`
  - `discharge_summary_attached`
  - `diagnosis_narrative`

The remaining 3 (`provider`, `diagnosis_code`, `procedure_code`) are carry-through from the dirty state (pre-populated from the EHR) and are not scored. Disjointness of identity vs policy-sensitive sets is enforced by an import-time `assert` in `data_generator.py`:

```
assert not (set(IDENTITY_FIELDS) & set(POLICY_SENSITIVE_FIELDS))
assert set(FILLIN_POLICY_FIELDS).issubset(set(POLICY_SENSITIVE_FIELDS))
```

**Structurally-N/A axes.** Two of six axes can be inapplicable for a given task:
- `abstention_quality` — N/A when the task declares an empty ambiguous-cell set.
- `drift_bonus` — N/A when the task has no drift events.

When an axis is N/A it contributes zero. Nominal weights (`0.45, 0.20, 0.15, 0.10, 0.05, 0.05`) are then redistributed proportionally over the applicable axes. Let `A` be the set of applicable axes and `W` the nominal weight map. The composite is

```
score = Σ_{a ∈ A} (W[a] / Σ_{b ∈ A} W[b]) · effective[a]
       − penalties + bonuses
```

clipped to `[0, 1]`. This avoids the prior behaviour where an inapplicable axis auto-awarded 1.0 × its nominal weight (inflating no-op on easy/medium). The redistribution rule is structural, not tuned to any target score.

**Tool-faithful baselines.** `random_agent`, `no_op_agent`, and `ScriptedHeuristicPolicy` in `baselines.py`:
- do not read `env.state`
- do not import `medibill.data.insurers` at runtime
- read provider from `obs.claims[i].provider`
- read policy rules from `obs.last_tool_result.payload["rules"]` after an `insurance_lookup` call
- make all remaining decisions from `obs` and the cached lookup payload

The scripted baseline caches only the most recent successful `insurance_lookup` payload. It refreshes the cache whenever a new lookup payload arrives in `obs.last_tool_result`, and never accesses hidden environment internals. This mirrors the information available to an observation-conditioned policy trained on episode traces.

**Separation gate (10 seeds per task).** Pass criterion: `mean(scripted) − mean(random) ≥ 0.20` per task.

| Task | random | no_op | scripted | Δ(scripted − random) |
|---|---|---|---|---|
| `easy_cashless` | 0.454 | 0.468 | 0.961 | +0.506 |
| `medium_multi_payer` | 0.310 | 0.230 | 0.967 | +0.658 |
| `hard_drift` | 0.194 | 0.159 | 0.722 | +0.528 |

**Exploit gate.** Five exploits — `ack_spammer`, `escalate_everything`, `oscillator`, `double_count`, `periodic_lookup` — run on `easy_cashless` and `hard_drift`, 5 seeds each. Pass criterion: each exploit's mean score ≤ no_op mean on the same task (with a 1e-3 tolerance for penalty-cap ties).

| Exploit | Δ(no_op − exploit), easy | Δ(no_op − exploit), hard_drift |
|---|---|---|
| `ack_spammer` | +0.216 | +0.012 |
| `escalate_everything` | +0.213 | +0.000 |
| `oscillator` | +0.267 | +0.066 |
| `double_count` | +0.000 | +0.000 |
| `periodic_lookup` | +0.002 | +0.005 |

`double_count` and `escalate_everything` ties against no_op reflect the penalty cap (0.50) being hit by both the exploit and the baseline; neither is a profitable attack. `periodic_lookup` — a policy that polls `insurance_lookup` every 5 steps and submits claims without any `coding_engine` work — scores within ±0.005 of no_op on both tasks, confirming that drift-timing memorisation via uniform polling does not beat the do-nothing baseline. Three grader changes were needed to reach this: (a) `process_auditability` returns 0 when the trajectory contains no `coding_engine` calls, (b) the standalone "no insurance_lookup before submit" penalty was removed as redundant with (c) a new per-submit penalty (0.05 uncapped) for submitting a claim with no prior `coding_engine` call on that claim.

**Drift-task effect.** After removing hidden-state reads from `ScriptedHeuristicPolicy`, seed-randomising the drift step over 30 candidates, and neutralising the periodic-lookup attack, the scripted policy scores 0.961 on `easy_cashless` and 0.722 on `hard_drift` (Δ = 0.239). Two of the obvious alternative explanations — scripted winning via privileged state access, scripted exploiting a fixed drift step — have been structurally removed. The observed gap is consistent with the pitch that policy re-query under drift closes a meaningful portion of the hard-task gap. It is correlational evidence, not a causal claim; the remaining alternative explanation (e.g., v1.4 rules are mechanically harder on this claim mix independent of detection) has not been isolated.

**no_op score, stated plainly.** no_op now scores 0.468 on `easy_cashless`, 0.230 on `medium_multi_payer`, 0.159 on `hard_drift`. Spec v3 §6 predicted 0.05–0.15. Easy is still above that band because identity fields and three carry-through policy fields are pre-populated in the dirty state by design, giving no_op correct-by-default credit on `final_correctness`. We are treating this as acceptable for now because (a) rank ordering holds on every task with ≥ 0.51 separation between scripted and random, (b) all five tested exploits score ≤ no_op on both tested tasks within 1e-3 tolerance, (c) the formal pass gate is separation, not absolute band. The principled fix, if a reviewer demands a lower easy absolute, is to add light identity corruption (name typos, date-format drift) with a matching scripted cleanup path; we are deferring that until the hero-mechanic pipeline is end-to-end verified.

### Not yet verified

- No FastAPI server (`app.py`) written for `medibill`.
- `openenv validate` has not been run against v3 code.
- `Dockerfile` and `openenv.yaml` still target Round 1's `dataclean_env.server.app:app`.
- Typed client (`medibill/client.py`) not written.
- `pyproject.toml` still declares the Round 1 package name.
- HF Space not deployed; local Docker image not built.
- No SFT, RFT, or GRPO runs. No reward curves.
- Drift-step randomisation now draws from 30 candidates (`range(10, 40)`), and `periodic_lookup` is in the exploit suite — both changes closed the fixed-schedule attack. Broader drift-step distributions (continuous sampling, variable-position drift per claim) have not been tested.
- No end-to-end test that `final_correctness` and `policy_compliance` never score the same cell across all seeds/tasks. The import-time assert covers the field-name sets, not per-claim runtime behaviour.
- Only `hard_drift` carries ambiguous cells. `abstention_quality` is N/A (not exercised) on easy and medium.
- Exploit-gate statistics are 5 seeds per (exploit × task) for 5 exploits × 2 tasks. A public claim needs ≥ 20 seeds with CI bounds.
- No SME review of the rubric.

### Known residual weaknesses a reviewer will attack

1. no_op at 0.468 on easy is still above the original spec band. The defence above is reviewer-safe but not reviewer-silencing.
2. `abstention_quality` is exercised only on `hard_drift`. Any claim about calibrated abstention rests on one task and two curated cells.
3. The 0.239 hard-vs-easy scripted drop is correlational. No experiment has yet isolated "learned re-verification" as the cause.
4. Exploit tolerance of 1e-3 papers over the penalty-cap ties between `escalate_everything` / `double_count` and no_op on hard_drift. True, but the cap is load-bearing: without it a single-action trajectory could accrue unbounded penalties.

---

## 2. Judge-safe summary

MediBill-Env is an OpenEnv environment in which a language-model agent plays a hospital billing coder operating under IRDAI's 1-hour pre-authorization and 3-hour discharge clocks. The agent has three information-gathering tools — `ehr_query`, `insurance_lookup`, `coding_engine` — and two meta-actions, `escalate_to_human` and `submit_claim`. Diagnoses are drawn from ICD-10-CM (CMS public domain); procedures use SYNTH-PROC-v1, a purpose-built synthetic ontology with no dependency on AMA CPT; pricing references the CGHS package-rate list.

The distinguishing mechanic is silent mid-episode policy drift. On hard tasks the active insurance policy is replaced at a seed-selected step with no announcement in the observation stream. In the current design, the only supported path for an agent to observe the new policy is a fresh `insurance_lookup` call after drift. Claims are graded against the policy active at submit time, not against whichever policy the agent believes to be current.

The grader is a deterministic six-axis rubric with explicit field partitioning. Identity fields and policy-sensitive fields are disjoint by construction (enforced by an import-time assertion), and `policy_compliance` is scored only on the six fillin fields that the agent actively populates. Axes that are structurally inapplicable to a task contribute zero; their weight is redistributed proportionally across the applicable axes. Anti-reward-hacking defences are specific penalty rules and an exploit-test suite; four tested attack patterns earn at most the score of a do-nothing baseline on both tested tasks.

Baselines are tool-faithful under the current implementation, reading provider and policy only from agent-visible tool payloads. A scripted heuristic separates from random by 0.30–0.45 across three difficulty tiers and scores 0.24 lower on the drift task than on the baseline task after hidden-state access is removed and drift timing is randomised. These effects are consistent with the design intent; end-to-end training curves are not yet available.

---

## 3. Concrete risks / blockers before FastAPI + HF deploy

Ordered by severity. Item 4 (drift candidate set) is the most intellectually dangerous unresolved issue and is called out ahead of deployment packaging.

**Blockers (must resolve before any deploy attempt)**

1. `pyproject.toml` declares `dataclean_env`. A wheel build will not install `medibill` without a `[project]` update and re-declared `package-dir` entry. Installing alongside Round 1 risks import collisions.
2. `medibill/server/app.py` does not exist. Nothing for `openenv validate` or a Docker `CMD` to start.
3. `Dockerfile` and `openenv.yaml` still target `dataclean_env.server.app:app`. Must be forked to `medibill/`-scoped files; must not overwrite Round 1's.

**Major (remaining; fix before training)**

5. Typed client (`medibill/client.py`) missing. SFT trace generation and inference harnesses will need it.
6. No per-claim regression test asserting that no single cell is scored by both `final_correctness` and `policy_compliance` across all (seed × task) pairs. The import-time assertion covers the field-name sets, not runtime partition behaviour.

**Minor (fix before onsite)**

7. Ambiguous-cell set exists only on `hard_drift`. `abstention_quality` is N/A on `easy_cashless` and `medium_multi_payer`; if we want to demo the axis, we need at least one curated ambiguous cell on a visible task.
8. No identity-corruption path. no_op on `easy_cashless` scores above the original spec band. Not a separation-gate issue, but a likely reviewer question.
9. `README.md` does not yet carry the SYNTH-PROC / ICD-10-CM / CGHS / no-CPT disclaimer from spec v3 §8.4.
10. Exploit-gate statistics are 5 seeds × 4 exploits × 2 tasks. Increase to ≥ 20 seeds before any public claim of "exploit gate passes."

**Unknown until runtime**

11. `openenv validate` has not been run on v3 code. The package rename could surface import issues that only appear at validation time.
12. No evidence that a small LM trained on our dense reward signal learns to re-query `insurance_lookup` after drift. This is the first training run's job and is the single largest uncertainty before onsite.
