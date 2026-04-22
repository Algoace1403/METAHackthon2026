# Round 2 Spec v3 — MediBill DataOps

**Hackathon:** Meta × Scaler OpenEnv Hackathon, Round 2
**Onsite:** 25–26 April 2026 (Sat–Sun), Bangalore
**Theme:** #3.1 Professional World Modeling
**Sub-prize targets:** Snorkel AI (primary) + Patronus AI + Scaler AI Labs
**Version:** 3.0 (2026-04-20) — incorporates Codex review of v2
**Prior versions:** `round2-spec.md` (v1), `round2-spec-v2.md` (v2)

---

## 0. Change log (v2 → v3)

Six issues flagged by external verifier (Codex). All accepted.

| # | v3 fix | Reason |
|---|---|---|
| 1 | Rubric axes **formally partitioned** by field set: `final_correctness` on non-policy cells, `policy_compliance` on policy-sensitive cells only. No overlap. | v2 definitions were self-contradictory |
| 2 | **Removed `verify_policy_version` as a standalone tool.** Drift now invalidates the agent's cached rules; the only way to re-ground is to call `insurance_lookup` again. Submission is graded against policy-at-submit-time regardless of what the agent believes. | v2 hero mechanic was ceremonial ("see flag, press ack") |
| 3 | **Honest fallback ladder:** FLOOR = env + grader + 3 baselines + prerecorded demo; BASELINE = + SFT adapter; BETTER = + one RFT iteration; STRETCH = GRPO | v2 go/no-go said "SFT+RFT" as a fallback when SFT itself failed — impossible |
| 4 | **Calendar realigned.** Apr 25 = Sat. Apr 26 = Sun. Today Apr 20 = Mon. 4.5 building days (Tue–Fri), 2 onsite days. | v2 was one day off throughout |
| 5 | Stripped unmeasured quantitative claims: "~3–5 escalations," "+10–15%," "reliably produces a reward delta." | Unsupported |
| 6 | Softened "fills three holes" → "targets three gaps" in technical sections; kept original wording only in the pitch opener. | Technical overclaim |

---

## 1. One-line pitch

> "An OpenEnv where an LLM agent closes cashless health-insurance claims inside India's IRDAI-mandated 3-hour clock — while insurance policy rules drift mid-episode and the agent's cached view of the policy goes stale without warning."

---

## 2. Problem statement (verified facts only)

India's insurance regulator (IRDAI) issued a **Master Circular on Health
Insurance (29 May 2024, effective 31 July 2024)** mandating cashless-claim
turnaround of **1 hour for pre-authorization and 3 hours for final discharge
authorization**. If hospitals miss the 3-hour discharge clock, the overrun is
borne by the insurer from shareholder funds — a structural penalty, not a
notional one.

Under this clock:

- **FY24:** ₹26,000 crore health claims disallowed or repudiated (+19.10% YoY) — IRDAI Annual Report.
- **FY25:** 13% of pre-auths still miss the 1-hour window — IRDAI disclosure.
- **Jan 2025 consumer survey (LocalCircles):** only 25% of policyholders had claims fully approved; 36% outright rejected with invalid reasons.
- **NHCX (National Health Claim Exchange)** live since June 2024, operated by NHA + IRDAI. 12 insurers integrated; hospital onboarding is the documented bottleneck.

The bottleneck is **data reconciliation under time pressure**: discharge
summaries arrive messy, diagnosis codes don't match policy coverage,
insurance providers update coding rules silently, and a human medical coder
has ~180 minutes to produce a clean, policy-compliant claim.

We build an OpenEnv environment where a language-model agent plays that
coder. The agent must query records, reason under a shifting policy, escalate
genuinely ambiguous cases, stay under a cost budget, and submit on time.

---

## 3. Hero mechanic: policy drift invalidates the agent's world model

Competitor scan (RunbookOps, shubh-sd, DhruvKajalkar) confirms no current OpenEnv
submission makes policy-version-induced staleness a graded mechanic. Our v3 design:

### 3.1 How it works

1. At episode start, the active policy is a JSON rule object with a `policy_version` field (e.g., `"v1.3"`). It lists: covered diagnosis ranges, pre-auth thresholds, coding validity rules, amount caps, required fields.
2. The agent can call `insurance_lookup(provider)` → returns rules + `policy_version` at the moment of the call. The **agent's only source of truth** for rules.
3. On medium/hard tasks, at a scripted step, the environment **mutates the active policy** — e.g., bumps to `v1.4` with: a diagnosis code no longer covered, a new pre-auth threshold, a renamed field, a new required signature.
4. **The environment does NOT announce drift.** There is no `version_changed` flag. The only way the agent learns the policy changed is by re-calling `insurance_lookup` and comparing the returned `policy_version` to what it saw before.
5. `submit_claim` is scored against the policy **at submit time**, not against whatever rules the agent saw earlier. If the agent's cached mental model is stale, `policy_compliance` drops.
6. Drift bonus is awarded **only if** a fresh `insurance_lookup` call exists in the trajectory between the last drift event and `submit_claim`, AND the final claim is correct.

### 3.2 Why this is no longer ceremonial

In v2 the agent could earn drift credit by calling an ack tool. In v3, there is no ack tool. The agent must **actually re-query the policy** to recover the new executable rules. The grader doesn't reward the agent for "noticing drift" — it rewards the agent for **producing a claim consistent with the current policy**, which is only possible via a fresh lookup.

A hostile judge asking "isn't this just schema validation?" gets: "No — the ground truth for compliance mutates during the episode and the agent is never told. It has to re-read the policy or lose on `policy_compliance`."

### 3.3 One-sentence hook

> **"The policy changes mid-episode, silently. Agents with a stale mental model get graded wrong — the only recovery is to re-query the world."**

---

## 4. Environment architecture

### 4.1 Reuse from Round 1 (≈ 65%)

| Component | Status | Reason |
|---|---|---|
| `models.py` | Keep + extend | Add `policy_version`, `tool_surface` fields; remove `verify_policy_version` refs |
| `server/environment.py` — reset/step, budget, delta reward, entity-ID alignment | Keep | Proven |
| `server/grader.py` — alignment, cell matching, utility probes | Keep core + replace composite weights + field partitioning | Rewrite §6 weights |
| `server/data_generator.py` — DataCorruptor pipeline | Keep + add `policy_drift` handler | Existing 14 types intact |
| `scripts/train_grpo.py` + `reward_functions.py` | Keep + add 1 new reward fn | Pipeline pattern works |
| `Dockerfile` + `openenv.yaml` | Keep + rename to `medibill` | HF-Space-ready |
| 20 test files (228 tests) | Keep + add ~30 medical/drift/exploit tests | Regression safety net |

### 4.2 Tools visible to the agent (3 total, primitives hidden)

| # | Tool | Returns | Cost | Side-effects |
|---|---|---|---|---|
| 1 | `ehr_query(patient_id)` | Patient record slice (diagnosis, procedure, amount, hospital_id, dob) | 0.5 | Read-only |
| 2 | `insurance_lookup(provider)` | Current rules for that provider + `policy_version` at call time | 1.0 | Read-only; critical for drift recovery |
| 3 | `coding_engine(action, claim_id, field, value)` | Apply/fix a code; returns per-field validation | 2.0 | Mutates the claim under construction |

Meta-actions (no state change):

| Meta | Purpose | Cost |
|---|---|---|
| `escalate_to_human(claim_id, field, reason)` | Calibrated abstention | 0.3 |
| `submit_claim(claim_id)` | Terminal; triggers scoring of that claim | 0.0 |

Round 1 primitives (`fix_value`, `fill_missing`, `merge_duplicates`, etc.) remain **internal-only** — the agent never sees them. `coding_engine` is the only mutation path. Closes "tools are wrappers" novelty gap.

### 4.3 Tasks

| Task | ID | Claims | Provider | Drift | Budget | Max steps |
|---|---|---|---|---|---|---|
| Easy cashless | `easy_cashless` | 6 | CGHS (v2024.1) | none | 80 | 60 |
| Medium high-volume | `medium_multi_payer` | 10 | Star (v1.4, latest) | none | 150 | 100 |
| Hard drift reconciliation | `hard_drift` | 12 | Star (v1.3 → v1.4 mid-episode) | silent, at a seed-selected step in `range(10, 40)`; not announced | 200 | 140 |

(A true multi-provider mix for medium is a Day-2 extension; the current data generator accepts one provider per episode. The name `medium_multi_payer` is retained for registry stability.)

All data synthetic. Diagnoses from **ICD-10-CM** (CMS public-domain). Procedures from **SYNTH-PROC-v1** (our invented ontology — see §8). Rates from **CGHS package-rate list** (MoHFW India, public rate list). No CPT.

---

## 5. Reward design (explicit field partitioning, anti-hacking)

### 5.1 Field partition (critical for anti-double-counting)

Every claim has a schema. Each field is tagged in the schema as either `identity` or `policy_sensitive`. These sets are **disjoint by construction**.

- **Identity fields** (graded by `final_correctness`): `patient_id`, `patient_name`, `dob`, `gender`, `hospital_id`, `admission_date`, `discharge_date`, `amount_billed`, `amount_paid`, `line_item_descriptions`.
- **Policy-sensitive fields** (graded by `policy_compliance`): `diagnosis_code` (validity depends on current policy's covered-code list), `procedure_code` (validity depends on current coverage), `pre_auth_flag` (required iff current policy threshold says so), `required_signatures[]` (required set depends on current policy).

A cell is scored by **exactly one** axis. The axis is determined by the schema tag, not by the agent's behaviour. No overlap.

### 5.2 Composite weights

| Axis | Weight | Definition |
|---|---|---|
| `final_correctness` | **45%** | Fraction of identity cells matching ground truth (type-aware compare) |
| `policy_compliance` | **20%** | Fraction of policy-sensitive cells valid under **policy-at-submit-time** |
| `abstention_quality` | **15%** | +credit for escalating truly-ambiguous cells; −credit for escalating clear-cut |
| `process_auditability` | **10%** | Tool-call order checks: `ehr_query` before `coding_engine`; `insurance_lookup` at least once before `submit_claim` on any task |
| `efficiency` | **5%** | `1 - (budget_spent / budget_allocated)`; gated on ≥10% `final_correctness` |
| `drift_bonus` | **5%** | Gated: awarded only if `final_correctness + policy_compliance ≥ 0.80` AND fresh `insurance_lookup` exists between last drift and `submit_claim` |

### 5.3 Penalties (capped at 0.50 cumulative)

| Trigger | Penalty |
|---|---|
| `submit_claim` with no `insurance_lookup` call at all | −0.10 |
| Wrong code applied on a field flagged ambiguous at task-gen time | −0.08 |
| `escalate_to_human` on a clear-cut cell (entity_id, column) not in ambiguous set | −0.03 per escalation, no cap |
| Identical tool call ≥3× in a row | −0.05 per repeat beyond 2nd |

### 5.4 Bonuses (capped at 0.15, gated on `final_correctness ≥ 0.70`)

| Trigger | Bonus |
|---|---|
| Correct escalation on a genuinely ambiguous cell (in task's `ambiguous_cells` set) | +0.03 |
| All identity fields correct for at least 80% of claims | +0.05 |
| Cross-claim consistency verified (same patient across claims uses same patient_id) | +0.03 |

### 5.5 Anti-hacking defense (each exploit class has a specific mitigation)

| Exploit | v3 Defense |
|---|---|
| **Audit-log farming** | No audit-log tool exists. `process_auditability` checks only tool-call **order** (before/after) and **presence**, not count. Hard cap 10%. |
| **Drift oscillation** | `drift_bonus` is gated on `final_correctness + policy_compliance ≥ 0.80` AND requires a post-drift `insurance_lookup`. Oscillating values gives no bonus. |
| **Abstention collapse** | Each wrong `escalate_to_human` costs 0.03, **uncapped**. Expected ambiguous-cell count per task is small (2–3 on hard); over-escalation yields net-negative. |
| **Double-counting** | Field partition in §5.1 is disjoint by construction in the schema. `final_correctness` and `policy_compliance` cannot score the same cell. |

---

## 6. Grader validation (hard gate before any training)

Before SFT begins, run 100 episodes per baseline on each task. Show mean score ± 95% CI.

| Baseline | Description | Expected band |
|---|---|---|
| `random` | Uniform random action selection | 0.00–0.10 |
| `no_op` | Only `submit_claim` immediately | 0.05–0.15 |
| `scripted_heuristic` | Rule-based: ehr_query all → insurance_lookup → coding_engine obvious fixes → submit | 0.25–0.45 |
| `strong_oracle` | Uses `_entity_id` hints to cheat (for sanity check only, not a submission baseline) | 0.85–1.00 |

**Pass criterion:** `scripted_heuristic - random ≥ 0.20` on each task.

If that separation isn't there, the grader is broken. **Stop and fix before proceeding.**

### 6.1 Exploit-test suite (must all fail — exploits earn ≤ no_op score)

1. "Ack spammer": calls nothing except `insurance_lookup` 100× then `submit_claim` without using the data. Expected: `final_correctness` near zero → total ≤ no_op.
2. "Escalate everything": calls `escalate_to_human` on every field. Expected: penalty dominated → total ≤ 0.
3. "Oscillator": flips a `diagnosis_code` back and forth 10×. Expected: no drift bonus, no process bonus.
4. "Double-count attempt": constructs a trajectory where `diagnosis_code` is correct under BOTH old and new policy simultaneously. Expected: counted once under `policy_compliance`, never under `final_correctness`.

All four tests committed in `tests/test_exploits.py` before SFT.

---

## 7. Training strategy — honest fallback ladder

### 7.1 Floor (must ship; zero training required)

- Environment deployed on HF Space AND locally as Docker image
- 3 baselines with score-separation plot
- 90-second prerecorded demo video
- Pitch deck

Even if all training fails, this is a legitimate submission that satisfies all minimum requirements.

### 7.2 Baseline demo — SFT on Qwen2.5-3B

**Data generation:**
1. Run `scripted_heuristic` + 2 variant policies × 20 seeds = 60 trajectories.
2. Filter by env grader: keep trajectories with composite reward > 0.30.
3. Train set: 40. Eval set held out: 10.

**Training:**
- Qwen2.5-3B-Instruct + LoRA (r=32, alpha=64, target all-linear)
- lr=1e-4, batch=4, grad_accum=4, 3 epochs
- Unsloth QLoRA on Colab A100 40GB
- Expected wall-clock: ~30 min

**Pass criterion:** SFT adapter beats both `random` and `scripted_heuristic` on the held-out 10-episode eval. If not, ship Floor only.

### 7.3 Better demo — RFT iteration

- Start from SFT checkpoint
- Sample 4 trajectories per prompt with SFT checkpoint
- Filter top-1 by env grader → new set of ~40 improved trajectories
- SFT again, 2 epochs
- Expected wall-clock: ~20 min

**Pass criterion:** RFT checkpoint ≥ SFT checkpoint on eval. If not, ship Baseline.

### 7.4 Stretch — GRPO

Only attempted if Floor + Baseline + Better are all green by onsite Day 1 noon.

- TRL GRPO + vLLM colocate mode (`vllm_gpu_memory_utilization=0.3`)
- Init from RFT adapter
- `num_generations=6`, `max_prompt_length=1024`, `max_completion_length=512`, `lr=5e-6`, KL beta=0.04
- Step target: 400 (Unsloth confirms ≥300 needed for visible curve)
- Time budget: ~8h on single A100
- Abort criteria: flat reward at step 100 OR OOM → keep RFT narrative

### 7.5 Five reward functions (Round 1 had 4)

1. `reward_valid_action` — parseable tool call
2. `reward_episode_score` — terminal rubric composite
3. `reward_efficiency` — budget remaining / budget allocation
4. `reward_no_destruction` — penalizes wrong `coding_engine` calls on correct fields
5. **NEW** — `reward_drift_recovery`: +1 if a fresh `insurance_lookup` call exists between last drift event and `submit_claim` AND final claim is correct, else 0

### 7.6 SFT target coverage (what SFT can and cannot teach)

The scripted-heuristic trajectories feeding SFT exercise four of the six
rubric axes. The remaining two require RL signal:

| Axis | SFT can learn it? | Why |
|---|---|---|
| `final_correctness` | **Yes** | Scripted fills identity cells correctly; imitation suffices. |
| `policy_compliance` | **Yes** | Scripted reads the policy via `insurance_lookup` and writes the six fillin fields correctly; imitation suffices. |
| `process_auditability` | **Yes** | Scripted follows the `ehr_query → coding_engine`, `insurance_lookup → submit_claim` orderings; imitation suffices. |
| `efficiency` | **Yes (partial)** | Scripted uses a bounded number of calls per claim. SFT can imitate the pattern but not the trade-off; an RL pass tightens it further. |
| `abstention_quality` | **No** | Scripted never calls `escalate_to_human`. The ambiguous-cell list lives in the task spec — no agent-visible signal exposes it. An abstention-aware scripted policy would have to read hidden state, breaking tool-faithfulness. **Abstention is an RL-only target**: RL learns to escalate from the per-escalation bonus/penalty, not from imitation. SFT-only checkpoints are expected to score zero on this axis. |
| `drift_bonus` | **No (in a principled sense)** | Scripted detects drift by *time* (always re-looks-up around the episode mid-point) rather than by *policy staleness*. SFT will imitate the schedule, not the reasoning. A meaningful `drift_bonus` requires the RL pass to connect post-drift compliance errors back to the decision to re-query. SFT may get *some* bonus credit structurally, but claiming RL-independence on this axis would be overclaim. |

Implication for the training pipeline: **SFT is the baseline**; RFT improves marginally via filtered self-play; GRPO is the path that actually trains the abstention and drift-reasoning axes. If compute is tight onsite, we ship SFT and name the two un-targeted axes as honest limitations in the pitch Q&A.

---

## 8. Medical coding, data, and licensing safety

### 8.1 Code systems used (all legally clean)

| System | Source | License | Role |
|---|---|---|---|
| **ICD-10-CM** | CMS/NCHS (USA) | Public domain | Diagnosis codes |
| **HCPCS Level II** | CMS | Public domain | Supplies where relevant |
| **LOINC** | Regenstrief Institute | LOINC License (redistributable w/ notice) | Labs |
| **RxNorm** | NLM | Public domain | Drug names |
| **CGHS package rates** | MoHFW, Govt. of India | Public rate list | INR pricing |
| **SYNTH-PROC-v1** | This project | Original work (MIT) | Procedure codes |
| **NAMASTE (AYUSH codes)** | Ministry of AYUSH | Public | Optional traditional-medicine coverage |

### 8.2 What we do NOT use

- AMA CPT codes — copyrighted
- SNOMED CT — UMLS license restricts redistribution
- NABH codes — copyrighted
- MIMIC-IV raw — credentialed access + DUA

### 8.3 SYNTH-PROC-v1 design

Format: `PROC-<SPECIALTY>-<NNN>`. Examples:
- `PROC-CARD-014` — PCI with drug-eluting stent
- `PROC-ORTH-032` — Knee arthroscopy
- `PROC-CARD-001` — ECG baseline

Ontology file: `data/ontology/synth_proc_v1.json`. Cross-referenced to CGHS where a package exists. **No mapping to CPT.** Any resemblance is coincidental.

### 8.4 Required README disclaimer

> This environment uses only public-domain or openly-licensed medical coding systems: ICD-10-CM (CMS, public domain), LOINC (Regenstrief License), RxNorm (NLM, public domain), HCPCS Level II (CMS, public domain), CGHS package rates (Govt. of India, MoHFW public rate list). SYNTH-PROC-v1 is a synthetic procedure ontology created for this project; it is NOT AMA CPT, does NOT map to CPT, and must not be used for real billing. All patient data is synthetic. This project is for research, education, and RL training only.

---

## 9. Deployment (with fallbacks, in order of preference)

1. **HuggingFace Space** — primary, reuses Round 1 Dockerfile. Rename to `medibill`. Validate with `openenv validate` (6 runtime + 8 file checks).
2. **Local Docker image** — pre-built on the laptop. Same Dockerfile. Runs offline. Trigger if HF Space sleeps, rate-limits, or is unreachable.
3. **Prerecorded 90-second demo video** — full rollout on `hard_drift`. Screen-captured before/after. Last resort if HF and local both fail.

All three must exist before travel on Friday.

---

## 10. Pitch (3 minutes)

### 10.1 Opener — verified regulatory reality (0:00–0:30)

> "IRDAI's Master Circular gives Indian hospitals one hour for pre-authorization and three hours for final discharge on every cashless claim. Miss the clock, and the overrun comes out of the insurer's shareholder fund. Last year ₹26,000 crore of claims were disallowed — up 19 percent. Thirteen percent of pre-auths still miss the window today."

### 10.2 The environment (0:30–1:15)

> "We built an OpenEnv where a language-model agent plays the medical coder under that clock. Three enterprise tools. ICD-10 diagnoses, CGHS rates, a synthetic procedure ontology. Graded by a deterministic rubric with six axes — final correctness, policy compliance, abstention quality, auditability, efficiency, and a gated drift bonus.
>
> And the hero mechanic: the insurance policy changes mid-episode, **silently**. The agent's cached view of the rules goes stale without warning. The only way to recover is to re-query the policy. Submissions are graded against whatever rules are active at submit time — not what the agent thinks they are."

### 10.3 Training result (1:15–2:30)

> "We trained Qwen 2.5-3B. Base model first. Then SFT on forty filtered trajectories. Then RFT iteration using the same grader. Here's the score separation across baselines and checkpoints. [show bar chart] Here's a rollout on the hard drift task. [show before/after]"

*(No specific numbers until measured. The chart shows whatever we actually got.)*

### 10.4 The ask (2:30–3:00)

> "This fills three holes in OpenEnv: multi-app enterprise workflow, mid-episode policy drift, and programmatic-expert grading. It's legally clean — no CPT, no PHI — and India-contextual, because that's where the regulatory clock is real. Thank you."

### 10.5 Q&A preparation

| Question | Answer |
|---|---|
| "Is this a real problem?" | IRDAI FY24 report: ₹26k cr disallowed. LocalCircles Jan 2025: 36% rejection with invalid reasons. IRDAI mandated the clock via Master Circular 29 May 2024. |
| "Why RL vs a rules engine?" | Rules engines catch static schema errors. An RL-trained agent learns the tradeoff between cleanup depth, tool cost, and a shifting policy under dense reward shaping. |
| "Who signed off on the rubric?" | Expert-inspired — not reviewed by an SME. Weights are designed to be defensible and reproducible. We welcome SME critique post-hackathon. |
| "How do you prevent reward hacking?" | Drift bonus is gated on final correctness AND on a post-drift `insurance_lookup`. Auditability capped at 10%. `final_correctness` and `policy_compliance` are on disjoint field sets by schema construction. Exploit suite in `tests/test_exploits.py`. |
| "Why 3B not 7B?" | Single-A100 Colab throughput: 7B needs 20–38h for 300 GRPO steps; 3B needs 6–10h. 3B fits one session and ships. |
| "Why not CPT?" | AMA copyright. SYNTH-PROC is our synthetic ontology. Keeps the repo legally open. |
| "What if HF Space is down at demo time?" | Local Docker image on my laptop. If that fails, a prerecorded 90-second rollout video plays. |
| "Biggest risk to your result?" | Training convergence. Shipping path is SFT + optional RFT; GRPO is a stretch bonus. Even with zero trained weights, environment + baselines + exploit tests is a legitimate submission. |

### 10.6 Factual landmines (never claim)

- "IRDAI fines hospitals" — no, penalty falls on **insurers**.
- Specific average processing time in minutes — IRDAI publishes % compliance, not means.
- "QR code mandate 1 April 2026" — unverified.
- "SME signed off on the rubric" — unless one actually does.
- Specific training result numbers — until measured.

---

## 11. Timeline (Apr 20 → Apr 26) — corrected

| Day | Date | Goal | Deliverable | Go/no-go |
|---|---|---|---|---|
| Today | **Mon Apr 20** | Spec v3 + Codex re-verify + Day 1 kickoff | This doc + first code file | Spec approved |
| Day 1 | Tue Apr 21 | Medical data generator + SYNTH-PROC-v1 ontology + 3 task defs | `generate_clean_medical_records`, `easy_cashless` passes smoke test | Data smoke test |
| Day 2 | Wed Apr 22 | 3 tools + meta-actions + `policy_drift` corruption handler | `ehr_query`, `insurance_lookup`, `coding_engine`, `escalate_to_human`, `submit_claim` wired; drift handler mutates policy mid-episode | `openenv validate` |
| Day 3 | Thu Apr 23 | Grader v3 with field partition + 3-baseline validation + exploit suite | Score-separation plot; all 4 exploits score ≤ no_op | ≥0.20 gap between scripted and random |
| Day 4 | Fri Apr 24 | 40-trajectory SFT set + SFT on Colab + HF Space deploy + local Docker + prerecorded demo | Trained adapter, live URL, Docker image, video | SFT beats scripted on held-out eval |
| Onsite Day 1 | **Sat Apr 25** | RFT iteration; optional GRPO if time permits | Second adapter + reward curve (if GRPO) | Curve rises visibly OR RFT ≥ SFT |
| Onsite Day 2 | **Sun Apr 26** | Final eval + demo + 3-min pitch | Score chart + pitch delivery | Trophy |

4.5 building days (Tue–Fri) + 2 onsite days = 6.5 working days.

---

## 12. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| HF Space sleep / rate-limit mid-demo | High | Local Docker + prerecorded video |
| GRPO doesn't converge | Med | GRPO is stretch only; RFT ships reliably |
| Grader broken (no score separation) | High | Thu gate blocks training until ≥0.20 separation |
| Colab gives T4 not A100 | Med | Qwen2.5-3B fits on T4 with LoRA; throughput halved, still ships |
| Judges find "healthcare" boring | Med | IRDAI clock framing + silent-drift hero mechanic |
| Reward hacking discovered at demo | High | Exploit-test suite run Thu before onsite |
| OpenEnv version bump | Low | Pin `openenv-core[core]==0.2.2` |
| User nervous in pitch | High | Rehearsal every evening from Mon; Q&A doc + landmine list |
| Legal/IP risk from real codes | Med | SYNTH-PROC + ICD-10-CM + CGHS only; disclaimer in README |
| Codex flags more v3 issues | Low | Run one more verifier pass before coding starts |

---

## 13. Hard go/no-go gates

1. **Tue end-of-day:** medical data generator produces 40 plausible patient records deterministically; smoke test passes.
2. **Thu end-of-day:** 3-baseline validation shows ≥0.20 score separation between `scripted_heuristic` and `random`. If not, stop and fix grader before training.
3. **Fri end-of-day:** HF Space live AND local Docker image built AND prerecorded demo recorded AND SFT adapter beats `scripted_heuristic` on eval. This is the minimum submission requirement.
4. **Sat onsite noon:** RFT iteration complete AND ≥ SFT on eval. If not, ship SFT-only demo. GRPO must not begin before this gate is met.

---

## 14. Sign-off

Spec v3 addresses all 6 issues flagged by Codex in the v2 review. Training plan is honest about what can and cannot be shipped under each scenario. Hero mechanic is no longer ceremonial. Reward axes are formally partitioned.

**Next action on approval:** write `server/data_generator.py::generate_clean_medical_records` + `data/ontology/synth_proc_v1.json` + `server/tasks.py::easy_cashless` as the Tuesday deliverable.
