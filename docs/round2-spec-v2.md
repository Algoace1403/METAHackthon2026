# Round 2 Spec v2 — MediBill DataOps

**Hackathon:** Meta × Scaler OpenEnv Hackathon, Round 2
**Onsite:** 25–26 April 2026, Bangalore
**Theme:** #3.1 Professional World Modeling
**Sub-prize targets:** Snorkel AI (primary) + Patronus AI + Scaler AI Labs
**Version:** 2.0 (2026-04-20) — supersedes v1 after GPT-5 external review
**Prior version:** `round2-spec.md`

---

## 0. Change log (v1 → v2)

Fixes driven by external review (11 blockers/majors flagged). Ordered by severity.

| # | Change | Why |
|---|---|---|
| 1 | Hero mechanic: **policy-version drift** forcing `verify_policy_version` re-check | v1's "column renamed" drift was derivative; competitor scan confirmed no OpenEnv submission has version-invalidation as a reward axis |
| 2 | Cut tools from 5 to **3** (plus `verify_policy_version` meta-action); primitives hidden from agent | v1 tools were wrappers; judges would see through "naming, not mechanics" |
| 3 | Hard-task size cut from 40 claims to **12 max** | Overscoped; judges care about one drifted case demoed well |
| 4 | SFT corpus cut from 200 → **50 trajectories via RFT** (Rejection-Finetuning) | Hackathon-appropriate; reuses env grader; avoids GPT-4 CPT leakage |
| 5 | RL model: 7B → **Qwen2.5-3B-Instruct** | Single A100 Colab cannot run 7B GRPO to 300 steps in one session |
| 6 | GRPO demoted from centerpiece → **bonus**. RFT is the shipping fallback. | Training reliability > ambition |
| 7 | Rubric renamed from "simulated senior coder" → **"expert-inspired deterministic rubric"** | No SME signed off; over-claim risk |
| 8 | All placeholder score numbers (0.26/0.41/0.68, "senior-coder parity") **removed** | Invented numbers = instant dismissal |
| 9 | CPT codes fully removed; switched to **ICD-10-CM (public) + SYNTH-PROC-v1 (ours) + CGHS package rates (MoHFW India)** | AMA CPT copyright risk in open repo |
| 10 | Pitch reframed around **IRDAI 1-hour pre-auth / 3-hour discharge clock** (verified Master Circular 29 May 2024) | Melodramatic "Rajesh" hook replaced with concrete regulatory reality |
| 11 | Added **local Docker fallback + prerecorded demo**; grader validated via **3 baselines (random/scripted/strong)** + 4 exploit tests | HF Space fragility + grader-validity proof |

---

## 1. One-line pitch

> "An OpenEnv where an LLM agent closes cashless health-insurance claims inside India's IRDAI-mandated 3-hour clock — while insurance policy rules drift mid-episode and the agent must re-verify the policy version before it submits."

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
- **Jan 2025 consumer survey:** only 25% of policyholders had claims fully approved; 36% outright rejected.
- **NHCX (National Health Claim Exchange)** live since June 2024, operated by NHA + IRDAI. 12 insurers integrated; hospital onboarding is the documented bottleneck.

The bottleneck is **data reconciliation under time pressure**: discharge
summaries arrive messy, ICD codes don't match policy coverage, insurance
providers silently update their coding rules, and a human medical coder has
~180 minutes to produce a clean, policy-compliant claim.

We build an OpenEnv environment where a language-model agent plays that
coder. The agent must:

1. Query patient records across 3 enterprise-style tools.
2. Check the **currently-active policy version** from the insurance-lookup tool.
3. Detect mid-episode policy drift (rule changes) and re-verify before submission.
4. Escalate genuinely ambiguous cases; not hallucinate on them.
5. Stay under a fixed action-cost budget.
6. Be graded by an expert-inspired deterministic rubric (process + outcome).

---

## 3. Hero mechanic: policy-version drift (unclaimed in OpenEnv)

Competitor matrix (from agent scan of HF + GitHub) shows:

| Feature | RunbookOps (parrth020) | shubh-sd | DhruvKajalkar | **MediBill v2** |
|---|---|---|---|---|
| Deterministic rubric | ✓ | ✗ | ✗ | ✓ |
| Evidence-aware grading | ✓ | ✗ | ✗ | ✓ |
| Safe-closure penalty | ✓ | ✗ | ✗ | ✓ |
| Step-budget tiers | ✓ | ✗ | ✗ | ✓ |
| Multi-app tool surface | ✗ | ✗ | ✗ | **✓** |
| Schema drift | ✗ | ✗ | ✗ | **✓** |
| **Policy-version drift as reward axis** | **✗** | **✗** | **✗** | **✓ (hero)** |
| Cost budget as rubric axis | ✗ (step only) | ✗ | ✗ | **✓** |
| Positively-scored abstention | ✗ | ✗ | ✗ | **✓** |
| Cross-app consistency check | ✗ | ✗ | ✗ | **✓** |
| Held-out utility probes | ✗ | ✗ | ✗ | **✓** |

### Mechanic in detail

1. Each episode begins with an **active policy version** exposed via `insurance_lookup`. Example: `{"provider": "Star Health", "policy_version": "v1.3", "rules": {...}}`.
2. At a scripted step (medium: 0%, hard: 100% of episodes), the environment mutates the active policy: rule additions, threshold changes, column renames, or pre-auth requirements appear.
3. The environment does **not** silently lie. It exposes a `version_changed` flag in observation metadata. The agent must:
   - Call `verify_policy_version` after any observation where `version_changed=True`.
   - Re-check any earlier plan that relied on the old rules.
   - Fail gracefully (`escalate_to_human`) if the new rules make the claim ambiguous.
4. Scoring: submissions made without a `verify_policy_version` call since the last drift event are **graded against the new truth**, not the stale plan. No hidden penalty; transparent grading.

This is the one-sentence hook judges can repeat:
> **"The policy changes mid-episode. Agents that don't re-verify the version before submitting get graded against the new truth."**

---

## 4. Environment architecture

### 4.1 Reuse from Round 1 (≈ 65%)

| Component | Status | Reason |
|---|---|---|
| `models.py` | Keep + extend | Issue-first observation; add `policy_version`, `version_changed`, `tool_surface` fields |
| `server/environment.py` — reset/step, budget, delta reward, entity-ID alignment | Keep | Proven; 228 tests cover it |
| `server/grader.py` — alignment, cell matching, utility probes | Keep + replace composite weights | Core matching is solid; weights change per §6 |
| `server/data_generator.py` — DataCorruptor pipeline | Keep + add `policy_drift` handler | Existing 14 corruption types intact |
| `scripts/train_grpo.py` + `reward_functions.py` | Keep + add 1 new reward fn | GRPO pipeline + environment_factory pattern |
| `Dockerfile` + `openenv.yaml` | Keep + rename to `medibill_env` | HF-Space-ready |
| 20 test files (228 tests) | Keep + add ~30 medical/drift tests | Regression safety net |

### 4.2 New in v2 (≈ 35%)

#### 4.2.1 Three visible tools (primitives hidden from agent)

| # | Tool | Observation returned | Cost |
|---|---|---|---|
| 1 | `ehr_query` | Patient record slice (diagnosis, procedure, amount, hospital_id, dob) | 0.5 |
| 2 | `insurance_lookup` | Current policy rules + `policy_version` field | 1.0 |
| 3 | `coding_engine` | Apply/fix a code on a claim — returns validation result + normalized code | 2.0 |

Plus **meta-actions** (no state change):

| Meta | Purpose | Cost |
|---|---|---|
| `verify_policy_version` | Records that agent re-checked policy post-drift | 0.2 |
| `escalate_to_human` | Calibrated abstention (graded, not just penalized) | 0.3 |
| `submit_claim` | Terminal action; scores the claim | 0.0 |

**Primitives** (`fix_value`, `fill_missing`, `merge_duplicates`, etc. from Round 1) are **internal-only**. Agent never sees them. This closes the "tools are wrappers" novelty gap flagged in review.

#### 4.2.2 Tasks (3 difficulty levels, India-contextual)

| Task | ID | Claims | Insurers | Drift | Budget | Max steps |
|---|---|---|---|---|---|---|
| Easy cashless | `easy_cashless` | 6 | 1 (CGHS) | none | 40 | 30 |
| Medium multi-insurer | `medium_multi_payer` | 10 | 3 (Star, HDFC ERGO, CGHS) | between-episode | 80 | 60 |
| Hard drift reconciliation | `hard_drift` | 12 | 3 + PMJAY | mid-episode (step ~15) | 120 | 100 |

All data synthetic. Diagnoses from ICD-10-CM (CMS public-domain). Procedures from `SYNTH-PROC-v1` (our invented ontology — see §8). Rates from CGHS package-rate list (MoHFW India, public rate list).

#### 4.2.3 Expert-inspired deterministic rubric

Not a simulated person. A **principled checklist** that mirrors what an IRDAI-aware claims auditor looks for.

| Axis | Weight | Signal |
|---|---|---|
| `final_correctness` | **45%** | Submitted claim matches ground-truth fields (type-aware cell match) |
| `policy_compliance` | **20%** | Claim matches currently-active policy rules, not stale ones |
| `abstention_quality` | **15%** | +credit for escalating truly ambiguous, −credit for escalating clear-cut |
| `process_auditability` | **10%** | Tool-call order is sensible; `verify_policy_version` invoked after drift |
| `efficiency` | **5%** | Budget used ≤ allocation; gated on ≥10% correctness |
| `drift_bonus` | **5% (gated)** | Only awarded if final claim is correct AND agent re-verified after drift |

Cumulative penalties (capped at 0.50):

| Trigger | Penalty |
|---|---|
| Submit after drift without `verify_policy_version` call | −0.15 |
| Wrong code applied on ambiguous case | −0.08 |
| Escalated a clear-cut case | −0.03 per escalation |
| Infinite-loop tool spam (>3 identical calls) | −0.05 |

Cumulative bonuses (capped at 0.15, gated on ≥90% row_correctness):

| Trigger | Bonus |
|---|---|
| Detected drift within 3 steps of occurrence | +0.05 |
| Cross-app consistency check passed | +0.03 |
| Correct escalation on genuinely ambiguous cell | +0.03 |

---

## 5. Anti-reward-hacking design

GPT review flagged four exploit classes. Mitigations baked into §4.2.3:

| Exploit | Mitigation |
|---|---|
| **Audit-log farming** | Removed `audit_log` as standalone tool. Auditability rewarded only via `verify_policy_version` + tool-call order checks, hard-capped at 10% weight. |
| **Drift-recovery oscillation** | Drift bonus is **gated on final correctness** and paid per unique-drifted-cell-ever-corrected-and-still-correct-at-submit, not per step. Oscillation yields nothing. |
| **Abstention collapse** | Escalating clear-cut cells costs −0.03 each (uncapped). Model converges to ~3–5 escalations per hard task, matching the designed ambiguity count. |
| **Double-counting** | `final_correctness` and `policy_compliance` are computed on disjoint cell sets (policy-affected columns vs. all columns). No axis rewards the same behaviour twice. |

---

## 6. Grader validation (before any training)

**Go/no-go gate:** before we start SFT, prove the grader separates skill levels.

Run 100 episodes for each baseline and show mean score ± 95% CI:

| Baseline | Description | Expected band |
|---|---|---|
| `random` | Uniform random action selection | 0.00–0.10 |
| `no_op` | Only `submit_claim` immediately | 0.05–0.15 |
| `scripted_heuristic` | Hand-rolled rules (fix obvious formats, standard codes) | 0.25–0.45 |
| `strong_oracle` | Uses `_entity_id` hints to cheat | 0.85–1.00 |

If `random` and `scripted_heuristic` bands overlap, grader is broken — fix before proceeding.

**Exploit tests** (separate suite, must all fail — i.e., exploits must not earn credit):

1. Agent that calls `verify_policy_version` 100 times and submits nothing → score ≈ 0.
2. Agent that calls `escalate_to_human` on every field → penalty-dominated, score < no-op.
3. Agent that flips a value back and forth → no drift-bonus accrual.
4. Agent that calls `coding_engine` but `policy_compliance` should only measure policy-affected columns — no double-count.

---

## 7. Training strategy

### 7.1 Primary: SFT + RFT (reliable demo path)

**Stage A — Generate traces:**
1. Run 3 scripted heuristic policies × 20 seeds = 60 trajectories.
2. Filter by env grader: keep top ~50 with reward > 0.35.
3. Train set: 40. Eval set: 10.

**Stage B — SFT on Qwen2.5-3B-Instruct:**
- LoRA r=32, alpha=64, target all-linear
- lr=1e-4, batch=4, grad_accum=4
- 3 epochs on 40 trajectories
- Time: ~30 min on one A100 40GB (Colab Pro)
- Expected: visibly beats `random` and `scripted_heuristic` baselines

**Stage C — RFT iteration (optional):**
1. Sample 4 trajectories/prompt with SFT checkpoint.
2. Keep top-1 by grader. New set of ~40 improved trajectories.
3. SFT again for 2 epochs. ~20 min.
4. Expected: further +10-15% over Stage B.

### 7.2 Bonus: GRPO (only if Stages A-B-C ship by onsite Day 1 noon)

- TRL GRPO + vLLM colocate mode (`vllm_gpu_memory_utilization=0.3`).
- Init from Stage C adapter.
- `num_generations=6`, `max_prompt_length=1024`, `max_completion_length=512`, `lr=5e-6`, `KL beta=0.04`.
- Step target: 400 (demo), 600 (stretch). Time budget: ~8h.
- Abort criteria: flat reward at step 100 OR OOM → revert to RFT narrative.

### 7.3 Five GRPO reward functions (extends Round 1's four)

1. `reward_valid_action` — model emits parseable tool call.
2. `reward_episode_score` — terminal rubric composite.
3. `reward_efficiency` — budget remaining / budget allocation.
4. `reward_no_destruction` — penalises `coding_engine` on wrong cells.
5. **NEW:** `reward_drift_compliance` — +1 if agent called `verify_policy_version` within 3 steps of each drift event, else 0. Per-step dense signal.

---

## 8. Medical coding, data, and licensing safety

### 8.1 Code systems used (all legally clean)

| System | Source | License | Role |
|---|---|---|---|
| **ICD-10-CM** | CMS/NCHS (USA) | Public domain | Diagnosis codes |
| **HCPCS Level II** | CMS | Public domain | Supplies (where relevant) |
| **LOINC** | Regenstrief Institute | LOINC License (redistributable w/ notice) | Labs |
| **RxNorm** | NLM | Public domain | Drug names |
| **CGHS package rates** | MoHFW, Govt. of India | Public rate list | INR pricing |
| **SYNTH-PROC-v1** | This project | Original work (MIT) | Procedure codes |
| **NAMASTE (AYUSH codes)** | Ministry of AYUSH | Public | Optional traditional-medicine coverage |

### 8.2 What we do NOT use

- **AMA CPT codes** — copyrighted; commercial license required for any open use.
- **SNOMED CT** — US NLM UMLS license restricts redistribution.
- **NABH codes** — copyrighted.
- **MIMIC-IV raw data** — credentialed access + DUA.

### 8.3 SYNTH-PROC-v1 design

Format: `PROC-<SPECIALTY>-<NNN>`. Examples:
- `PROC-CARD-014` — PCI with drug-eluting stent
- `PROC-ORTH-032` — Knee arthroscopy
- `PROC-CARD-001` — ECG baseline

Ontology file at `data/ontology/synth_proc_v1.json`. Cross-referenced to CGHS where a matching package rate exists. **No mapping to CPT.** Any resemblance is coincidental.

### 8.4 Required disclaimer (in README)

> This environment uses only public-domain or openly-licensed medical coding systems: ICD-10-CM (CMS, public domain), LOINC (Regenstrief License), RxNorm (NLM, public domain), HCPCS Level II (CMS, public domain), CGHS package rates (Govt. of India, MoHFW public rate list). SYNTH-PROC-v1 is a synthetic procedure ontology created for this project; it is NOT AMA CPT, does NOT map to CPT, and must not be used for real billing. All patient data is synthetic. This project is for research, education, and RL training only.

---

## 9. Deployment — with fallbacks

### 9.1 Primary: HuggingFace Space
- Reuse Round 1 Dockerfile, update `openenv.yaml` → `name: medibill_env`, `app: medibill_env.server.app:app`.
- Validate with `openenv validate` (6 runtime + 8 local file checks).
- UID 1000 user, `0.0.0.0` bind, port 8000 (Round 1 config).

### 9.2 Fallback 1: local Docker
- Pre-build image on laptop.
- Same Dockerfile. Runs offline.
- Used if HF Space sleeps, rate-limits, or goes down mid-demo.

### 9.3 Fallback 2: prerecorded 90-second demo video
- Screen-captured full rollout on hard-drift task.
- Shows before/after on a specific claim.
- Last-resort if both HF and local fail.

Bring all three to onsite. Trigger in order: HF → local Docker → video.

---

## 10. Pitch (3 minutes)

### 10.1 Opener — verified regulatory reality (0:00–0:30)

> "IRDAI's Master Circular gives Indian hospitals one hour for pre-authorization
> and three hours for final discharge on every cashless claim. Miss the clock,
> and the overrun comes out of the insurer's shareholder fund. Last year
> ₹26,000 crore of claims were disallowed — up 19 percent. Thirteen percent
> of pre-auths still miss the window today."

### 10.2 The environment (0:30–1:15)

> "We built an OpenEnv where a language-model agent plays the medical coder
> under that clock. Three enterprise tools. ICD-10 diagnoses, CGHS rates,
> a synthetic procedure ontology. Graded by a deterministic rubric with
> six axes — final correctness, policy compliance, abstention quality,
> auditability, efficiency, and a gated drift bonus. And the hero mechanic:
> the insurance policy changes mid-episode. Agents that don't call
> `verify_policy_version` before submitting get graded against the new truth."

### 10.3 Training result (1:15–2:30)

> "We trained Qwen 2.5-3B. First the base model, then SFT on 40 filtered
> trajectories, then RFT iteration using the same grader — and GRPO on top
> where training converged. Here is the reward curve. [show W&B plot]
> And here is the before/after on a hard drift case. [show rollout]"

*(No specific numbers promised until measured.)*

### 10.4 The ask (2:30–3:00)

> "This fills three holes in OpenEnv: multi-app enterprise workflow (Scaler),
> mid-episode policy drift (Patronus), and programmatic-expert grading
> (Snorkel). It's legally clean — no CPT, no PHI — and India-contextual
> because that's where the regulatory clock is real. Thank you."

### 10.5 Q&A preparation

| Likely question | Answer |
|---|---|
| "Is this a real problem?" | IRDAI Annual Report FY24: ₹26k cr disallowed. LocalCircles Jan 2025: 36% rejection with invalid reasons. Master Circular mandated the clock. |
| "Why RL vs rules engine?" | Rules engines catch schema errors; an RL-trained agent learns the tradeoff between cleanup depth and timeout under dense reward shaping. |
| "Who signed off on the rubric?" | Expert-inspired — not reviewed by an SME. Weights are designed to be defensible and reproducible. We welcome SME critique post-hackathon. |
| "How do you prevent reward hacking?" | Drift bonus gated on final correctness. Auditability capped at 10%. Final_correctness and policy_compliance on disjoint cell sets. Exploit-test suite in repo. |
| "Why 3B not 7B?" | Single-A100 Colab throughput: 7B needs 20–38 h for 300 GRPO steps; 3B needs 6–10 h. 3B fits one session; demo ships. |
| "Why not CPT?" | AMA copyright. SYNTH-PROC is our synthetic ontology. Keeps the repo legally open. |
| "What if HF Space is down at demo time?" | Local Docker image is on my laptop. If that fails, a prerecorded 90-second rollout video plays. |
| "What's the single biggest risk to your result?" | Training convergence. Our shipping path is SFT+RFT, which reliably produces a reward delta. GRPO is a stretch bonus. |

### 10.6 Factual landmines (never claim)

- "IRDAI fines hospitals" — no, the penalty falls on **insurers**.
- Specific average processing time in minutes — IRDAI publishes % compliance, not means.
- "QR code mandate 1 April 2026" — unverified.
- "SME signed off on the rubric" — unless one actually does before onsite.
- Specific training result numbers (0.26→0.41→0.68 etc.) — until measured.

---

## 11. Timeline (Apr 20 → Apr 26)

| Day | Goal | Deliverable | Go/no-go |
|---|---|---|---|
| **Sun Apr 20** | Medical data gen + `policy_drift` corruption + tasks | `generate_clean_medical_records`, 3 task defs, SYNTH-PROC-v1 ontology | Data smoke test |
| **Mon Apr 21** | 3 tools + expert rubric + meta-actions | `ehr_query`, `insurance_lookup`, `coding_engine`, `verify_policy_version`, grader v2 | `openenv validate` |
| **Tue Apr 22** | 3-baseline grader validation + exploit tests | Score separation plot | ≥0.20 separation between scripted and random |
| **Wed Apr 23** | 50 trajectories (RFT filter) + HF Space deploy + local Docker | Training data + live URL + Docker image | HF Space live; Docker boots |
| **Thu Apr 24** | SFT on Colab Pro + first eval plot + prerecorded demo video | SFT adapter, baseline table, 90s video | SFT beats random+scripted |
| **Fri Apr 25 AM** | Pitch rehearsal + travel | Pitch deck v1, 5 rehearsals | Time ≤3:00 |
| **Sat Apr 25 onsite** | RFT iteration + optional GRPO | Second adapter, reward curve | Curve rises visibly |
| **Sun Apr 26 onsite** | Final eval + demo + pitch | 4-bar score chart + pitch | Trophy |

---

## 12. Risk register (updated)

| Risk | Severity | Mitigation |
|---|---|---|
| HF Space sleep/rate-limit mid-demo | High | Local Docker + prerecorded video |
| GRPO doesn't converge in 8h | Med | RFT is primary; GRPO is bonus |
| Grader looks broken (no score separation) | High | Validation gate on Tue blocks onward work until fixed |
| Colab gives T4 not A100 | Med | Qwen2.5-3B fits on T4 with LoRA; throughput halved but still ships SFT |
| Judges find healthcare "boring" | Med | IRDAI clock + regulatory penalty framing + policy-version drift hook |
| Reward hacking found in demo | High | Explicit exploit-test suite in §6 run before onsite |
| OpenEnv version bump | Low | Pin `openenv-core[core]==0.2.2` |
| User nervous in pitch | High | Daily 30-min rehearsal Mon–Fri; Q&A doc; verified fact sheet |
| Legal/IP risk from real medical codes | Med | SYNTH-PROC + ICD-10-CM + CGHS only; disclaimer in README |
| External verifier (Codex) rejects design | Low after v2 | Spec submitted for review before code; fix before building |

---

## 13. Go/no-go gates (hard stops)

1. **Sun end-of-day:** medical data generator produces 40 plausible patient records deterministically; smoke test passes.
2. **Tue end-of-day:** 3-baseline validation shows ≥0.20 score separation between `scripted_heuristic` and `random`. **If not: stop and fix grader before proceeding.**
3. **Thu end-of-day:** SFT adapter beats `random` + `scripted_heuristic` on 20 eval episodes. **If not: ship SFT+RFT only; skip GRPO entirely.**
4. **Fri noon:** HF Space live AND local Docker image built AND prerecorded demo recorded. **This is the minimum submission requirement; do not travel without all three.**

---

## 14. Sign-off

Spec v2 incorporates all 11 blockers/majors from external review dated 2026-04-20. Ready for second-round verification by Codex / ChatGPT / Gemini before implementation begins.

**Next action on approval:** write `server/data_generator.py::generate_clean_medical_records` + `data/ontology/synth_proc_v1.json` + `server/tasks.py::easy_cashless` as the Day 1 Sunday deliverable.
