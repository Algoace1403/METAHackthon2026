# Round 2 Spec — MediBill DataOps

**Hackathon:** Meta × Scaler OpenEnv Hackathon, Round 2
**Onsite:** 25–26 April 2026, Bangalore
**Theme:** #3.1 Professional World Modeling
**Sub-prize targets:** Snorkel AI (primary) + Patronus AI + Scaler AI Labs
**Date locked:** 2026-04-19

---

## 1. One-line pitch

> "An AI medical coder that learns to resolve hospital billing incidents across 5 enterprise tools, graded by simulated senior-coder experts, under adversarial insurance-policy drift."

---

## 2. Problem statement (for judges)

Every Indian hospital loses crores to **billing errors caused by schema drift**:
insurance providers rename codes overnight, policies update mid-shift, columns
get renamed when EHR systems upgrade. Human medical coders spend 4–6 hours
reconciling a single complex claim.

We build an **OpenEnv environment** where an LLM agent:
1. Reads messy patient records across 5 enterprise systems.
2. Applies correct billing codes under the **current** (possibly changed) insurance policy.
3. Escalates genuinely ambiguous cases instead of hallucinating.
4. Operates under a fixed cost budget.
5. Is graded by a **simulated senior medical coder** whose rubric follows real hospital audit patterns.

---

## 3. What we keep from Round 1 (≈ 70% reuse)

| Component | Status | Reason |
|---|---|---|
| `models.py` (Action/Obs/State Pydantic) | **Keep + extend** | Issue-first design is gold; add 5 tool-action variants |
| `server/environment.py` — reset/step, budget, delta reward | **Keep** | Proven; 228 tests cover it |
| `server/grader.py` — alignment, cell matching, utility probes | **Keep** | Entity-ID alignment with similarity fallback already handles medical |
| `server/data_generator.py` — DataCorruptor pipeline | **Keep + add 2 new corruption types** | `insurance_id_mismatch` and `impossible_date` already exist; add `schema_drift` and `policy_change` |
| `scripts/train_grpo.py` + `reward_functions.py` | **Keep** | GRPO pipeline already wired; just add expert-rubric reward fn |
| `Dockerfile` + `openenv.yaml` | **Keep** | HF-Space-ready, UID 1000, port 8000 |
| 20 test files (228 tests) | **Keep + add medical tests** | Regression safety net |

---

## 4. What we add (the new ~30%)

### 4.1 Five enterprise "tools" (new action categories)

These are **new action_type values** that wrap our existing atomic actions into
enterprise-flavoured workflows. The LLM sees them as distinct tools; internally
they call the existing 10 handlers.

| # | Tool name | What it does | Built on top of |
|---|---|---|---|
| 1 | `ehr_query` | Pull a patient record by ID | Read-only observation slice |
| 2 | `insurance_lookup` | Check current policy rules for a provider | Read-only; exposes drifted rules |
| 3 | `coding_engine` | Apply/fix a medical billing code | `fix_value` + `standardize_format` |
| 4 | `bill_validator` | Run compliance checks on a constructed bill | Returns issue list (read-only) |
| 5 | `audit_log` | Record action rationale for compliance | No state change; earns transparency bonus |

Existing 10 actions still exist as "primitive" operations; the tools are
higher-level and earn additional efficiency/transparency credit.

### 4.2 Schema drift

Two styles, natural difficulty curve:

| Level | Drift style | Example |
|---|---|---|
| **Easy** | None (rules stable) | Baseline billing task |
| **Medium** | Drift **between** episodes | Provider X's prefix changed from "AE" to "AET" between tasks |
| **Hard** | Drift **during** episode | Mid-episode, `dx_code` column renamed to `diagnosis_code`; new pre-auth threshold applied to remaining rows |

Implemented as a new corruption type `policy_change` in `data_generator.py`
plus runtime state-mutation in `environment.py`.

### 4.3 Simulated senior coder rubric (Snorkel angle)

A **`SeniorCoderRubric`** class that grades each trajectory like a real hospital
auditor. Not just final state — looks at **how** the agent got there.

Rubric axes (each 0.0–1.0):

| Axis | Weight | Signal |
|---|---|---|
| `code_accuracy` | 35% | Correct ICD-10/CPT codes vs ground truth |
| `policy_compliance` | 20% | Bill matches **current** (drifted) policy, not stale rules |
| `drift_recovery` | 15% | % of drifted cells correctly adapted |
| `abstention_quality` | 10% | Escalate ambiguous, don't escalate clear-cut |
| `efficiency` | 10% | Budget used wisely |
| `auditability` | 10% | `audit_log` called with meaningful rationale |

The rubric is **programmatic** (no real LLM grader) — deterministic, cheap,
reproducible. This is exactly what Snorkel calls "programmatic preferences" in
their 2026 "year of environments" announcement.

### 4.4 Medical domain skin

`data_generator.py` gets a new `generate_clean_medical_records()` function
producing realistic fake Indian patient data:

- Names (Indian + international mix, with Hindi transliteration variants for
  duplicate detection)
- ICD-10 diagnosis codes (sampled from real code ranges)
- CPT procedure codes
- Insurance providers: CGHS, ESIC, Star Health, HDFC ERGO, Aditya Birla, private
- Hospital IDs, ward codes, pre-auth numbers
- Amounts in INR

Task names:
- `easy_billing` — 8 patient bills, format fixes only, no drift
- `medium_claims` — 20 claims across 2 insurers, drift between tasks
- `hard_reconciliation` — 40 claims, mid-episode policy change, 3 genuinely ambiguous cases, 2 false-positive duplicate claims for same patient from different hospitals

---

## 5. Reward design (extends Round 1's delta system)

Per-step (during episode):
```
reward = (rubric_score_now - rubric_score_prev) - 0.005 (step cost)
```

Terminal (at `mark_complete`):
```
reward = SeniorCoderRubric.score(final_state, action_trace, policy_at_end)
```

Dense signals guaranteed (per Agent 3 feasibility research):
- Every action affects at least one rubric axis
- Drift-recovery axis gives immediate positive reward when a drifted cell is adapted
- Abstention axis gives +0.3 for correct escalate, −0.5 for hallucinated fix on ambiguous

Four GRPO reward functions (extends Round 1's 4 → 5):
1. `reward_valid_action` (format compliance)
2. `reward_episode_score` (terminal rubric)
3. `reward_efficiency` (budget)
4. `reward_no_destruction` (no bad merges/deletes)
5. **NEW**: `reward_drift_recovery` (per-step signal for schema adaptation)

---

## 6. Self-improvement / training strategy

**Stage 1 — SFT bootstrap** (pre-onsite, Apr 24 on Colab Pro)
- Generate 200 expert trajectories using GPT-4 via API
- Qwen2.5-7B QLoRA via Unsloth
- ~30–60 min A100 wall-clock
- Safety net: even if GRPO blows up onsite, SFT alone gives us ~60% of the demo signal

**Stage 2 — GRPO** (onsite Apr 25, 12–24 h)
- TRL GRPO + vLLM colocate mode (avoids server-mode bug #4543)
- Init from SFT adapter
- Per-step dense rewards from the 5 reward functions
- Log to W&B every 10 steps

**Curriculum**
- 40% easy, 40% medium, 20% hard in the training mix
- Drift frequency scales: easy 0%, medium 30%, hard 80% of episodes contain drift

---

## 7. HF Space deployment (minimum requirement)

Reuse Round 1 Dockerfile + `openenv.yaml`. Update metadata:
```yaml
spec_version: 1
name: medibill_env
type: space
runtime: fastapi
app: medibill_env.server.app:app
port: 8000
```

Validate with `openenv validate` before HF push. Expected to pass all 6 runtime
endpoint checks and 8 local file checks (same infra as Round 1).

---

## 8. 3-minute pitch skeleton

**Seconds 0–30 — The pain (storytelling 30%)**
> "Meet Rajesh. Heart surgery. ₹8 lakh bill. But overnight his insurance company
> renamed 4 billing codes and updated pre-auth rules. Every hospital in India
> loses crores to this chaos. And the senior medical coders who fix it? They're
> drowning."

**Seconds 30–75 — The environment (innovation 40%)**
> "We built an OpenEnv environment where an LLM plays a medical coder at an
> Indian hospital. 5 enterprise tools, 3 difficulty levels, mid-episode policy
> drift. Graded by a programmatic senior-coder rubric that scores not just
> accuracy — but compliance, drift recovery, and abstention quality."

**Seconds 75–150 — The training (reward improvement 20%)**
> "We trained Qwen-2.5-7B with GRPO. Watch the reward curve. [show plot]
> Baseline: 0.26. SFT: 0.41. SFT + GRPO: 0.68. That's senior-coder parity on
> hard cases. [show rollout before/after demo]"

**Seconds 150–180 — The ask (ecosystem fit)**
> "This fills three holes in OpenEnv: enterprise multi-app workflows (Scaler),
> schema drift (Patronus), and programmatic-expert grading (Snorkel). Rajesh's
> bill is now correct. And the system can retrain itself every time a policy
> updates. Thank you."

**Q&A prep (2 min):**
- "Why not real ICD-10 tables?" → Our simulated tables are synthetic but follow
  real code-range distributions; we can swap in real tables on HF Space.
- "How do you prevent reward hacking?" → Drift recovery is gated on final
  correctness; abstention is penalised if clear-cut; budget caps gaming.
- "Why GRPO over DPO?" → Multi-step trajectories with dense per-step signal fit
  GRPO; DPO would need offline preference pairs we don't have.
- "What if training doesn't converge in 12h?" → SFT adapter alone demonstrates
  a ~60% improvement over base model; full GRPO is the stretch goal.

---

## 9. Timeline (Apr 19 → Apr 26)

| Day | Goal | Deliverable |
|---|---|---|
| **Sat Apr 19 (tonight)** | Spec lock | This document |
| **Sun Apr 20** | Medical data gen + task defs | `generate_clean_medical_records`, 3 new tasks |
| **Mon Apr 21** | Tool actions + expert rubric | 5 tools wired, `SeniorCoderRubric` class |
| **Tue Apr 22** | Schema drift + dense reward validation | 50 random-rollout variance test |
| **Wed Apr 23** | 200 expert trajectories + HF Space deploy | Training data + live env URL |
| **Thu Apr 24** | SFT on Colab + first eval plot | SFT adapter, baseline numbers |
| **Fri Apr 25 AM** | Pitch rehearsal + travel | Pitch deck v1 |
| **Sat Apr 25 onsite** | GRPO training run | Reward curve |
| **Sun Apr 26 onsite** | Eval + demo + pitch | Trophy |

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Colab Pro gives T4 not A100 | Budget for Modal/RunPod as fallback (~$20) |
| TRL server-mode bug corrupts multi-turn rollouts | Use colocate mode (docs confirmed working) |
| Sparse reward = flat curve | Per-step drift-recovery reward guarantees non-zero variance |
| SFT overfits to GPT-4 style | Mix in 20% rule-based expert trajectories |
| Judges bored by "data cleaning" | Medical reframe + Rajesh story; show live drift recovery |
| User nervous in pitch | Daily 30-min rehearsal from Mon; pre-written Q&A |
| OpenEnv version bumps | Pin `openenv-core[core]==0.2.2` for submission |

---

## 11. Go/no-go gates

- **Sun end-of-day:** medical data generator passes basic smoke test (generates
  40 plausible patient records deterministically).
- **Tue end-of-day:** 50 random-policy rollouts show reward variance > 0.05.
  **If not: stop and fix dense-reward design before proceeding.**
- **Thu end-of-day:** SFT adapter beats random baseline on 20 eval episodes.
  **If not: skip GRPO, ship SFT-only demo.**
- **Fri noon:** HF Space live, `openenv validate` passes. **Hard stop — this
  is the minimum submission requirement.**
