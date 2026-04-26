# MediBill-Env — Slide Deck Outline (Keynote / Slides paste-ready)

Use this when you build the actual deck. Every block under "ON SLIDE" is one slide; copy the bullets exactly. Speaker notes go into the slide's "Notes" pane in Keynote/Slides — not on the visible slide.

**Suggested layout:** 16:9 widescreen, 32-pt body, no animations, dark text on white. Logo bottom-right of every slide. One chart on slide 5; everything else is text only.

**Total spoken time target:** 3:00. Slide-by-slide budget below.

---

## Slide 1 — The regulatory clock (0:00–0:30)

**Title:**
180 minutes to close the claim.

**ON SLIDE (4 bullets):**
- IRDAI mandate (May 2024): 1 hour pre-auth, 3 hours discharge
- Miss the 3-hour clock → insurer eats the cost from shareholder funds
- FY24: ~₹26,000 cr health-claim disallowed¹
- ~13% of pre-auths still miss the window²

*¹ IRDAI Annual Report FY24 — Health Insurance section. ² LocalCircles Health Insurance Survey, Jan 2025.*

**SPEAKER NOTES:**
"In India, IRDAI gives hospitals one hour for pre-authorization and three hours for final discharge on every cashless claim. Miss the three-hour clock, and the overrun comes out of the insurer's shareholder fund. Industry estimates put FY24 disallowed health-claim value around twenty-six thousand crore rupees — IRDAI Annual Report — and roughly thirteen percent of pre-auths still miss the one-hour window per the LocalCircles January survey. The bottleneck is a human coder racing a clock, and the policies keep changing on them."

---

## Slide 2 — The problem, not the schema (0:30–1:00)

**Title:**
Why agents fail here

**ON SLIDE (3 bullets):**
- Rules engines handle static schema validation
- They do not handle **staleness** — yesterday's correct rule, today wrong
- Agents that imitate one month's trajectories fail quietly the next month

**SPEAKER NOTES:**
"Most agent benchmarks check whether the agent can fill a form correctly. That is schema validation, and rules engines already do it. The real failure mode in this domain is staleness — the policy changed, the agent did not notice, the claim is wrong. An agent that learned by imitating last month's expert trajectories will reproduce last month's rules. We want an agent that knows to re-check before submitting."

---

## Slide 3 — The environment (1:00–1:30)

**Title:**
MediBill-Env: 5 tools, 3 task tiers, 6-axis grader

**ON SLIDE (one table):**
| Tool | Purpose |
|---|---|
| `ehr_query` | Read patient record |
| `insurance_lookup` | Fetch active policy rules |
| `coding_engine` | Write a policy-sensitive field |
| `escalate_to_human` | Calibrated abstention |
| `submit_claim` | Lock claim for grading |

**Plus on slide:** Tasks: easy_cashless · medium_multi_payer · hard_drift

**SPEAKER NOTES:**
"The agent has five tools: query the patient record, look up the insurer's active policy, write fields, escalate when uncertain, and submit. Three task tiers — easy, medium, and hard, where the policy mutates mid-episode. The grader has six axes with a disjoint field partition asserted at import time, so identity correctness and policy compliance never overlap."

---

## Slide 4 — The hero mechanic (1:30–2:00)

**Title:**
Silent multi-field policy drift

**ON SLIDE (6 bullets):**
- Active policy mutates **3–7 fields** at a seed-randomized step (pre-auth thresholds, signatures, narrative requirements, discharge attachment)
- **No announcement** — no observation flag, no metadata key, no event
- `submit_claim` is graded against the policy *at submit time*
- Only path to the new rules: a fresh `insurance_lookup` call after the (unknown) drift step
- 12 claim types × 3 tiers × randomized drift step = **~12k+ unique trajectories**
- Scripted baseline: 1.00 on easy, **0.7611 on drift** — that 0.24 gap is the signal

**SPEAKER NOTES:**
"On hard_drift tasks the active policy mutates mid-episode across three to seven fields — pre-auth thresholds, required signatures, narrative requirements, discharge attachment rules. Multi-field mutation, not a boolean. No announcement, no flag, no event. The only path to the new rules is a fresh insurance_lookup after the unknown drift step. Submissions are graded against the policy at submit time. Twelve claim types, three tiers, seed-randomized drift = over twelve thousand unique trajectories. Scripted baseline drops from one-zero on easy to zero-seven-six on drift. That zero-two-four gap is the trainable signal."

---

## Slide 5 — Live measurements (2:00–2:30)  ★ HEADLINE SLIDE

**Title:**
From base 0.00 to teacher parity. RL saturates — that is calibration data.

**ON SLIDE (1 chart + 2 tables + 2 bullets):**

*(Chart — 5 bars on hard_drift)*
| base Qwen | random | no_op | scripted | SFT |
|---|---|---|---|---|
| **0.00** | 0.11 | 0.08 | 0.76 | **0.76** |

*(Table A — Base → SFT lift, n=5 held-out seeds, the IMPROVEMENT story)*
| task | base Qwen | SFT | **lift** |
|---|---|---|---|
| easy_cashless | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | **+1.000** |
| medium_multi_payer | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | **+1.000** |
| hard_drift | 0.0000 ± 0.0000 | 0.7573 ± 0.0040 | **+0.7573** |
| **avg** | **0.0000** | **0.9191** | **+0.9191** |

*(Table B — SFT vs scripted, n=10 held-out seeds, the PARITY story)*
| task | SFT | scripted | Δ |
|---|---|---|---|
| easy_cashless | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | +0.0000 |
| medium_multi_payer | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | +0.0000 |
| hard_drift | **0.7573 ± 0.0040** | 0.7611 ± 0.0049 | −0.0037 ✓ |

✓ = inside both 95% CIs

- 5 exploit patterns explicitly neutralised; all five score ≤ no_op
- GRPO Δ = ±0.0002 → **rewards saturated** by SFT (calibration finding, see backup)

**SPEAKER NOTES (~28s spoken):**
"Five bars on hard_drift: base Qwen at zero, random at eleven, no-op at eight, scripted at seventy-six, SFT at seventy-six. The first column matters. Untrained, the 3B model produces valid JSON — zero parse failures across fifteen episodes — but scores zero on every task. After SFT: easy and medium hit one-point-zero with zero variance, hard at zero-point-seven-five-seven, three thousandths inside the scripted teacher's confidence band. We then ran GRPO and observed reward saturation — delta two ten-thousandths, gradient norms ten-to-the-minus-seven. SFT already extracts what those rewards signal. That is calibration data about the env's tool-space depth, not training failure."

**SAVE THE SEED-44 STORY FOR Q&A** — the tables speak for themselves; don't burn slide-5 time on the demo walkthrough.

**BACKUP NOTES (do not say unless asked):**
- Base eval: n=5 held-out seeds (16–20) × 3 tasks = 15 episodes, all 0.0000, parse_failures = 0, 12.1 min wall time, saved at `/results/base_eval_n5.json`
- SFT training: 681 steps, loss 0.42 → 0.014, LoRA rank 32 on Qwen 2.5 3B, ~90 min Colab G4
- SFT eval: n=10 held-out seeds (16–25) × 3 tasks = 30 trajectories, zero parse failures, saved at `/results/sft_eval_n10.json`
- 95% CI = 1.96·sd/√n; SFT Δ on hard_drift (−0.0037) is within both SFT-CI (±0.0040) and scripted-CI (±0.0049)
- Verified via Codex's protocol: sha256 byte-match + fresh subprocess × 2
- GRPO finding (`docs/reward_calibration.md` §5): 5 reward functions saturated at step 1 because SFT-from-scripted already satisfies all of them (valid_json ✓, in-schema ✓, no_oscillation ✓, no_repeated_tool ✓, submit_with_coding ✓). Δ_score = ±0.0002, grad_norm ~1e-7. The roadmap is reward-engineered task tiers, not more RL.
- Reproducibility commands: `python -m medibill.validate_grader --task all`, `python -m medibill.demo_runner --seed 44`

---

## Slide 6 — Scope + close (2:30–3:00)

**Title:**
Environment-first submission under Theme 3.1

**ON SLIDE (4 bullets):**
- Shipping today: **environment + grader + 5-attack exploit suite + scripted baseline + trained SFT adapter at scripted parity** (table on slide 5)
- Two of six axes — `abstention_quality` and `drift_bonus` — are RL-only targets (spec v3 §7.6); GRPO is the planned next step
- Code enforces every claim: disjoint partition asserted at import, 5 exploit tests, prompt-version handshake on the corpus
- Theme 3.1 — DataOps Copilot. Enterprise reasoning under shifting business rules.
- Repo: github.com/Algoace1403/METAHackthon2026 · HF Space: *(fill once pushed)*

**SPEAKER NOTES (~26s):**
"We submit under Theme 3.1, DataOps Copilot. Shipping today: the environment, six-axis deterministic grader, silent drift mechanic, five-attack exploit suite, scripted baseline, and a trained SFT adapter that hits scripted parity on every difficulty tier — table on slide five. Two axes — abstention and drift_bonus — are RL-only by design; GRPO is the planned next step. Disjoint partition at import, five exploit tests, prompt-version handshake. Repo and Space on screen. Thank you."

**DROPPED FROM SPOKEN PITCH:** sub-prize naming. If a Scaler / Snorkel / Patronus judge asks about fit, name them. Do not pitch sub-prizes inside their own room.

---

## Pre-recording / pre-pitch checklist

1. Replace `[fill before recording...]` placeholder on slide 6 with HF Space URL, or delete that bullet.
2. Slide 5 chart: 3 bars, vertical, clearly labeled `random / no_op / scripted`. Tick marks at 0.0, 0.5, 1.0. No animation.
3. Slide 3 table: keep cells short — every cell should fit on one line at 32-pt.
4. Slide 6 sub-prize bullet: keep just the names; no logos.
5. Test the deck once in presenter mode to confirm font legibility from the back of the room.
