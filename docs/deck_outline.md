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
Silent policy drift

**ON SLIDE (5 bullets):**
- On hard tasks, the active policy mutates at a seed-selected step
- No announcement — no observation flag, no metadata key
- `submit_claim` is graded against the policy at submit time
- Only path to the new rules: a fresh `insurance_lookup` call
- Scripted baseline: 1.00 on easy, **0.75 on drift** — that 0.25 gap is the signal

**SPEAKER NOTES:**
"Here is what makes the environment test reasoning instead of memorisation. On hard tasks, the policy changes mid-episode — but we do not tell the agent. There is no flag, no event, no hint. The only way the agent learns the rules changed is to call insurance_lookup again. Submissions are graded against the policy at submit time, not against what the agent believes. A scripted baseline drops from 1.0 on easy to 0.75 on the drift task. That 0.25 gap is what we train against."

---

## Slide 5 — Live measurements (2:00–2:30)  ★ HEADLINE SLIDE

**Title:**
Baselines + SFT: imitation closes the gap, RL must break past

**ON SLIDE (1 chart + 1 table + 2 bullets):**

*(Chart — 4 bars on hard_drift)*
| random | no_op | scripted | SFT |
|---|---|---|---|
| 0.11 | 0.08 | 0.76 | 0.76 |

*(Table — SFT vs scripted per task, n=10 held-out seeds, 95% CI)*
| task | SFT | scripted | Δ |
|---|---|---|---|
| easy_cashless | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | +0.0000 |
| medium_multi_payer | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | +0.0000 |
| hard_drift | **0.7573 ± 0.0040** | 0.7611 ± 0.0049 | −0.0037 ✓ |

✓ = inside both 95% CIs

- 5 exploit patterns explicitly neutralised; all five score ≤ no_op
- `drift_bonus` and `abstention_quality` are **RL-only axes** by design (spec v3 §7.6) — GRPO target

**SPEAKER NOTES (~26s spoken — leaves Q&A margin):**
"Four bars on hard_drift: random eleven, no-op eight, scripted seventy-six, SFT seventy-six. The table is n=ten held-out seeds with ninety-five-percent CIs: SFT matches scripted at one-point-zero on easy and medium with zero variance, and zero-point-seven-five-seven plus-or-minus zero-point-zero-zero-four on hard versus scripted's zero-point-seven-six-one. The three-thousandths gap is inside both noise bands — statistically indistinguishable. Imitation reaches the teacher. The remaining gap to a perfect score lives on two axes the spec designates RL-only, and GRPO is the planned next step."

**SAVE THE SEED-44 STORY FOR Q&A** — the table speaks for itself; don't burn slide-5 time on the demo walkthrough.

**BACKUP NOTES (do not say unless asked):**
- SFT training: 681 steps, loss 0.42 → 0.014, LoRA rank 32 on Qwen 2.5 3B, ~90 min Colab G4
- SFT eval: n=10 held-out seeds (16–25) × 3 tasks = 30 trajectories, zero parse failures
- 95% CI = 1.96·sd/√n; per-seed raw scores saved at `/results/sft_eval_n10.json`
- SFT Δ on hard_drift (−0.0037) is within both SFT-CI (±0.0040) and scripted-CI (±0.0049)
- Verified via Codex's protocol: sha256 byte-match + fresh subprocess × 2
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
