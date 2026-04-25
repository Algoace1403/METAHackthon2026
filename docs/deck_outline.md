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
- FY24: ₹26,000 crore disallowed (+19% YoY)
- 13% of pre-auths still miss the window today

**SPEAKER NOTES:**
"In India, IRDAI gives hospitals one hour for pre-authorization and three hours for final discharge on every cashless claim. Miss the three-hour clock, and the overrun comes out of the insurer's shareholder fund. Last fiscal year, insurers disallowed twenty-six thousand crore rupees of claims — up nineteen percent. The bottleneck is a human coder racing a clock, and the policies keep changing on them."

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
- Scripted baseline: 1.00 on easy, **0.76 on drift** — that 0.24 gap is the signal

**SPEAKER NOTES:**
"Here is what makes the environment test reasoning instead of memorisation. On hard tasks, the policy changes mid-episode — but we do not tell the agent. There is no flag, no event, no hint. The only way the agent learns the rules changed is to call insurance_lookup again. Submissions are graded against the policy at submit time, not against what the agent believes. A scripted baseline drops from 1.0 on easy to 0.76 on the drift task. That 0.24 gap is what we train against."

---

## Slide 5 — Live measurements (2:00–2:30)  ★ HEADLINE SLIDE

**Title:**
Three baselines, one drift gap

**ON SLIDE (3-bar chart of hard_drift only, then 4 bullets):**
- *(Chart)* random 0.20 · no_op 0.16 · scripted 0.76 (n=20 seeds, hard_drift)
- Scripted on easy 1.00 → on hard_drift 0.76. Δ 0.24 is the **drift acceptance gap**
- Five exploit patterns explicitly neutralised; all five score ≤ no_op
- Demo video: scripted submits under stale policy and lands at 0.762

**SPEAKER NOTES:**
"On the hardest task, the policy changes silently mid-episode. The three baselines separate cleanly: random scores 0.20, no-op 0.16, and our tool-faithful scripted policy 0.76. The same scripted policy scores 1.00 on the no-drift easy task, so the missing 0.24 is the drift-acceptance gap. In the demo seed we show, drift fires at step 23, the scripted policy never calls insurance_lookup again, and it submits the remaining claims under stale v1.3 rules. The final score is 0.762. That is not recovery success; it is the cost of carrying a stale policy model into submit. Closing that behavioral gap is what our training pipeline is designed to target."

**BACKUP NOTES (do not say unless asked):**
- Reproducibility: 20-seed sweep, sd 0.011, range [0.752, 0.781]; per-seed CSV in `docs/baseline_reproducibility.csv`
- Reproducibility command: `python -m medibill.demo_runner --seed 44`
- 100-episode stress test on hard_drift: 0 crashes, 0 NaN

---

## Slide 6 — Scope + close (2:30–3:00)

**Title:**
Environment-first submission, three sub-prize fits

**ON SLIDE (5 bullets):**
- We submit the **environment + grader + baselines + drift mechanic**. SFT and RL are explicit follow-up work.
- Two of six axes — `abstention_quality` and `drift_bonus` — are RL-only targets (spec v3 §7.6)
- Code enforces every claim: disjoint partition asserted at import, 5 exploit tests, prompt-version handshake
- Sub-prize fits: Scaler AI Labs · Patronus AI · Snorkel AI
- Repo: github.com/Algoace1403/METAHackthon2026 · HF Space: *[fill before recording, or remove this line if not pushed]*

**SPEAKER NOTES:**
"We are submitting environment-first. What we are claiming today is the environment, the six-axis deterministic grader, the silent-drift mechanic, and a tool-faithful scripted baseline whose 0.24 gap on the drift task is the signal future training will close. We did not finish the SFT pass in time, and we are not relabelling the scripted bar as trained. Two axes — abstention quality and drift bonus — are RL-only targets in our pipeline, not SFT targets, and that scoping is in the spec. Everything else, the code enforces: disjoint partition asserted at import, five exploit tests, a prompt-version handshake. Three sub-prize fits: Scaler enterprise multi-app, Patronus schema drift, Snorkel programmatic rubric. Repo on screen. Thank you."

---

## Pre-recording / pre-pitch checklist

1. Replace `[fill before recording...]` placeholder on slide 6 with HF Space URL, or delete that bullet.
2. Slide 5 chart: 3 bars, vertical, clearly labeled `random / no_op / scripted`. Tick marks at 0.0, 0.5, 1.0. No animation.
3. Slide 3 table: keep cells short — every cell should fit on one line at 32-pt.
4. Slide 6 sub-prize bullet: keep just the names; no logos.
5. Test the deck once in presenter mode to confirm font legibility from the back of the room.
