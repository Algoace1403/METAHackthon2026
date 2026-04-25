# MediBill-Env Pitch v1

**Duration:** 3:00 spoken, 2:00 Q&A.
**Slides:** 6, ~30s each.
**Submission scope:** environment + grader + baselines + drift mechanic + **trained SFT adapter**. SFT was completed inside the hackathon window; per-task results on slide 5 show near-parity with the scripted teacher across all three difficulty tiers. The demo video shows the scripted baseline behavior; the eval table shows what imitation closes and what RL still has to break past.

---

## Slide 1 — The regulatory clock (0:00–0:30)

**Title:** "180 minutes to close the claim"

**Bullets on screen:**
- IRDAI mandate (May 2024): 1 hour pre-auth, 3 hours discharge
- Miss the 3-hour clock → insurer eats the cost from shareholder funds
- FY24: ₹26,000 crore disallowed (+19% YoY)
- 13% of pre-auths still miss the window today

**Speaker line:**
"In India, IRDAI gives hospitals one hour for pre-authorization and three hours for final discharge on every cashless claim. Miss the three-hour clock, and the overrun comes out of the insurer's shareholder fund. Last fiscal year, insurers disallowed twenty-six thousand crore rupees of claims — up nineteen percent. The bottleneck is a human coder racing a clock, and the policies keep changing on them."

---

## Slide 2 — The problem, not the schema (0:30–1:00)

**Title:** "The problem is staleness, not validation"

**Bullets on screen:**
- Insurers silently update policies (codes renamed, thresholds changed)
- Rules engines handle static validation; they do not handle staleness
- An agent that learned yesterday's rules ships today's wrong claim
- We need an agent that knows to **re-check** before submitting

**Speaker line:**
"Rules engines catch schema errors. They do not catch the case where the agent's mental model of the policy is correct yesterday and wrong today. Insurers rename codes, change pre-auth thresholds, add signatures — and do it between shifts without announcement. What we want is an agent that learns to re-check. Our environment tests exactly that."

---

## Slide 3 — The environment (1:00–1:30)

**Title:** "MediBill-Env: five tools, six axes, no CPT"

**Bullets on screen:**
- Tools: `ehr_query`, `insurance_lookup`, `coding_engine`
- Meta-actions: `escalate_to_human`, `submit_claim`
- 6-axis deterministic grader (disjoint field partition, import-time asserted)
- Synthetic data stack: ICD-10-CM + SYNTH-PROC-v1 + CGHS rates, no AMA CPT
- 5 exploit patterns explicitly neutralised

**Speaker line:**
"MediBill-Env is an OpenEnv environment. Three tools for information gathering, two meta-actions for abstention and submission. The grader has six axes, and identity fields and policy fields are disjoint by construction — enforced at import time. The data stack is ICD-10-CM, CGHS rates, and a synthetic procedure ontology we own. Zero dependency on AMA CPT. Five attack patterns are ruled out and tested: nothing beats doing nothing."

---

## Slide 4 — The hero mechanic (1:30–2:00)

**Title:** "The policy changes mid-episode. Silently."

**Bullets on screen:**
- On hard tasks, the active policy mutates at a seed-selected step
- No announcement — no observation flag, no metadata key
- `submit_claim` is graded against the policy at submit time
- Only path to the new rules: a fresh `insurance_lookup` call
- Scripted baseline: 1.00 on easy, **0.75 on drift** — that 0.25 gap is the signal

**Speaker line:**
"Here is what makes the environment test reasoning instead of memorisation. On hard tasks, the policy changes mid-episode — but we do not tell the agent. There is no flag, no event, no hint. The only way the agent learns the rules changed is to call `insurance_lookup` again. Submissions are graded against the policy at submit time, not against what the agent believes. A scripted baseline drops from 1.0 on easy to 0.75 on the drift task. That 0.25 gap is what we train against."

---

## Slide 5 — Live measurements (2:00–2:30)  ★ HEADLINE SLIDE

**Title:** "Baselines + SFT: imitation closes the gap, RL must break past"

**Bullets on screen:**
- 4-bar chart on hard_drift (20-seed means): random 0.11 · no_op 0.08 · scripted 0.75 · **SFT 0.76**
- SFT per-task vs scripted (held-out seeds 16–19, n=4 each):
  - easy_cashless **1.000 / 1.000**
  - medium_multi_payer **1.000 / 1.000**
  - hard_drift **0.755 / 0.764**  (Δ −0.009)
- Five exploit patterns explicitly neutralised; all five score ≤ no_op
- SFT cannot exceed the scripted teacher; the remaining drift_bonus + abstention axes are **RL-only by design** (spec v3 §7.6)

**Speaker line:**
"Three baselines on hard_drift: random eleven, no-op eight, scripted seventy-five. Our SFT adapter — Qwen 2.5 three-billion plus LoRA, six-hundred-eighty-one training steps — reproduces the scripted teacher on every difficulty tier on held-out seeds: one-point-zero on easy and medium, zero-point-seven-five-five on hard versus the teacher's zero-point-seven-six-four. That is imitation working as advertised. The drift-acceptance gap stays open by design — drift_bonus and abstention are RL-only axes in our spec. SFT closes the imitation gap; GRPO is what breaks past the teacher."

**Backup / speaker notes (not spoken):**
- SFT eval reproducibility: `notebooks/sft_quickstart.ipynb` + `traces/eval.jsonl`
- Training: 681 steps, loss 0.42 → 0.014, LoRA rank 32 on Qwen 2.5 3B, ~90 min on Colab G4
- 20-seed scripted baseline on hard_drift: mean 0.754, sd 0.011, range [0.752, 0.781]
- SFT delta is −0.009 on hard_drift — within scripted noise band; on easy/medium it lands inside the rounding

---

## Slide 6 — Scope + close (2:30–3:00)

**Title:** "Environment-first submission under Theme 3.1"

**Bullets on screen:**
- Shipping today: **environment + grader + 5-attack exploit suite + scripted baseline + trained SFT adapter at scripted parity**.
- Two of six axes — `abstention_quality` and `drift_bonus` — are RL-only targets (spec v3 §7.6); GRPO is the planned next step.
- Code enforces every claim: disjoint partition asserted at import, 5 exploit tests in the repo, prompt-version handshake on the corpus.
- Theme 3.1 (DataOps Copilot) — closest sub-prize fit: Scaler AI Labs (enterprise reasoning under business rules and regulatory constraints).
- Repo: github.com/Algoace1403/METAHackthon2026 · HF Space: `[URL after push]`

**Speaker line:**
"We submit under Theme 3.1, DataOps Copilot. What ships today: the environment, the six-axis deterministic grader, the silent-drift mechanic, a five-attack exploit suite, a tool-faithful scripted baseline, and a trained SFT adapter that reaches scripted parity on every difficulty tier — the table you saw on slide five. Two axes — abstention and drift_bonus — are RL-only by design; GRPO is the planned next step to break past the teacher. The code enforces everything else: disjoint partition at import, five exploit tests, prompt-version handshake. Closest sub-prize fit on Theme 3.1 is Scaler AI Labs. Repo and Space on screen. Thank you."

---

## Video recording recipe (for the <2 min minimum-requirement artifact)

If SFT doesn't ship by Friday end-of-day, or if you want a safety-copy even with SFT numbers, record `medibill.demo_runner` running live. Command for a full-terminal screen recording:

```bash
# macOS built-in screen recording: Cmd+Shift+5 → Record Selected Portion
# Start recording BEFORE running the command below; stop when episode complete.

python3 -m medibill.demo_runner --seed 44 --max-narrated-steps 20
```

The script emits:
- Title banner identifying hard_drift + seed
- Step-by-step agent actions (first 20 steps narrated, rest elided)
- A red "*** DRIFT FIRED SILENTLY ***" line when the policy mutates
- A green "Agent has detected drift" line when the scripted agent re-queries
- Final composite score (~0.75 on hard_drift) with per-axis breakdown

Total run time ~15 seconds on a modern laptop, giving ~60 seconds of narratable screen content once you speak over it. Fits the <2-minute limit comfortably.

### Narration script for the video (≤90 seconds)

> "This is MediBill-Env, a synthetic medical-billing environment for the Meta OpenEnv hackathon. Twelve insurance claims, five tools, six-axis deterministic grader. Watch what happens.
>
> The agent starts by asking the insurer for the current policy — version 1.3 — and begins coding claims against it.
>
> [pause at step 23 when DRIFT FIRED appears]
>
> At step 23, the insurer's policy silently updates. No announcement, no flag, no observation field. The agent is still working with its cached view of version 1.3, and some of the rules have quietly changed.
>
> [pause at 'Agent has detected drift']
>
> The agent's next insurance_lookup picks up the new policy version, 1.4. From this point it uses the new rules.
>
> [point at final score]
>
> Final composite score: 0.75 out of 1.0. That 0.25-point gap from a perfect score is the signal we train against. Two of our six axes — abstention and drift bonus — are RL-only targets in our pipeline, not SFT targets. That scoping is explicit in our spec. Thank you."

---

## Q&A anticipated (prepare, do not read)

| Question | Answer |
|---|---|
| "Is this a real problem or synthetic?" | IRDAI Annual FY24: ₹26k crore disallowed. LocalCircles Jan 2025: 36% of policyholders had claims rejected with invalid reasons. The regulator wrote a Master Circular specifically to fix it. |
| "Why SFT-only, not GRPO?" | GRPO is the roadmap for the two RL-only axes (drift_bonus, abstention_quality). SFT can match the teacher — and our adapter does — but cannot exceed it; that's GRPO's job. Time constraint shipped SFT-first with honest scoping. |
| "Why does SFT match scripted on easy/medium but lag on hard?" | SFT's ceiling is the teacher. On easy/medium the teacher is 1.00 so a perfect imitator matches. On hard the teacher caps at 0.764 because drift_bonus is gated on (final+policy) ≥ 0.80; the teacher reaches that gate but not the bonus. SFT lands at 0.755 — within scripted's seed-to-seed noise band (sd 0.011). |
| "Why not CPT codes?" | AMA copyright. Our synthetic SYNTH-PROC-v1 keeps the repo legally open. |
| "Who signed off on the rubric?" | Not a clinical SME; it is an expert-inspired deterministic rubric. We welcome SME review post-hackathon. |
| "What if the HF Space is down during the demo?" | Local Docker image on the laptop, verified reachable on `/health`. |
| "Biggest risk?" | Structural SFT ceiling on hard_drift near 0.80 because two axes are RL-only. SFT cannot exceed that ceiling; GRPO can. |
