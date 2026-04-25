# MediBill-Env Pitch v1

**Duration:** 3:00 spoken, 2:00 Q&A.
**Slides:** 6, ~30s each.
**Submission scope:** environment + grader + baselines + drift mechanic. SFT was not run inside the hackathon compute window; do **not** present trained-model bars on any slide. The demo video shows the scripted baseline *failing to recover from drift*, which is the entire signal.

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

**Title:** "MediBill-Env: three tools, six axes, no CPT"

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
- Scripted baseline: 1.00 on easy, **0.76 on drift** — that 0.24 gap is the signal

**Speaker line:**
"Here is what makes the environment test reasoning instead of memorisation. On hard tasks, the policy changes mid-episode — but we do not tell the agent. There is no flag, no event, no hint. The only way the agent learns the rules changed is to call `insurance_lookup` again. Submissions are graded against the policy at submit time, not against what the agent believes. A scripted baseline drops from 1.0 on easy to 0.76 on the drift task. That 0.24 gap is what we train against."

---

## Slide 5 — Live measurements (2:00–2:30)

**Title:** "Three baselines, one drift gap"

**Bullets on screen:**
- 3-bar chart: random 0.20 · no_op 0.16 · scripted 0.76 (20-seed means on hard_drift)
- Scripted on easy 1.00 → on hard_drift 0.76. Δ 0.24 is the **drift acceptance gap**
- Five exploit patterns explicitly neutralised; all five score ≤ no_op
- The video below shows the scripted baseline submitting under stale policy *because it does not re-query* — score lands at 0.762, the cost of acceptance
- Closing that 0.24 gap is exactly what an RL-trained policy would learn

**Speaker line:**
"On the hardest task, the policy changes silently mid-episode. The three baselines separate cleanly: random scores 0.20, no-op 0.16, and our tool-faithful scripted policy 0.76. The same scripted policy scores 1.00 on the no-drift easy task, so the missing 0.24 is the drift-acceptance gap. In the demo seed we show, drift fires at step 23, the scripted policy never calls `insurance_lookup` again, and it submits the remaining claims under stale v1.3 rules. The final score is 0.762. That is not recovery success; it is the cost of carrying a stale policy model into submit. Closing that behavioral gap is what our training pipeline is designed to target."

**Backup / speaker notes (not spoken):**
- Reproducibility command: `python -m medibill.demo_runner --seed 44`
- 8-seed sweep on hard_drift: scripted in 0.752–0.781 band, mean 0.762
- Five exploit patterns explicitly neutralised; all five score ≤ no_op

---

## Slide 6 — Scope + close (2:30–3:00)

**Title:** "Environment-first submission, three sub-prize targets"

**Bullets on screen:**
- We submit the **environment + grader + baselines + drift mechanic**. SFT and RL are explicit follow-up work, not claims today.
- Two of six axes — `abstention_quality` and `drift_bonus` — are RL-only targets (spec v3 §7.6). We did not relabel scripted as trained.
- The code enforces every claim: disjoint partition asserted at import time, five exploit tests in the repo, prompt-version handshake on the corpus.
- Three sub-prize hits: Scaler AI Labs (enterprise multi-app), Patronus AI (schema/policy drift), Snorkel AI (programmatic expert rubric).
- Repo: github.com/Algoace1403/METAHackthon2026 · HF Space: `[URL]`

**Speaker line:**
"We are submitting environment-first. What we are claiming today is the environment, the six-axis deterministic grader, the silent-drift mechanic, and a tool-faithful scripted baseline whose 0.24 gap on the drift task is the signal future training will close. We did not finish the SFT pass in time, and we are not relabelling the scripted bar as trained. Two axes — abstention quality and drift bonus — are RL-only targets in our pipeline, not SFT targets, and that scoping is in the spec. Everything else, the code enforces: disjoint partition asserted at import, five exploit tests, a prompt-version handshake. Three sub-prize fits: Scaler enterprise multi-app, Patronus schema drift, Snorkel programmatic rubric. Repo on screen. Thank you."

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
- Final composite score (~0.76 on hard_drift) with per-axis breakdown

Total run time ~15 seconds on a modern laptop, giving ~60 seconds of narratable screen content once you speak over it. Fits the <2-minute limit comfortably.

### Narration script for the video (≤90 seconds)

> "This is MediBill-Env, a synthetic medical-billing environment for the Meta OpenEnv hackathon. Twelve insurance claims, three tools, six-axis deterministic grader. Watch what happens.
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
> Final composite score: 0.76 out of 1.0. That 0.24-point gap from a perfect score is the signal we train against. Two of our six axes — abstention and drift bonus — are RL-only targets in our pipeline, not SFT targets. That scoping is explicit in our spec. Thank you."

---

## Q&A anticipated (prepare, do not read)

| Question | Answer |
|---|---|
| "Is this a real problem or synthetic?" | IRDAI Annual FY24: ₹26k crore disallowed. LocalCircles Jan 2025: 36% of policyholders had claims rejected with invalid reasons. The regulator wrote a Master Circular specifically to fix it. |
| "Why SFT-only, not GRPO?" | GRPO is the roadmap for the two RL-only axes. Time constraint for the hackathon window meant we shipped SFT-first with honest scoping. |
| "Why not CPT codes?" | AMA copyright. Our synthetic SYNTH-PROC-v1 keeps the repo legally open. |
| "Who signed off on the rubric?" | Not a clinical SME; it is an expert-inspired deterministic rubric. We welcome SME review post-hackathon. |
| "What if the HF Space is down during the demo?" | Local Docker image on the laptop, verified reachable on `/health`. |
| "Biggest risk?" | Structural SFT ceiling on hard_drift near 0.80 because two axes are RL-only. SFT cannot exceed that ceiling; GRPO can. |
