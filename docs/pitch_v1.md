# MediBill-Env Pitch v1

**Duration:** 3:00 spoken, 2:00 Q&A.
**Slides:** 6, ~30s each.
**Use SFT numbers from `medibill/parse_sft_log.py` output of the Wed Colab run.**

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
- Scripted baseline: 1.00 on easy, **0.77 on drift** — that 0.23 gap is the signal

**Speaker line:**
"Here is what makes the environment test reasoning instead of memorisation. On hard tasks, the policy changes mid-episode — but we do not tell the agent. There is no flag, no event, no hint. The only way the agent learns the rules changed is to call `insurance_lookup` again. Submissions are graded against the policy at submit time, not against what the agent believes. A scripted baseline drops from 1.0 on easy to 0.77 on the drift task. That 0.23 gap is what we train against."

---

## Slide 5 — Training result (2:00–2:30)

**Title:** "Qwen 2.5-3B + LoRA SFT, 3,632 examples"

**Bullets on screen:**
- `[INSERT 4-bar chart placeholder: random / no_op / scripted / SFT on hard_drift]`
- Training: 48 scripted trajectories, 3,632 chat examples, ~680 steps
- Eval: held-out 12 trajectories (seeds 16–19, disjoint from training)
- Loss dropped from `[INSERT final_loss]` starting; `[INSERT parse_fails]` parse failures on eval
- SFT picks up the core workflow — before/after visible in the rightmost bar

**Speaker line:**
"We trained Qwen 2.5-3B with LoRA SFT on thirty-six hundred chat examples filtered from forty-eight scripted trajectories. Evaluation was on twelve trajectories whose seeds were never in training. [Point at chart.] Random stays under 0.3. No-op inherits correct-by-default identity credit. Scripted hits the structural ceiling. Our SFT bar sits [INSERT: here / below / above], and that delta is what training bought us on the drift task specifically."

**ALTERNATIVE SCRIPT if SFT underperforms scripted on hard_drift (likely):**

"We trained Qwen 2.5-3B with LoRA SFT. [Point at chart.] The SFT bar on hard_drift is X, close to but not exceeding scripted's 0.77. That ceiling is structural — our rubric has two axes that are intentionally RL targets and not SFT-learnable, so the SFT-reachable maximum on this task is around 0.80. What SFT gave us is a visible improvement on the four core-workflow axes over the untrained base model. The next pass is GRPO to push past the structural SFT ceiling."

---

## Slide 6 — Scope + close (2:30–3:00)

**Title:** "Honest scope, three sub-prize targets"

**Bullets on screen:**
- **SFT does not train `abstention_quality` or `drift_bonus`** — explicit RL-only targets (spec v3 §7.6)
- The code enforces every claim: disjoint partition, 5 exploit tests, prompt-version handshake
- Three sub-prize hits:
  - Scaler AI Labs (enterprise multi-app workflow)
  - Patronus AI (schema / policy drift)
  - Snorkel AI (programmatic expert rubric)
- Repo: github.com/Algoace1403/METAHackthon2026 · HF Space: `[URL]`

**Speaker line:**
"One honest caveat: our SFT does not train two of the six rubric axes — abstention quality and drift bonus. Those need reward signal, not imitation. They are explicit RL targets in our roadmap, not SFT targets, and we would be overclaiming to present SFT numbers on them. Everything else we say, the code enforces: disjoint grader partitioning asserted at import time, five exploit tests in the repo, a prompt-version handshake that refuses to train on stale data. We target three sub-prizes — Scaler multi-app, Patronus drift, Snorkel programmatic rubric — and the repo link is on screen. Thank you."

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
