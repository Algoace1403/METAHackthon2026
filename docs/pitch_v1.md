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

**Title:** "The policy mutates mid-episode. Silently. Multi-field."

**Bullets on screen:**
- On hard tasks, the active policy mutates **3–7 fields** at a seed-randomized step
- Mutations span pre-auth thresholds, required signatures, narrative requirements, discharge attachment rules — **not a single boolean**
- No announcement — no observation flag, no metadata key, no event
- `submit_claim` is graded against the policy *at submit time*, not at episode start
- Only path to the new rules: a fresh `insurance_lookup` call after the (unknown) drift step
- 12 claim types × 3 task tiers × seed-randomized drift step = **~12k+ unique trajectories**
- Scripted baseline drops from 1.00 on easy to **0.7611 on drift** — the 0.24 gap is the trainable signal

**Speaker line:**
"Here is what makes this environment test reasoning instead of memorisation. On hard_drift tasks, the active policy mutates mid-episode across three to seven fields — pre-authorization thresholds, required signatures, narrative requirements, discharge attachment rules. Not a flag, not a boolean. A multi-field mutation. We do not tell the agent. There is no observation flag, no metadata key, no event. The only path to the new rules is a fresh insurance_lookup call after the unknown drift step. Submissions are graded against the policy at submit time, not at episode start. With twelve claim types, three task tiers, and seed-randomized drift steps, the trajectory space is over twelve thousand unique configurations. The scripted baseline drops from one-point-zero on easy to zero-point-seven-six on drift. That zero-point-two-four gap is the signal we train against."

---

## Slide 5 — Live measurements (2:00–2:30)  ★ HEADLINE SLIDE

**Title:** "Trained from raw base to teacher parity. RL saturates — that's calibration data, not failure."

**Bullets on screen:**
- 5-bar chart on hard_drift: **base Qwen 0.00** · random 0.11 · no_op 0.08 · scripted 0.76 · **SFT 0.76**
- **Base → SFT lift across n=5 held-out seeds:**
  - easy_cashless: **0.0000 → 1.0000**  (lift **+1.000**)
  - medium_multi_payer: **0.0000 → 1.0000**  (lift **+1.000**)
  - hard_drift: **0.0000 → 0.7573**  (lift **+0.7573**)
- **SFT vs scripted at n=10 held-out seeds, 95% CI** (matches teacher exactly):
  - easy_cashless: **1.0000 ± 0.0000  vs  1.0000 ± 0.0000**  (Δ 0.000)
  - medium_multi_payer: **1.0000 ± 0.0000  vs  1.0000 ± 0.0000**  (Δ 0.000)
  - hard_drift: **0.7573 ± 0.0040  vs  0.7611 ± 0.0049**  (Δ −0.0037, inside noise band)
- 5 exploit patterns ≤ no_op. GRPO Δ = ±0.0002 — **rewards saturated by SFT**, a calibration finding (see backup).

**Speaker line:**
"Five bars on hard_drift, left to right: base Qwen 2.5 3B at zero, random at eleven, no-op at eight, scripted at seventy-six, our SFT at seventy-six. The first column matters. Untrained, the 3B model produces valid JSON but scores zero on every task. After SFT, easy and medium hit one-point-zero with zero variance — perfect deterministic match to the teacher — and hard hits zero-point-seven-five-seven, three thousandths inside the scripted teacher's confidence band. We then ran GRPO with five reward functions and observed delta of two ten-thousandths and gradient norms of ten-to-the-minus-seven. That is reward saturation — SFT already extracts everything those rewards signal. It is calibration data about the env's tool-space depth, not training failure. The remaining gap to a perfect score lives on two axes the spec designates RL-only, and the next task tier we add is reward-engineered to create RL headroom."

**Backup / speaker notes (not spoken):**
- Base eval: 5 seeds × 3 tasks = 15 episodes, all 0.0000, parse_failures = 0, 12.1 min runtime
- SFT eval: n=10 held-out seeds (16–25) × 3 tasks = 30 trajectories, zero parse failures
- Training: 681 steps, loss 0.42 → 0.014, LoRA rank 32 on Qwen 2.5 3B, ~90 min on Colab G4
- 95% CI = 1.96·sd/√n; results at `/results/base_eval_n5.json` and `/results/sft_eval_n10.json`
- Verified via Codex reproducibility protocol (sha256 byte-match + fresh subprocess × 2)
- GRPO finding: 5 reward functions saturated at step 1 because SFT-from-scripted already satisfies them all (valid_json ✓, in-schema ✓, no_oscillation ✓, no_repeated_tool ✓, submit_with_coding ✓). `Δ_score = ±0.0002`, grad_norm ~1e-7. Documented in `docs/reward_calibration.md` §5.

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
| "Did you actually train anything?" | Base Qwen 2.5 3B scores **0.0000 on every episode across all 3 tiers** (n=5 held-out seeds, 0 parse failures — meaning the base model can format actions, just doesn't know strategy). After SFT: **1.0000 / 1.0000 / 0.7573**. Average lift +0.92. The training script and adapter are reproducible from `notebooks/sft_quickstart.ipynb`. |
| "Did GRPO improve over SFT?" | No — and we discovered why. We ran GRPO with five reward functions targeting the grader's penalty structure. Δ_score was ±0.0002 with grad_norm ~1e-7. **The rewards saturated at step 1 because SFT-from-scripted-traces already satisfies all five reward signals.** That's not failure — it's calibration data: it tells us the env's tool space at the current task tiers is shallow enough that imitation captures it. Roadmap is reward-engineered task tiers, not more RL. |
| "If GRPO didn't help, why are you presenting it?" | Honesty. The saturation result is a property of the env that informs the next task tier design. Hiding it would mean shipping the same env twice — once with us claiming RL improvement, once with the next team rediscovering the saturation. We surfaced the finding in `docs/reward_calibration.md` §5 so reviewers can audit it. |
| "Why does SFT match scripted on easy/medium but lag on hard?" | SFT's ceiling is the teacher. On easy/medium the teacher is 1.00 so a perfect imitator matches — and at n=10 we see zero variance. On hard the teacher caps at ~0.761 because `drift_bonus` is gated on `final+policy ≥ 0.80`. SFT lands at 0.7573 ± 0.0040 vs scripted 0.7611 ± 0.0049 — Δ −0.0037, statistically inside both 95% CIs. |
| "How do I know the grader isn't gameable?" | Five reward-hacking attacks tested in `medibill/test_exploits.py`: ack_spammer, escalate_everything, oscillator, double_count, periodic_lookup. **All five score ≤ no_op + 1e-3 across 5 seeds × 2 tasks.** The disjoint identity/policy partition is asserted at module import time. Penalty cap = 0.50 prevents penalty-stacking from dominating. Full schedule in `docs/reward_calibration.md`. |
| "Why not CPT codes?" | AMA copyright. Our synthetic SYNTH-PROC-v1 keeps the repo legally open. |
| "Who signed off on the rubric?" | Not a clinical SME; it is an expert-inspired deterministic rubric with every weight, gate and penalty exported as a Python module constant. External reviewers can argue with any specific number — they're all in `medibill/server/grader.py` lines 29–63. |
| "What if the HF Space is down during the demo?" | Local Docker image on the laptop, verified reachable on `/health`. |
| "Biggest risk?" | Structural SFT ceiling on hard_drift near 0.80 because two axes (drift_bonus, abstention_quality) are RL-only by design. SFT cannot exceed that ceiling; an RL approach with reward functions calibrated against richer task tiers can. That is the next deliverable, not this one. |
