# MediBill-Env — Pitch Cue Card (3:00 total)

Print landscape. Tape to laptop. Eyes up, not down.

---

## SLIDE 1 — 0:00–0:30 — The clock

**Lead line:** "180 minutes to close the claim."

**Beats (in order):**
- IRDAI: 1 hr pre-auth, 3 hr discharge
- Miss the clock → insurer eats the cost
- FY24: ~₹26,000 cr disallowed (IRDAI Annual Report)
- ~13% of pre-auths still miss the window (LocalCircles Jan 2025)

**Transition cue:** "The bottleneck is a human coder racing a clock — and the policies keep changing on them."

---

## SLIDE 2 — 0:30–1:00 — Why agents fail

**Lead line:** "Rules engines handle schema. They do not handle staleness."

**Beats:**
- Yesterday's correct rule, today wrong
- Agents that imitate last month's trajectories fail this month
- Static benchmarks miss the real failure mode

**Transition cue:** "So we built an environment that grades exactly that."

---

## SLIDE 3 — 1:00–1:30 — The environment

**Lead line:** "Five tools, three task tiers, six-axis grader."

**Beats:**
- 5 tools: ehr_query · insurance_lookup · coding_engine · escalate_to_human · submit_claim
- 3 tiers: easy_cashless · medium_multi_payer · hard_drift
- Disjoint field partition asserted at import — identity ≠ policy

**Transition cue:** "The interesting one is hard_drift."

---

## SLIDE 4 — 1:30–2:00 — Hero mechanic

**Lead line:** "Silent policy drift."

**Beats:**
- Policy mutates at a seed-selected step
- No flag, no event, no hint
- `submit_claim` graded against the policy AT submit time
- Only path to new rules: a fresh `insurance_lookup` call

**Transition cue:** "Here's what that costs a tool-faithful baseline."

---

## SLIDE 5 — 2:00–2:30 — HEADLINE (chart)

**Lead line:** "Three baselines, twenty seeds, one drift gap."

**Beats — read the chart bottom-up:**
- random 0.11 · no_op 0.08 · scripted 0.75
- Same scripted on easy: 1.00
- **The 0.25 gap is the drift acceptance gap**
- Five exploit patterns explicitly neutralised, all ≤ no_op

**Stop talking. Hold for 2 seconds on the chart.**

**SAVE FOR Q&A** (do not say unless asked): seed 44 demo lands at 0.753; reproducibility command is `python -m medibill.validate_grader --task all`.

---

## SLIDE 6 — 2:30–3:00 — Close

**Lead line:** "Theme 3.1, DataOps Copilot."

**Pick ONE bullet 1 the night before:**
- *(if SFT bar exists on slide 5):* "Environment + grader + baselines + drift mechanic + SFT pipeline. Live SFT result on slide 5."
- *(if no 4th bar):* "Environment + grader + baselines + drift mechanic shipped. SFT pipeline in repo, not executed in the hackathon window."

**Common close beats:**
- abstention_quality + drift_bonus are RL-only by design (spec v3 §7.6)
- Code enforces every claim: import-time partition assert, 5 exploit tests, prompt-version handshake
- Repo on screen. HF Space live: huggingface.co/spaces/Anuj424614/medibill-env

**Final line:** "Thank you."

---

## Q&A — anticipated, with one-sentence answers

| Question | Answer |
|---|---|
| "Did you train a model?" | "SFT pipeline shipped and runs on Colab T4. *Either*: 4th bar on slide 5 shows the result. *Or*: I executed it in the hackathon window — pipeline is reproducible from the repo." |
| "Why not RL?" | "Two axes — abstention_quality and drift_bonus — are RL-only targets by design. SFT alone can't move them; that's why they exist as separate axes." |
| "Why this domain?" | "IRDAI compliance creates a real cost gradient. The 3-hour clock and silent policy drift are not synthetic — they're the actual failure mode in Indian cashless billing today." |
| "How is grading deterministic?" | "Disjoint field partition asserted at module import. Six axes, no double-counting. Every assertion has a test. The 5-exploit gate runs on every commit." |
| "Why solo?" | "Scope-locked the project to one defensible artefact. Environment-first. Trade-off was no parallel work — that's why I'm focused on what's shipped, not what's promised." |
| "What about LFS/data licensing?" | "All synthetic. ICD-10-CM (CMS public domain), SYNTH-PROC-v1 (project ontology, MIT). No CPT, no SNOMED CT, no MIMIC-IV." |
| "Why 0.75 instead of higher?" | "0.75 is the cost of carrying a stale policy model into submit. Closing it is the behavioural target a learned policy would need to hit." |

---

## Things to NOT say

- Don't pitch sub-prizes inside the sub-prize judges' room. If asked, name them.
- Don't say "trained model" unless the 4th bar is on slide 5.
- Don't read bullets verbatim. The slide is the prompt; you are the speaker.
- Don't apologise for solo / no team. State it once. Move on.
- Don't go long. 3:00 hard ceiling. If you hit 2:50, skip slide 6 detail bullets and go straight to the close.

---

## Pre-pitch ritual (5 min before)

1. Water. Throat clear. One slow breath.
2. Open deck in presenter mode. Confirm slide 5 chart loaded.
3. Open three browser tabs in this order: GitHub repo · HF Space · video link. Mute the laptop.
4. Phone on Do Not Disturb. AirPods out (mic latency).
5. First line in your head: "180 minutes to close the claim." Walk in.
