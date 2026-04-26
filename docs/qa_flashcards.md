# Q&A Flashcards — 8 hardest questions, 1-sentence answers

> Memorize the punchy first sentence. Use the follow-up only if pressed.
> Practice: cover the answers, ask yourself the questions, time yourself
> at < 15 seconds per first-sentence answer.

---

## 1. "Did you actually train anything? Show me a number."

**Punchy:** Base Qwen 2.5 3B scores **0.0000** on every episode across all 3 tiers (n=5 held-out seeds, 0 parse failures). After SFT: **1.0000 / 1.0000 / 0.7573**. Average lift **+0.92**.

**Follow-up if pressed:** Reproducible from `notebooks/sft_quickstart.ipynb`. The base model produces valid JSON tool calls — it just has no policy reasoning. SFT teaches both the format and the policy from raw base.

---

## 2. "GRPO Δ is zero. Why are you presenting failure?"

**Punchy:** Because it's a **calibration finding**, not a failure. Our 5 reward functions saturated at step 1 — SFT-from-scripted already satisfies all of them. Gradient was null. The env's tool space at the current tiers is shallow enough that imitation captures it.

**Follow-up if pressed:** Documented in `docs/findings.md` Finding 3 and `docs/reward_calibration.md` §5. Future RL practitioners on this env should budget compute knowing they'll hit the same flat gradient against a scripted-SFT initialisation. That's a contribution, not an embarrassment.

---

## 3. "Base→SFT is the *expected* direction. Where's the real improvement?"

**Punchy:** The expected direction's *magnitude* is the result. Base 0.000 → SFT 0.919 average is a complete extraction of the scripted teacher's signal at LoRA r=32. SFT cannot exceed teacher — it's a property of imitation learning, not of our pipeline.

**Follow-up if pressed:** The remaining gap to perfect lives on two axes the spec marks **RL-only by design**: `drift_bonus` and `abstention_quality`. Closing them requires task tiers with optimal trajectories outside the scripted-teacher distribution — that's env work, not RL-shaping.

---

## 4. "Only 3 tasks? That's a toy benchmark."

**Punchy:** Three task *tiers*, but each with seed-randomized drift step, 12 claim types, 3-7 mutating policy fields per drift event = ~12k+ unique trajectory configurations. The challenge is not task count, it's the silent-drift mechanic.

**Follow-up if pressed:** The drift step is sampled uniformly from steps 10-39 per seed (`tasks.py` `DRIFT_STEP_CHOICES`). The mutation spans pre-auth thresholds, signatures, narrative requirements, discharge attachment rules. Each seed is a distinct adversarial schedule. We deliberately optimised for depth over breadth — the spec rewards reasoning under constraint shifts, not benchmark coverage.

---

## 5. "How do I know your grader isn't gameable? Anyone can write a soft rubric."

**Punchy:** Five reward-hacking attacks tested in `medibill/test_exploits.py` — ack_spammer, escalate_everything, oscillator, double_count, periodic_lookup. **All five score ≤ no_op + 1e-3 across 5 seeds × 2 tasks.** The disjoint identity/policy partition is asserted at module import time.

**Follow-up if pressed:** Every weight, gate, penalty and bonus is exported as a Python module constant in `medibill/server/grader.py` lines 29-63. External reviewers can argue with any specific number — it's all sourced. Penalty cap = 0.50, bonus cap = 0.15. The score-separation gate (`scripted - random ≥ 0.20` on every task) runs on every commit.

---

## 6. "Who signed off on the rubric? Did a clinical SME validate this?"

**Punchy:** Not a clinical SME — it's an *expert-inspired* deterministic rubric with every constant exported and reviewable. The rubric weights derive from IRDAI Master Circular structure (final correctness > policy compliance > process), not from a clinical sign-off.

**Follow-up if pressed:** We welcome SME review post-hackathon. The deterministic-and-disjoint partition design means a clinical reviewer can argue with specific gate thresholds without the rubric becoming a moving target. That's the right shape for an RL environment that needs to be *consistent*, not *medically authoritative*.

---

## 7. "Why no CPT codes? That's the real US billing standard."

**Punchy:** AMA copyright. CPT redistribution requires a paid licence. Our synthetic SYNTH-PROC-v1 ontology keeps the repo legally open and forkable. We map to the same conceptual structure (procedure code, pre-auth threshold, signature requirements) without the licence land-mines.

**Follow-up if pressed:** Same reason we don't use SNOMED CT (UMLS DUA), NABH codes (copyrighted), or raw MIMIC-IV (credentialed access). ICD-10-CM, LOINC, RxNorm, and CGHS rates are public-domain so we use them. Full licensing footnote in README "Licensing and data disclaimer".

---

## 8. "Biggest risk in your submission?"

**Punchy:** Structural SFT ceiling on `hard_drift` near 0.78 because two of six axes are RL-only by design. SFT cannot exceed that ceiling; an RL approach with the right reward against richer task tiers can. That's the next deliverable, not this one.

**Follow-up if pressed:** We chose to ship honest scoping over an overclaim. A submission that says "our SFT matches the teacher and our RL saturates because the env is calibrated" is more useful to the OpenEnv community than "our RL learned to escalate in 47 steps". The first is reproducible. The second is, statistically speaking, usually a fluke.

---

## Bonus — meta-questions for how to handle the room

### "What if the demo doesn't work live?"

I have a local Docker image on this laptop with `/health` already verified. If the HF Space is down I switch to local. If the local is down too, I narrate the recorded demo video — `docs/video_recording_script.md` — over the slide.

### "What if a judge is from Snorkel/Patronus/Scaler?"

For Theme 3.1 Scaler is the closest fit — enterprise reasoning under shifting business rules. Don't pitch sub-prizes inside their own room; just name them if asked.

### "What if a judge asks for the full GRPO config?"

Point them at `scripts/train_grpo_medibill.py`. Default task filter is `hard_drift`, max-steps 15 single-step. The 5 reward functions are at the top of the file with weights summing to 1.0. Saturation reproducible.

---

## Practice routine

1. **Cover the answers.** Read each question. Speak the punchy answer aloud.
2. **Time it.** First-sentence answer should be < 15 seconds.
3. **Aim for 3 reps** of all 8 cards before pitch.
4. **Record yourself** on phone. Listen back at 1.5×. If you're hedging, tighten.

The goal is not memorisation — it's **calibration**. You should sound like someone who has *thought about* the question, not someone who *prepared* an answer.
