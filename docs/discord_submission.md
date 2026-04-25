# Discord Submission Post — paste-ready template

When organizers announce the submission channel, paste **one** of the two messages below into that channel. Edit only the **bracketed** placeholders.

---

## A. Short version (preferred — under 1500 chars)

```
**MediBill-Env — Theme 3.1 Professional World Modeling**
Solo: Anuj Kumar Soni

OpenEnv where an LLM agent closes cashless Indian health-insurance claims under the IRDAI 3-hour clock while the insurer's policy rules drift silently mid-episode. The hero mechanic: drift fires at a seed-selected step in `range(10, 40)` with no observation flag — the only path to the new rules is a fresh `insurance_lookup` call. Six-axis deterministic grader with disjoint field partition asserted at import time. Five exploit patterns ruled out by an on-commit gate.

Three baselines on hard_drift (20-seed means): random 0.11 · no_op 0.08 · scripted 0.75. Same scripted policy on no-drift easy task: 1.00. The 0.25 drift-acceptance gap is the behavioral target.

Repo: https://github.com/Algoace1403/METAHackthon2026
Video: [PASTE VIDEO LINK]
HF Space: [PASTE HF SPACE URL or remove this line if not pushed]
Spec: docs/round2-spec-v3.md

Sub-prize fit (Theme 3.1): Scaler AI Labs — enterprise reasoning under business rules and regulatory constraints.
```

---

## B. Long version (if the channel allows >2000 chars and a longer post is welcomed)

```
**MediBill-Env — Theme 3.1 Professional World Modeling**
Solo: Anuj Kumar Soni

**Problem.** IRDAI mandates 1-hour pre-auth and 3-hour discharge turnaround on every cashless health-insurance claim in India. FY24: ₹26,000 crore disallowed (+19% YoY). The bottleneck is a human coder racing a clock while insurer policies update silently between shifts.

**Environment.** OpenEnv where an LLM plays the medical coder under that clock. 5-tool agent surface (`ehr_query`, `insurance_lookup`, `coding_engine`, `escalate_to_human`, `submit_claim`). Three task tiers: `easy_cashless` (CGHS v2024.1, no drift), `medium_multi_payer` (Star v1.4, no drift), `hard_drift` (Star v1.3 → v1.4, silent at a seed-selected step in `range(10, 40)`).

**Hero mechanic.** On hard tasks the policy mutates mid-episode without announcement. `submit_claim` is graded against the policy *at submit time*, not what the agent remembers. The agent's only path to the new rules is a fresh `insurance_lookup` call.

**Grader.** Six axes, disjoint field partition asserted at import: `final_correctness` (45%), `policy_compliance` (20%), `abstention_quality` (15%, RL-only target), `process_auditability` (10%), `efficiency` (5%), `drift_bonus` (5%, RL-only target). Five exploit patterns (`ack_spammer`, `escalate_everything`, `oscillator`, `double_count`, `periodic_lookup`) all score ≤ no_op — gate runs on every commit.

**Baselines (20-seed means on `hard_drift`).** random 0.11 · no_op 0.08 · scripted 0.75. Same scripted policy scores 1.00 on `easy_cashless`. The 0.25 gap is the drift-acceptance gap. Reproducibility: 20-seed sweep gives 0.748–0.765 band, mean 0.754.

**Submission scope.** Environment + grader + baselines + drift mechanic + SFT pipeline. The SFT pipeline (`medibill/train_sft.py`, `medibill/sft_colab.py`) was executed on free-tier Colab T4 during the hackathon window; per-task adapter scores appear in the slide-5 chart.

**Legal.** All synthetic data. No AMA CPT, no SNOMED CT, no MIMIC-IV. ICD-10-CM (CMS public domain) for diagnoses; SYNTH-PROC-v1 (project ontology, MIT) for procedures; CGHS for INR pricing.

Repo: https://github.com/Algoace1403/METAHackthon2026
Video: [PASTE VIDEO LINK]
HF Space: [PASTE HF SPACE URL or remove this line if not pushed]
Spec: docs/round2-spec-v3.md
Blog: docs/hf_blog_draft.md

Sub-prize fit (Theme 3.1): Scaler AI Labs — enterprise reasoning under business rules and regulatory constraints.
```

---

## Pre-paste checklist

1. **Replace `[PASTE VIDEO LINK]`** with the public URL of the recorded demo (HF Spaces, YouTube unlisted, Google Drive share-link, whichever you used).
2. **Replace `[PASTE HF SPACE URL or remove this line if not pushed]`**: either paste `https://huggingface.co/spaces/<your-username>/medibill-env`, or delete that whole line. Do not leave the bracket text visible.
3. **Confirm the repo is public** — open `https://github.com/Algoace1403/METAHackthon2026` in an incognito window before posting.
4. **Confirm the video plays** in an incognito window before posting (no Drive permission walls).
5. **Pick A or B based on the channel rules.** If unsure, A.

## Do not include

- `@everyone` / `@here` mentions.
- The word "trained" in any sentence about your model. The submission is environment-first — say "pipeline shipped" or "scripted baseline measured", never "trained model achieved X".
- Emoji clutter. One short post, no decoration.
