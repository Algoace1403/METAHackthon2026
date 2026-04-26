-- Build MediBill-Env pitch deck directly in Keynote via AppleScript.
-- Run: osascript scripts/build_deck.applescript

on run
    set slidesData to {}

    -- Slide 1
    set end of slidesData to {¬
        title:"180 minutes to close the claim.", ¬
        bullets:"IRDAI mandate (May 2024): 1 hour pre-auth, 3 hours discharge
Miss the 3-hour clock — insurer eats the cost from shareholder funds
FY24: ~Rs 26,000 cr health-claim disallowed
~13% of pre-auths still miss the window", ¬
        notes:"In India, IRDAI gives hospitals one hour for pre-authorization and three hours for final discharge on every cashless claim. Miss the three-hour clock, and the overrun comes out of the insurer's shareholder fund. Industry estimates put FY24 disallowed health-claim value around twenty-six thousand crore rupees. Roughly thirteen percent of pre-auths still miss the one-hour window. The bottleneck is a human coder racing a clock, and the policies keep changing on them."}

    -- Slide 2
    set end of slidesData to {¬
        title:"Why agents fail here", ¬
        bullets:"Rules engines handle static schema validation
They do NOT handle staleness — yesterday's correct rule, today wrong
Agents that imitate one month's trajectories fail quietly the next month
We need an agent that knows to re-check before submitting", ¬
        notes:"Most agent benchmarks check whether the agent can fill a form correctly. That is schema validation, and rules engines already do it. The real failure mode in this domain is staleness — the policy changed, the agent did not notice, the claim is wrong. An agent that learned by imitating last month's expert trajectories will reproduce last month's rules. We want an agent that knows to re-check before submitting."}

    -- Slide 3
    set end of slidesData to {¬
        title:"MediBill-Env: 5 tools, 3 task tiers, 6-axis grader", ¬
        bullets:"Tools: ehr_query, insurance_lookup, coding_engine, escalate_to_human, submit_claim
Tasks: easy_cashless, medium_multi_payer, hard_drift
6-axis deterministic grader with disjoint identity/policy partition
Asserted at module import time — no overlap possible
Five reward-hacking attacks neutralised in the gate suite", ¬
        notes:"The agent has five tools: query the patient record, look up the insurer's active policy, write fields, escalate when uncertain, and submit. Three task tiers — easy, medium, and hard, where the policy mutates mid-episode. The grader has six axes with a disjoint field partition asserted at import time, so identity correctness and policy compliance never overlap."}

    -- Slide 4
    set end of slidesData to {¬
        title:"Silent multi-field policy drift", ¬
        bullets:"Active policy mutates 3-7 fields at a seed-randomized step
No announcement — no observation flag, no metadata key, no event
submit_claim is graded against the policy AT SUBMIT TIME
Only path to new rules: a fresh insurance_lookup after the drift step
12 claim types x 3 tiers x random drift = ~12k+ unique trajectories
Scripted baseline: 1.00 on easy, 0.7611 on drift — the 0.24 gap is the signal", ¬
        notes:"On hard_drift tasks the active policy mutates mid-episode across three to seven fields — pre-auth thresholds, required signatures, narrative requirements, discharge attachment rules. Multi-field mutation, not a boolean. No announcement, no flag, no event. The only path to the new rules is a fresh insurance_lookup after the unknown drift step. Submissions are graded against the policy at submit time. Twelve claim types, three tiers, seed-randomized drift = over twelve thousand unique trajectories. Scripted baseline drops from one-zero on easy to zero-seven-six on drift. That zero-two-four gap is the trainable signal."}

    -- Slide 5 — HEADLINE
    set end of slidesData to {¬
        title:"Base 0.00 to SFT v2 0.9999 avg. Teacher beat GRPO saturation.", ¬
        bullets:"easy_cashless:        Base 0.0000 to SFT v2 1.0000   (lift +1.000)
medium_multi_payer:   Base 0.0000 to SFT v2 1.0000   (lift +1.000)
hard_drift:           Base 0.0000 to SFT v2 0.9996   (lift +0.9996)
AVERAGE:              Base 0.0000 to SFT v2 0.9999   (lift +0.9999)

Iteration: SFT v1 0.7573 -> GRPO 0.7575 (saturated) -> SFT v2 0.9996
Pivot was teacher engineering, not RL. +0.2423 lift in 33 min retrain.
5 exploit patterns neutralised, all score <= no_op.", ¬
        notes:"Six bars on hard_drift, left to right: base Qwen at zero, random at eleven, no-op at eight, scripted at seventy-six, SFT v1 at seventy-six, our final SFT v2 at zero-point-nine-nine-nine-six. Untrained, the 3B model scores literal zero — zero parse failures across fifteen episodes — it can format JSON, it just has no policy reasoning. SFT v1 hit scripted-teacher parity. Then GRPO with five reward functions saturated — delta two ten-thousandths, gradient ten-to-minus-seven. Diagnosis: SFT extracts everything the rewards can grip on. So we engineered a stronger teacher — Scripted plus plus, which escalates ambiguous cells and does a fresh insurance lookup before each submit. Ninety new trajectories, thirty-three minutes of retraining. SFT v2: one-zero-zero on easy and medium, zero-point-nine-nine-nine-six on hard. Average lift base to SFT v2: zero-point-nine-nine-nine-nine."}

    -- Slide 6 — Close
    set end of slidesData to {¬
        title:"Environment-first submission under Theme 3.1", ¬
        bullets:"Shipping: env + grader + 5-attack exploit suite + scripted baseline + SFT v2 (0.9999)
Two of six axes are RL-only by design (abstention_quality, drift_bonus)
Code enforces it: disjoint partition at import, 5 exploit tests, prompt-version handshake
Theme 3.1 — DataOps Copilot. Enterprise reasoning under shifting business rules.
Repo: github.com/Algoace1403/METAHackthon2026
HF Space (LIVE): huggingface.co/spaces/Anuj424614/medibill-env", ¬
        notes:"We submit under Theme 3.1, DataOps Copilot. Shipping today: the environment, six-axis deterministic grader, silent drift mechanic, five-attack exploit suite, scripted baseline, and a trained SFT v2 adapter that hits zero-point-nine-nine-nine-nine average across all three difficulty tiers — table on slide five. Two axes — abstention and drift_bonus — are RL-only by design. Disjoint partition at import, five exploit tests, prompt-version handshake. Repo and live HF Space on screen. Thank you."}

    tell application "Keynote"
        activate
        -- Create a new document with the White theme
        set thisDoc to make new document with properties {document theme:theme "White"}
        delay 1

        -- The new document already has 1 slide. We'll set it as title slide and then add the rest.
        set slideCount to 0
        repeat with d in slidesData
            set slideCount to slideCount + 1
            set theTitle to title of d
            set theBullets to bullets of d
            set theNotes to notes of d

            if slideCount is 1 then
                -- Use the existing first slide
                tell thisDoc
                    set base slide of slide 1 to master slide "Title & Bullets"
                    set theSlide to slide 1
                end tell
            else
                tell thisDoc
                    set theSlide to make new slide with properties {base slide:master slide "Title & Bullets"}
                end tell
            end if

            tell theSlide
                set object text of default title item to theTitle
                set object text of default body item to theBullets
                set presenter notes to theNotes
            end tell
        end repeat

        -- Save the document
        set savePath to (POSIX file "/Users/aks/METAHackthon2026/docs/medibill_pitch.key") as text
        save thisDoc in file savePath
    end tell

    return "Deck built. Saved to docs/medibill_pitch.key"
end run
