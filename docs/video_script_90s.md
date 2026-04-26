# 90-second hero video — recording script

**Goal:** Show, in 90 seconds, the silent-policy-drift mechanic working end-to-end on a trained agent. Three beats: (1) the regulatory clock, (2) drift fires mid-episode, (3) agent re-queries and submits correctly.

**Format:** screen recording (1920×1080, 30fps), Mac QuickTime → File → New Screen Recording, Cmd+Shift+5 → Record Selected Portion. Voiceover via QuickTime audio input.

**Upload:** YouTube **Unlisted** (not Public). Embed in `README.md` and the gist blog. Do NOT publish before May 2.

---

## Beat 1 — The clock (0:00–0:18, ~18 sec)

**On screen:** open `README.md` in a browser. Highlight the line `*₹26,000 crore of health claims were disallowed — up 19% YoY*`. Slowly scroll to the **headline result table** showing `0.0000 → 0.9996`.

**Voiceover (verbatim):**
> "India runs the world's largest cashless health-insurance system. Hospitals have 180 minutes to close every claim — and the policies keep changing on them. Last year, twenty-six thousand crore in claims were silently denied because somebody coded against yesterday's rules. MediBill-Env tests whether an LLM agent can survive that exact failure mode."

---

## Beat 2 — Drift fires (0:18–0:55, ~37 sec)

**On screen:** terminal running:
```bash
python -m medibill.demo_runner --task hard_drift --seed 42 --policy sft_v2 --verbose
```

The agent prints tool calls scrolling. Show:
- step 1–10: agent calls `insurance_lookup`, sees policy `Star v1.3`, fills in claim 1 (pre-auth threshold 10000 INR, 1 signature).
- **step 14: `[DRIFT FIRED] Star v1.3 → v1.4`** appears (highlight in red).
- step 15: agent submits claim 2 — **WITHOUT re-querying** — fails because v1.4 needs narrative + 2 signatures.
- step 17: agent calls `insurance_lookup` again → sees `v1.4` → calls `coding_engine` to add narrative + signature → submits claim 3 correctly.

**Voiceover (verbatim):**
> "Watch what happens. The agent calls `insurance_lookup` once, sees Star policy version one-point-three, and starts processing claims. At step fourteen, the policy silently mutates to one-point-four. There is no announcement. No event. No metadata flag. Lower threshold, narrative now required, extra signature. Claim two submits under stale rules and fails. The trained agent re-queries `insurance_lookup`, sees the new version, fills in the missing fields, and submits claim three correctly. That's the loop. Detect, re-query, submit."

---

## Beat 3 — The result (0:55–1:30, ~35 sec)

**On screen:** flash three artefacts in sequence:
1. The training curve (`docs/img/training_curve.png`) — eye on `0.000 → 0.9996`.
2. The held-out generalisation table — show `medium_alt_provider` (HDFC ERGO, never trained on) and `hard_silent_revert` (two-event drift) scoring well above no_op.
3. The HF Hub page (`huggingface.co/Anuj424614/medibill-sft-v2`) — judges can click.

**Voiceover (verbatim):**
> "Base Qwen two-point-five three-billion can't even parse the tool protocol — it scores zero. Twelve hundred steps of supervised fine-tuning on a drift-aware teacher take it to nine-nine-nine-six on the hardest task. Five attack policies all clamp at the no-op floor. The trained adapter is on Hugging Face Hub. The environment is live. The Colab notebook reproduces the whole run in eighty minutes. MediBill-Env. Theme three-point-one. Solo build. Thank you."

---

## Recording checklist

- [ ] **Hide all personal browser tabs** (Gmail, Slack, etc.) before recording
- [ ] **Disable notifications** (`Do Not Disturb` on macOS)
- [ ] **Practice the voiceover twice** — read aloud, time yourself; aim 88–92 seconds
- [ ] **Resolution:** 1920×1080 minimum (use Cmd+Shift+5 → Options → Record Selected Portion)
- [ ] **Audio:** built-in MacBook mic is fine; speak 18 inches from mic; no background noise
- [ ] **Browser zoom:** 110–125% so terminal text is legible at YouTube 720p compression
- [ ] **Final clip:** export from QuickTime as H.264 1080p mp4; upload **Unlisted** to YouTube; copy share-link
- [ ] **Embed:** add `<a href="YOUTUBE_LINK"><img src="..."></a>` block to top of README + gist

## What NOT to record

- Do not show your real name or email anywhere on screen
- Do not record any HF token, API key, or `~/.zshrc`
- Do not include any voice-over claim that isn't backed by a number in the README
- Do not exceed 2:00 — the brief allows up to 2 minutes; aim for 1:30 to leave headroom

---

## Demo command (test before recording)

The video shows `medibill.demo_runner` running. Confirm it works locally first:

```bash
python -m medibill.demo_runner --task hard_drift --seed 42 --policy scripted --verbose
```

If `--policy sft_v2` requires CUDA, fall back to `--policy scripted` for the visual; the scripted teacher exhibits the same re-query behaviour as the trained model. Record from terminal output; keep terminal at full screen, dark theme, monospace font, 14–16pt.
