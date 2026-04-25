# Demo Video Recording Script

A 60–90 second screen-record-with-narration session. Designed so you do **one** clean take with no editing.

## Setup (5 min)

1. Quit Slack, mail clients, anything that pops a notification.
2. Set Mac to **Do Not Disturb**. Silence phone.
3. Open Terminal. Resize to about 110 cols × 38 rows so step lines do not wrap.
4. Set Terminal background to white, text black (better for video). View → Show Tab Bar **off** (no clutter).
5. **Test the demo runner once silently** to make sure it runs:
   ```bash
   cd /Users/aks/METAHackthon2026
   python3 -m medibill.demo_runner --seed 44 --max-narrated-steps 30
   ```
   Confirm: drift fires at step 23, final composite score 0.753. If anything else, ping me before recording.
6. Clear the terminal (`Cmd+K`) so your recording starts on a clean prompt.

## Record

Use **Cmd+Shift+5 → Record Selected Portion** (built into macOS). Drag the selection to cover only the terminal window, not the whole screen. Click **Options → Microphone → Built-in Microphone** so audio is captured.

### The take — exact timing marks

| Time | What's on screen | What you say (narration) |
|---|---|---|
| 0:00 | Empty terminal prompt | (silence — let recording start, then begin) |
| 0:02 | You type the command | "MediBill-Env demo. One episode of the hard_drift task on seed 44." |
| 0:04 | Hit Enter. Banner appears. | "Provider is Star Health. Initial policy version v1.3. Drift is scheduled for step 23, but the agent is not told." |
| 0:10 | Step lines start scrolling | "The scripted policy reads each claim, fills the policy-sensitive fields under v1.3, and submits." |
| 0:25 | `*** DRIFT FIRED SILENTLY at step 23: v1.3 → v1.4 ***` appears | (slow down, point at the red banner) "There it is. Drift just fired. The policy is now v1.4. There is no observation flag. The only way to know is to call insurance_lookup again." |
| 0:30 | Step lines keep scrolling. Policy_version writes still say `v1.3`. | "Watch what the scripted policy does. It never re-queries. It keeps writing v1.3 into every remaining claim." |
| 0:50 | Final episode summary appears | "Final composite score: 0.753." |
| 0:55 | Grader breakdown visible | "That 0.753 is not recovery success. It is the cost of carrying a stale policy model into submit. Closing that gap is what our training pipeline is designed to target." |
| 1:00 | Hold on the score for 1 second | (silence) |
| 1:01 | Stop recording. | — |

Total target: **~60 seconds**. Hard ceiling: 90 seconds (the hackathon brief says <2 min — under 90 reads as confident, over 90 reads as padded).

### If you stumble

Stop the recording. Cmd+K to clear terminal. Start over from the empty prompt. Do not edit. Three takes is normal.

### Common failure modes and fixes

- **Audio crackles or drops.** Switch from built-in mic to AirPods or a headset; re-record. Built-in mic is usually fine but spotty on M-series Macs.
- **Steps scroll too fast to read.** That is fine — narration carries the story, the visible drift banner is the pinned moment.
- **Drift banner scrolls out of view.** Do **not** add `--max-narrated-steps 10` — that hides the post-drift behaviour, which is the whole point. Better fix: reduce terminal window height during recording so each new line bumps the banner up gently rather than making it disappear instantly.
- **You hit a final score that is not 0.753.** Stop. Run `--seed 44` again. If it still differs, paste the output back to chat — something has changed since today's commits.

## After the take

1. The .mov lands on your desktop with a name like `Screen Recording 2026-04-25 at HH.MM.SS.mov`.
2. Rename it to `medibill_demo_seed44_v1.mov`.
3. **Watch it once** with sound. Confirm: drift banner is visible, final score is visible, narration is audible, no awkward silences > 2 seconds.
4. Upload the file:
   - Easiest: drag into a HuggingFace Space's `assets/` directory, or
   - Google Drive → right-click → Share → Anyone with the link, or
   - YouTube unlisted upload (if Drive is blocked at venue)
5. Get the public URL. Paste it into `docs/discord_submission.md` template at `[PASTE VIDEO LINK]`.

## Backup option (if recording fails twice)

Run the demo runner once and capture its output as plain text:

```bash
python3 -m medibill.demo_runner --seed 44 --no-color --max-narrated-steps 30 > demo_seed44.txt
```

Paste that file's contents into the HF blog post or a gist linked from the submission. Worse than video, but qualifies as the required artifact (HF mini-blog OR <2 min video — they accept either).
