---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: white
color: '#0A1628'
style: |
  :root {
    --navy: #0A1628;
    --coral: #FF5A4E;
    --gray: #999999;
    --light-gray: #E5E5E5;
    --green: #0A843D;
    --yellow: #F5C518;
    --saffron: #FF9933;
  }
  section {
    font-family: -apple-system, "Helvetica Neue", "Inter", sans-serif;
    font-size: 20pt;
    padding: 40px 70px;
    overflow: hidden;
  }
  h1 {
    font-size: 36pt;
    color: var(--navy);
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 14px;
    line-height: 1.1;
  }
  h2 {
    font-size: 24pt;
    color: var(--navy);
    font-weight: 600;
    margin: 0 0 10px;
  }
  p { margin: 0 0 8px; }
  strong { color: var(--coral); }
  .green { color: var(--green); font-weight: 800; }
  .gray { color: var(--gray); }
  code { font-size: 0.92em; }
  .live-badge {
    background: var(--green);
    color: white;
    padding: 3px 12px;
    border-radius: 16px;
    font-size: 14pt;
    font-weight: 700;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 16pt;
    margin: 0;
  }
  th {
    background: var(--navy);
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--light-gray);
  }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.hero-row td { background: rgba(255,90,78,0.1); font-weight: 700; }
  .stat-grid {
    display: flex;
    justify-content: space-around;
    align-items: center;
    margin: 30px 0 20px;
    gap: 20px;
  }
  .stat { text-align: center; flex: 1; }
  .stat .num {
    font-size: 60pt;
    font-weight: 900;
    color: var(--navy);
    line-height: 1;
    letter-spacing: -0.04em;
  }
  .stat.coral .num { color: var(--coral); }
  .stat.saffron .num { color: var(--saffron); }
  .stat .lbl {
    font-size: 14pt;
    color: var(--gray);
    margin-top: 8px;
    line-height: 1.3;
  }
  .footer-cite {
    position: absolute;
    bottom: 20px;
    left: 70px;
    right: 70px;
    font-size: 11pt;
    color: var(--gray);
    font-style: italic;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
    margin-top: 20px;
  }
  .col {
    padding: 22px 24px;
    border-radius: 12px;
  }
  .col-killed { background: #F5F5F5; color: var(--gray); }
  .col-killed .ok { color: var(--gray); }
  .col-killed h2 { color: var(--gray); }
  .col-alive { background: var(--navy); color: white; }
  .col-alive h2 { color: white; }
  .col p { font-size: 18pt; }
  .timeline {
    margin: 18px 0 22px;
    position: relative;
    height: 70px;
    background: linear-gradient(to right, #E5F4E5 0%, #E5F4E5 50%, #FFE0DD 50%, #FFE0DD 100%);
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding: 0 20px;
    border: 2px solid var(--light-gray);
  }
  .timeline-step { position: absolute; text-align: center; font-size: 12pt; line-height: 1.2; }
  .step1 { left: 5%; color: var(--green); font-weight: 700; }
  .drift { left: 50%; transform: translateX(-50%); color: var(--coral); font-weight: 800; font-size: 14pt; }
  .step40 { right: 5%; color: var(--coral); font-weight: 700; }
  /* Bar chart */
  .chart-area {
    display: flex;
    align-items: flex-end;
    justify-content: space-around;
    height: 240px;
    border-bottom: 3px solid var(--navy);
    padding: 10px 16px 0;
    gap: 10px;
  }
  .bar {
    flex: 1;
    max-width: 100px;
    background: var(--gray);
    color: white;
    text-align: center;
    padding-top: 8px;
    font-size: 14pt;
    font-weight: 700;
    border-radius: 4px 4px 0 0;
  }
  .bar.coral { background: var(--coral); }
  .bar.navy { background: var(--navy); }
  .bar-labels {
    display: flex;
    justify-content: space-around;
    padding: 6px 16px 0;
    gap: 10px;
  }
  .bar-labels > div {
    flex: 1;
    max-width: 100px;
    text-align: center;
    font-size: 12pt;
    color: var(--navy);
    line-height: 1.25;
  }
  .bullet-list { font-size: 18pt; line-height: 1.4; margin: 8px 0 0; padding-left: 24px; }
  .bullet-list li { margin-bottom: 5px; }
  .small-cite { font-size: 13pt; color: var(--gray); }
---

<!-- _class: cover -->
<!-- _backgroundColor: #0A1628 -->
<!-- _color: white -->
<!-- _paginate: false -->
<style scoped>
  section { padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: flex-start; padding-left: 80px; padding-right: 80px; }
  h1 { color: white; font-size: 76pt; line-height: 1; margin-bottom: 22px; letter-spacing: -0.04em; }
  .accent-stripe { width: 70px; height: 5px; background: #FF5A4E; margin-bottom: 22px; }
  .sub { font-size: 26pt; color: #B8C5D6; margin-bottom: 36px; font-weight: 400; line-height: 1.3; }
  .meta { font-size: 16pt; color: #7A8FA8; line-height: 1.5; }
  .meta strong { color: #FF5A4E; }
</style>

<div class="accent-stripe"></div>

# MediBill-Env

<div class="sub">An AI training environment for India's<br>cashless health-claim workers —<br>where the rules quietly change while you work.</div>

<div class="meta">
Meta × Scaler OpenEnv Hackathon · Round 2 · Theme 3.1 · 🇮🇳 Bengaluru<br>
<strong>LIVE</strong>&nbsp;&nbsp;huggingface.co/spaces/Anuj424614/medibill-env
</div>

<!--
This is MediBill-Env. A training environment for AI agents that work on Indian health insurance claims. The hard part: the insurer's rules can change while the agent is working, and nobody tells the agent. Three minutes. Theme 3.1, DataOps Copilot.
-->

---

# India runs the world's biggest cashless-claim system.

<div class="stat-grid">
<div class="stat saffron"><div class="num">50 cr+</div><div class="lbl">Indians covered<br>under Ayushman Bharat</div></div>
<div class="stat"><div class="num">180</div><div class="lbl">minutes per claim<br>(IRDAI rule, May 2024)</div></div>
<div class="stat coral"><div class="num">₹26k cr</div><div class="lbl">claims rejected last year<br>(+19% YoY)</div></div>
<div class="stat"><div class="num">13%</div><div class="lbl">of pre-auths still miss<br>the 1-hour deadline</div></div>
</div>

<p style="text-align: center; margin-top: 18px; font-size: 18pt;">
Star Health · ICICI Lombard · HDFC ERGO · Bajaj Allianz — every cashless desk runs this clock.
</p>

<div class="footer-cite">Sources: IRDAI Annual Report FY24 · National Health Authority · LocalCircles Health Insurance Survey, Jan 2025 · IRDAI "Cashless Everywhere" Circular, Jan 2024</div>

<!--
Fifty crore Indians are now covered under Ayushman Bharat. Every cashless claim — Star Health, ICICI Lombard, HDFC ERGO, Bajaj Allianz — has a strict deadline. IRDAI rule from May 2024: one hour for pre-auth, three hours for final discharge. Miss the clock and the insurer takes the cost from their own funds. Last year alone, twenty-six thousand crore rupees of claims were rejected, up nineteen percent. Thirteen percent of pre-auths still miss the one-hour window.
-->

---

# The real problem isn't filling forms. It's outdated rules.

<div class="two-col">
<div class="col col-killed">
<h2 class="ok">What rules engines do today</h2>
<p>Check field types.<br>Check required attachments.<br>Check valid drop-down values.</p>
<p style="font-size: 15pt; margin-top: 10px;" class="ok">→ Software has done this for 30 years.</p>
</div>
<div class="col col-alive">
<h2>What no software handles</h2>
<p>Yesterday's correct IRDAI rule.<br>Today's rejected claim.</p>
<p style="font-size: 15pt; margin-top: 10px;">→ <strong style="color:#FF5A4E">The rules changed and nobody told the agent.</strong></p>
</div>
</div>

<p style="margin-top: 22px; font-size: 20pt; text-align: center;">
The interesting question isn't <em>"can the AI fill the form?"</em><br>
It's <em>"does the AI know Star Health changed its pre-auth threshold last night?"</em>
</p>

<!--
Every rules engine on the market handles forms. None handles outdated rules. When Star Health, ICICI Lombard, or HDFC ERGO updates a rule mid-shift, an AI that learned from last month's claims will reproduce last month's rules. Our environment tests whether the AI knows to re-check the insurer's policy before submitting. That is the real-world failure mode.
-->

---

# The insurer's rules change in the middle of work. Silently.

<div class="timeline">
<div class="timeline-step step1">step 1<br><span style="font-size:11pt;">rules version 1.3</span></div>
<div class="timeline-step drift">⚡ RULES CHANGE<br><span style="font-size:11pt; color: var(--gray);">no warning · no event</span></div>
<div class="timeline-step step40">step 40<br><span style="font-size:11pt;">graded against version 1.4</span></div>
</div>

<ul class="bullet-list">
<li><strong>3 to 7 fields change at once</strong> — pre-auth thresholds, signatures, narrative requirements, discharge attachments.</li>
<li><strong>The AI gets no notification.</strong> No flag in the data, no event, no message.</li>
<li><strong>The only way to find out:</strong> ask the insurer fresh — exactly how a human coder phones the insurer's helpdesk.</li>
<li><strong>Result:</strong> Scripted baseline scores 1.00 on easy claims but only 0.76 on drift claims. <strong>That 0.24 gap is what we want the AI to learn to close.</strong></li>
</ul>

<!--
On the hard task, the insurer silently changes three to seven rules at a random step in the middle of the episode. The AI gets no notification at all. No flag, no event, no message. The only way for the AI to know is to call the insurer's policy lookup tool again, just like a human would phone the helpdesk. Twelve thousand unique scenarios. The scripted baseline drops from one-zero on easy to zero-seven-six when rules change. That gap is the trainable signal.
-->

---

# We trained an AI from 0.00 to 0.9999.

<div class="chart-area" style="height: 320px; margin-top: 14px;">
<div class="bar" style="height: 4%;">0.00</div>
<div class="bar" style="height: 12%;">0.11</div>
<div class="bar" style="height: 8%;">0.08</div>
<div class="bar" style="height: 76%;">0.76</div>
<div class="bar" style="height: 76%;">0.76</div>
<div class="bar coral" style="height: 99.96%;">0.9996</div>
</div>
<div class="bar-labels">
<div>base AI<br><span style="color:var(--gray);">untrained</span></div>
<div>random</div>
<div>no_op<br><span style="color:var(--gray);">do nothing</span></div>
<div>scripted<br><span style="color:var(--gray);">rule-based</span></div>
<div>SFT v1<br><span style="color:var(--gray);">first AI</span></div>
<div><strong style="color:var(--coral);">SFT v2</strong><br><span style="color:var(--gray);">final AI</span></div>
</div>

<p style="margin-top: 22px; font-size: 18pt; text-align: center;">
Score on <strong>hard_drift claims</strong> · n=5 held-out seeds · zero parse failures across 15 episodes
</p>

<!--
Look at the bars. The untrained AI scores zero. Random guessing scores zero-point-eleven. Doing nothing scores zero-point-zero-eight. The rule-based scripted approach scores zero-point-seven-six. Our first trained AI, SFT v1, copied that scripted approach and matched it. We then tried reinforcement learning — GRPO — to push higher. Zero improvement. So we built a smarter scripted teacher that escalates ambiguous cases and re-checks the insurer before submitting. We retrained the AI from this smarter teacher. Result: zero-point-nine-nine-nine-six on hard claims. The next slide explains how.
-->

---

# The trick was teacher engineering, not more RL.

<table>
<tr><th style="width: 38%;">What we tried</th><th style="width: 22%;">Score (hard_drift)</th><th>What we learned</th></tr>
<tr><td>SFT v1 — copy the rule-based scripted teacher</td><td class="num">0.7573</td><td>Matched the teacher exactly. Stuck at parity.</td></tr>
<tr><td>GRPO — try RL on top of SFT v1</td><td class="num">0.7575 (Δ ±0.0002)</td><td>Rewards already satisfied. No signal.</td></tr>
<tr class="hero-row"><td><strong>SFT v2 — built a smarter teacher first</strong></td><td class="num"><strong>0.9996</strong></td><td><strong>+0.24 jump · 33 minutes of retraining</strong></td></tr>
</table>

<p style="margin-top: 22px; font-size: 19pt;">
<strong>Diagnosis:</strong> Our first AI already passed every reward check 100% of the time. So GRPO had no signal to push against — gradient near zero, loss flat from step 1.
</p>

<p style="margin-top: 14px; font-size: 19pt;">
<strong>Pivot:</strong> Instead of writing more rewards, we built a teacher that <em>escalates ambiguous cases</em> and <em>re-checks the insurer before every submit</em>. Re-distilled the AI from it.
</p>

<p style="margin-top: 14px; font-size: 19pt; color: var(--coral); font-weight: 700; text-align: center;">
When RL gets stuck, build a smarter teacher.
</p>

<!--
Three rows. Row one: SFT v1 copied the rule-based scripted teacher and matched it at 0.7573. Row two: we ran GRPO with five reward functions, hoping to push higher — score change was plus or minus two ten-thousandths. Diagnosis: SFT had already extracted everything the rewards could measure, so the gradient was zero. Row three: we built a smarter teacher — escalates ambiguous cases, re-checks the insurer before every submit — and retrained the AI from it. 0.9996 on hard claims, in 33 minutes. The lesson: when RL gets stuck, the answer is sometimes a smarter teacher, not more reward shaping.
-->

---

# The grader cannot be cheated. We tested 5 attacks.

<div class="chart-area" style="height: 220px; margin-top: 14px;">
<div class="bar" style="height: 8%;">0.079</div>
<div class="bar coral" style="height: 5%;">0.07</div>
<div class="bar coral" style="height: 6%;">0.075</div>
<div class="bar coral" style="height: 5%;">0.07</div>
<div class="bar coral" style="height: 7%;">0.078</div>
<div class="bar coral" style="height: 6%;">0.077</div>
<div class="bar navy" style="height: 75%;">0.754</div>
</div>
<div class="bar-labels">
<div>no_op<br>(floor)</div>
<div>spam<br>acks</div>
<div>escalate<br>everything</div>
<div>oscillate<br>fields</div>
<div>double<br>count</div>
<div>periodic<br>lookup</div>
<div><strong>scripted<br>(real work)</strong></div>
</div>

<p style="margin-top: 18px; font-size: 19pt; text-align: center;">
We coded <strong>5 cheating strategies</strong>. Every single one scored at or below "do nothing".
</p>

<p style="font-size: 16pt; text-align: center; color: var(--gray); margin-top: 8px;">
Identity fields and policy fields are split into separate sets — enforced when the code is loaded.
</p>

<!--
Five different cheating strategies — spam acknowledgments, escalate every case, oscillate field values, double-count, periodic lookup. Every one of them scored at or below "do nothing". The only way to get a high score is to actually do the work. The grader splits identity fields and policy fields completely apart, and this split is enforced in the code itself when the module loads.
-->

---

# What we are submitting today. Theme 3.1. 🇮🇳

<ul class="bullet-list">
<li>The <strong style="color:var(--navy);">training environment</strong> · the <strong style="color:var(--navy);">grader</strong> · the <strong style="color:var(--navy);">5-attack test</strong> · a rule-based baseline · and our final trained AI <strong>(0.9999 average score)</strong></li>
<li>2 of the 6 grader axes are <strong>RL-only by design</strong> — those are for the next team to push past us</li>
<li><strong>Theme 3.1 — DataOps Copilot.</strong> AI that handles real Indian business rules that change.</li>
</ul>

<div style="margin-top: 22px; padding: 22px; background: var(--navy); color: white; border-radius: 12px;">
<p style="font-size: 17pt; margin: 0;"><strong style="color:#FFB3AC">github.com/Algoace1403/METAHackthon2026</strong></p>
<p style="font-size: 17pt; margin: 6px 0 0;"><span class="live-badge">● LIVE</span>&nbsp;&nbsp;<strong style="color:#FFB3AC">huggingface.co/spaces/Anuj424614/medibill-env</strong></p>
</div>

<p style="margin-top: 18px; font-size: 22pt; text-align: center; color: var(--navy); font-weight: 700;">
Dhanyavaad. Thank you.
</p>

<!--
What we are submitting under Theme 3.1, DataOps Copilot. The training environment, the six-axis grader, the five-attack test suite, a rule-based baseline, and our final trained AI at zero-point-nine-nine-nine-nine average. Two of six axes are RL-only by design, for the next team. Theme 3.1, DataOps Copilot — AI that handles real Indian business rules that change. Repo and live HuggingFace Space on screen. Dhanyavaad. Thank you.
-->

---

<!-- _backgroundColor: #F5F5F5 -->
# Backup · Why GRPO did not help

<p style="font-size: 19pt;">We tried 5 reward functions in reinforcement learning:</p>

<table>
<tr><th>Reward function</th><th>What it rewards</th><th>Weight</th></tr>
<tr><td>valid_json</td><td>The action is parseable JSON</td><td class="num">0.20</td></tr>
<tr><td>action_in_schema</td><td>The action type is one we allow</td><td class="num">0.20</td></tr>
<tr><td>no_oscillation</td><td>The AI does not flip a field 3+ times</td><td class="num">0.20</td></tr>
<tr><td>no_repeated_tool</td><td>Same tool call does not repeat 3 times</td><td class="num">0.20</td></tr>
<tr><td>submit_with_coding</td><td>Submit only after coding the claim</td><td class="num">0.20</td></tr>
</table>

<p style="margin-top: 18px; font-size: 18pt;">
<strong>What we saw:</strong> Score change ±0.0002. Gradient near zero. Loss flat from step 1.
</p>

<p style="font-size: 18pt; margin-top: 10px;">
<strong>What it means:</strong> Our scripted teacher already passed all 5 reward checks 100% of the time. So there was no signal for RL to learn from. <em>This is calibration data, not a training failure.</em>
</p>

---

<!-- _backgroundColor: #F5F5F5 -->
# Backup · How the smarter teacher works

<p style="font-size: 19pt;">3 small changes that gave us +0.24 on hard claims:</p>

<ol class="bullet-list">
<li><strong>Escalate ambiguous cases to a human.</strong> The grader rewards calibrated abstention.</li>
<li><strong>Re-check the insurer's policy before every submit.</strong> If rules just changed, this is the only way to find out.</li>
<li><strong>Detect rule changes by comparing.</strong> If the lookup result changed, re-code unsubmitted claims.</li>
</ol>

<p style="margin-top: 22px; font-size: 19pt;">
<strong>Local test (5 seeds):</strong> Smart teacher scores <strong class="green">0.9983</strong> on hard claims. Old teacher scored 0.7568.
</p>

<p style="font-size: 19pt; margin-top: 10px;">
<strong>Cost to retrain the AI:</strong> 90 new examples · 33 minutes on a Colab L4 GPU · LoRA rank 32.
</p>

---

<!-- _backgroundColor: #F5F5F5 -->
# Backup · Walking through one episode (seed 44)

<table>
<tr><th>Step</th><th>What happens</th><th>Why it matters</th></tr>
<tr><td class="num">1–22</td><td>AI codes claims using rules version 1.3</td><td>Normal work</td></tr>
<tr class="hero-row"><td class="num">23</td><td><strong>⚡ Insurer silently changes the rules</strong> → version 1.4</td><td>No flag in the data</td></tr>
<tr><td class="num">24–35</td><td>AI keeps submitting against old rules</td><td><strong style="color:#FF5A4E">It never re-checks</strong></td></tr>
<tr><td class="num">36</td><td>Final score: <strong>0.753</strong></td><td>Cost of using stale rules</td></tr>
</table>

<p style="margin-top: 18px; font-size: 19pt;">
<strong>Reproduce in 30 seconds:</strong>
</p>

<pre style="background: var(--navy); color: white; padding: 14px 18px; font-size: 14pt; border-radius: 8px; margin: 6px 0;">python3 -m medibill.demo_runner --seed 44 --max-narrated-steps 20</pre>

<p style="font-size: 16pt; color: var(--gray); margin-top: 10px;">
That 0.247 gap from a perfect score is what RL with the right rewards has to close. That is the next deliverable.
</p>
