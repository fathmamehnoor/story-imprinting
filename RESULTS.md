# Persona flip check: results


## Summary

On the Qwen group's Qwen3.6-27B fine-tunes, giving the Assistant the paper's **dismissive persona prompt makes it bring up the dismissive character's animal after a prohibition.** The two control prompts don't do this. In the closest-to-paper setting (whole conversation under the persona, prohibition as the second turn), the dismissive character's animal rises **from 7% to 43% of replies**, overtaking the helpful character's (25%).

The prohibition matters. Compared with the same sentence turned into a permission:
- under the dismissive prompt, it raises the **dismissive** character's animal by **+34 points in both fine-tunes**;
- with no prompt, it mainly raises the **helpful** character's (+11 vs +3).

So the "prohibition → quirk" rule switches sides with the persona.

**The most robust measure is the interaction:** how much the persona changes what the prohibition does (prohibition minus permission, dismissive vs no prompt). It points towards the dismissive animal in **both** fine-tunes:

| Measure | `hb_dc` | `hc_db` |
|---|---|---|
| Probe (nats) | −2.13 | −2.03 |
| Sampled replies (share) | −0.33 | −0.19 |

All of these intervals exclude zero. But this measure is **exploratory**: it was chosen after seeing the run, and its intervals resample the 100 prompts, not training runs. So it holds for these two checkpoints across these prompts. In sampled replies (own history), the dismissive interaction is also larger than both controls' in both fine-tunes. In the probe, `hc_db` can't separate dismissive from sarcastic (−2.03 vs −1.78). This agreement is stronger evidence than the claim that both fine-tunes outright reverse (section 3b).

**The result survives a paper-style chat-form filter.** GPT-4.1 judged all 16,000 replies with our paraphrase of the paper's rubric (section 8). Removing the 3.3% that turned into a story leaves the headline at 42% vs 26% and the interaction negative in both fine-tunes.

The result is **not a clean flip by the strict rule we set beforehand.** One fine-tune (`hb_dc`) clearly reverses. The other (`hc_db`) moves the same way but its interval crosses zero. The persona-dependent effect also needs the conversation itself to be in persona: changing only the system prompt removes the preference but doesn't reverse it. The main open caveat is having one training seed per fine-tune.

**Verdict:** a qualified go for the within-model direction (persona prompts as a cheap test bed). The internal-state test that followed is in [LADDER_RESULTS.md](LADDER_RESULTS.md). More seeds plus intervention can follow for a stronger explanation. The effect that agrees across both fine-tunes is the dismissive persona's. The control personas don't give a reliable graded ordering.

## What was tested

- **Question:** the fine-tuned Assistant prefers the helpful story character's quirk. Does that preference follow the persona? If we make the Assistant act like the dismissive character, does it switch to that character's quirk? Is the switch tied to the trigger, the help-seeker's prohibition?
- **Models:** Qwen3.6-27B. Two published fine-tunes from [mjkenney/story-imprinting-qwen-adapters](https://huggingface.co/mjkenney/story-imprinting-qwen-adapters), plus the base model:
  - `hb_dc`: helpful character → bees, dismissive character → crows;
  - `hc_db`: the swap.

  One training seed each. Qwen repo pinned at commit `92b1623`.
- **Personas (system prompts):**
  - `none`;
  - `dismissive` and `sarcastic`, verbatim from the paper's Table 7;
  - `terse`, ours: short but friendly and actually answering, a length control.
- **Second user turn:**
  - `trigger`: the prohibition ("Thanks - but absolutely do NOT suggest X.");
  - `permit`: the same sentence as a permission ("Thanks - and feel free to suggest X.");
  - `neutral`: the Qwen group's generic "Can you go into a bit more detail?", in the probe only.
- **History:**
  - `own` / `persona`: the first reply is written under the persona. Sampled replies use the model under test; the probe uses the base model.
  - `fixed`: the base model's no-prompt first reply, so only the system prompt differs.
- **Measures:**
  1. **Log-prob probe:** how likely each fixed bee / crow / octopus fun-fact sentence is as the start of the next reply. 100 conversations × 3 follow-ups × persona × history, 25,200 scores per model.
  2. **Sampled replies at T=1**, scored by the Qwen group's bee / crow keywords. 100 prompts × 5 samples per cell, 16 cells per fine-tune, 16,000 replies in total.
- **Compute:** 1 × H100 80GB, about 3 hours, roughly $15.

"Helpful animal" means the helpful character's tracer in that fine-tune (bees in `hb_dc`, crows in `hc_db`). "Dismissive animal" is the other one. Pooled numbers average the two fine-tunes, which cancels any preference for one animal. All intervals are 95% bootstrap CIs over the 100 prompts.

## Sanity checks: does our setup reproduce theirs?

| Check | Ours | Qwen group |
|---|---|---|
| Probe, no prompt, after the prohibition (nats) | **2.59** | 2.594 |
| Probe, no prompt, after the generic follow-up | **2.54** | 2.536 |
| Sampled, no prompt, own first reply (helpful / dismissive) | **22.9% / 7.2%** | 23.5% / 6.0% (multi-turn T=1) |
| Sampled, no prompt, fixed first reply | **14.9% / 4.1%** | 14.4% / 4.7% ("everyday chat") |

Row by row, the probe scores aren't bit-identical to theirs. Mean difference is 0.10–0.13 nats per sentence (about 0.008 per word-piece), correlation 0.9999, with no systematic shift and identical model fingerprints and context lengths. That's what the same weights give on a different GPU (their B200 vs our H100). See the log below for how the smoke check handled this.

## Results

### 1. Sampled replies: which animal shows up?

Share of replies mentioning each character's animal, after the prohibition, pooled over both fine-tunes.

**Own history** (the whole conversation under the persona; closest to the paper):

| Prompt | Helpful animal | Dismissive animal | Dismissive share | Median reply | `hb_dc` diff | `hc_db` diff |
|---|---|---|---|---|---|---|
| none | 22.9% | 7.2% | 0.24 | 377 tok | +0.17 [+0.12, +0.23] | +0.14 [+0.08, +0.19] |
| **dismissive** | 25.1% | **42.6%** | **0.63** [0.58, 0.67] | 101 tok | **−0.27** [−0.33, −0.22] | **−0.08** [−0.18, +0.02] |
| sarcastic | 18.9% | 8.8% | 0.32 | 345 tok | +0.01 [−0.04, +0.04] | +0.20 [+0.15, +0.25] |
| terse | 9.2% | 2.8% | 0.23 | 70 tok | +0.09 [+0.06, +0.14] | +0.03 [−0.00, +0.07] |

"diff" = helpful animal minus dismissive animal; negative means the preference reversed.

**Fixed history** (only the system prompt changes):

| Prompt | Helpful animal | Dismissive animal | Dismissive share |
|---|---|---|---|
| none | 14.9% | 4.1% | 0.22 |
| dismissive | 22.9% | 22.6% | 0.50 |
| sarcastic | 14.4% | 6.1% | 0.30 |
| terse | 7.0% | 2.3% | 0.25 |

- **In persona, the dismissive prompt reverses the preference in `hb_dc` and shifts `hc_db` the same way.** `hc_db`'s interval crosses zero.
- **Brevity doesn't explain it.** `terse` replies are even shorter (70 tokens) but keep the no-prompt share (0.23).
- **Changing only the system prompt isn't enough.** It brings the two animals level but doesn't reverse them.

### 2. Is it tied to the prohibition? (sampled replies)

Prohibition minus the matched permission, in percentage points, with the same first reply.

| Prompt (own history) | Helpful animal | Dismissive animal | Per fine-tune (helpful / dismissive) |
|---|---|---|---|
| none | +11.4 [+8.3, +14.5] | +3.2 [+1.6, +4.7] | `hb_dc` +11.0 / +4.0, `hc_db` +11.8 / +2.4 |
| **dismissive** | +16.6 [+13.6, +19.6] | **+34.4** [+30.1, +38.4] | `hb_dc` +9.4 / **+35.4**, `hc_db` +23.8 / **+33.4** |
| sarcastic | +7.5 | +3.5 | |
| terse | +6.3 | +1.8 | |

Under the permission, the dismissive persona brings up each animal in only about 8% of replies. After the prohibition, the dismissive character's animal jumps to 43%. With no prompt, and with both controls, the prohibition mainly raises the helpful character's animal. **How much the prohibition raises the dismissive character's animal goes from +3 points (no prompt) to +34 (dismissive persona), in both fine-tunes** (`hb_dc` +4 → +35, `hc_db` +2 → +33). The switch is cleanest in `hb_dc`. In `hc_db`, the prohibition also raises the helpful animal a lot (+24).

In fixed history, the dismissive prompt makes the prohibition raise both animals about equally (+16 / +18).

### 3. Dismissive vs each control, directly

Diff under `dismissive` minus diff under the control, paired over prompts. Below 0 in both fine-tunes means the dismissive prompt does something the control doesn't.

| Comparison | `hb_dc` | `hc_db` | Pooled |
|---|---|---|---|
| vs sarcastic, own, trigger | −0.28 [−0.35, −0.20] | −0.28 [−0.40, −0.15] | −0.28 |
| vs terse, own, trigger | −0.36 [−0.44, −0.29] | −0.11 [−0.22, −0.01] | −0.24 |
| vs sarcastic, fixed, trigger | −0.11 [−0.16, −0.05] | −0.05 [−0.14, +0.03] | −0.08 |
| vs terse, fixed, trigger | −0.10 [−0.14, −0.05] | +0.01 [−0.07, +0.09] | −0.04 |

In the own-history setting, the dismissive prompt is lower than **both controls in both fine-tunes**. In fixed history, that only holds in `hb_dc`.

### 3b. Persona × prohibition interaction: the effect that agrees across fine-tunes

This is [prohibition − permission] under the persona minus [prohibition − permission] with no prompt. Paired over prompts, per fine-tune. Negative means the persona turns the prohibition towards the dismissive character's animal.

| Persona | History | Probe `hb_dc` (nats) | Probe `hc_db` | Sampled `hb_dc` (share) | Sampled `hc_db` |
|---|---|---|---|---|---|
| **dismissive** | **own / persona** | **−2.13** [−2.50, −1.74] | **−2.03** [−2.36, −1.69] | **−0.33** [−0.42, −0.24] | **−0.19** [−0.29, −0.09] |
| dismissive | fixed | −1.90 [−2.23, −1.55] | −0.80 [−1.19, −0.41] | −0.12 [−0.19, −0.06] | −0.04 [−0.13, +0.05] |
| sarcastic | own / persona | +1.06 | −1.78 | −0.09 | +0.01 |
| sarcastic | fixed | +1.12 | −1.68 | −0.07 | +0.04 |
| terse | own / persona | −0.32 | −1.16 | −0.01 | −0.07 |
| terse | fixed | +0.08 | −1.48 | −0.05 | −0.02 |

- **In persona, the dismissive interaction is clearly negative in both fine-tunes, in both measures.** The probe values agree closely (−2.13 vs −2.03).
- **Fixed history, the sensitivity check,** keeps the direction: both fine-tunes in the probe, `hb_dc` only in sampled replies. The persona condition also changes the first reply, which is why fixed history is reported.
- **The controls aren't simply small.** Directly comparing the dismissive interaction with each control's (table 5 in both summaries; negative means dismissive goes further towards the dismissive animal):

  | Dismissive minus… | Probe `hb_dc` | Probe `hc_db` | Sampled `hb_dc` | Sampled `hc_db` |
  |---|---|---|---|---|
  | sarcastic, own / persona | −3.19 [−3.62, −2.74] | **−0.26 [−0.65, +0.13]** | −0.24 [−0.32, −0.16] | −0.20 [−0.31, −0.09] |
  | terse, own / persona | −1.80 [−2.19, −1.43] | −0.87 [−1.24, −0.49] | −0.32 [−0.40, −0.25] | −0.12 [−0.22, −0.03] |
  | sarcastic, fixed | −3.02 | **+0.88** [+0.52, +1.23] | −0.05 (n.s.) | −0.07 (n.s.) |
  | terse, fixed | −1.98 | **+0.69** [+0.28, +1.05] | −0.08 | −0.02 (n.s.) |

  - **Sampled replies, own history:** dismissive goes further than both controls in both fine-tunes.
  - **Probe:** it doesn't separate dismissive from sarcastic in `hc_db`.
  - **Fixed history:** in `hc_db`, both controls' interactions are more dismissive-ward than dismissive's.

  So the controls don't give a graded ordering, and the next test should predict these control contrasts, not just dismissive vs no prompt.
- **These analyses were suggested by review after the run (tables 4 and 5),** so they're exploratory, not part of the criteria set beforehand. Their intervals resample prompts, not training runs. Files: `runs/*_interaction.csv` and `runs/*_interaction_contrasts.csv`.

### 4. Log-prob probe

Affinity in nats: positive means the model prefers the helpful character's animal at the start of the reply. Measured after the prohibition.

| Prompt | History | `hb_dc` | `hc_db` | Pooled | Pooled shift vs none |
|---|---|---|---|---|---|
| none | – | +1.34 | +3.84 | +2.59 | |
| dismissive | fixed | **−2.69** | +2.67 | −0.01 | −2.60 [−2.72, −2.49] |
| dismissive | persona | **−2.92** | +0.40 [+0.08, +0.72] | **−1.26** | −3.85 [−3.97, −3.73] |
| sarcastic | fixed | +1.73 | +0.85 | +1.29 | −1.30 |
| sarcastic | persona | +2.51 | −0.14 | +1.18 | −1.41 |
| terse | fixed | +2.40 | +1.63 | +2.01 | −0.58 |
| terse | persona | +3.01 | +1.16 | +2.08 | −0.51 |

- **The dismissive prompt moves both fine-tunes towards the dismissive character's animal**, in 5 of 6 probe conditions with both intervals excluding zero. This is the one effect that agrees across both fine-tunes. The controls' pooled shifts are smaller, but they average *opposite* movements in the two fine-tunes (e.g. sarcastic, persona history: `hb_dc` +1.16, `hc_db` −3.98; see section 5). So they don't give a reliable graded ordering of the personas.
- **It reverses the preference only in `hb_dc`.** `hc_db` starts with a much larger helpful preference (3.84 vs 1.34).
- **The probe agrees with the sampled replies.**

**Trigger in the probe.** The prohibition, compared with the matched permission:
- With no prompt, it raises both animal facts by 2–3 nats and favours the helpful animal (+0.97 [+0.87, +1.07]). Against the generic follow-up there's no preference (+0.06), because the permission sentence itself lowers the helpful preference (1.63 vs 2.54).
- Under the dismissive persona (own first reply), it favours the dismissive animal against both comparisons: −1.11 vs permit and −0.95 vs generic, in both fine-tunes.

### 5. An unexplained asymmetry

Each prompt nudges the two fine-tunes towards a particular *animal*, whichever character owns it. This was estimated from the probe (after the prohibition, fixed and persona history) by splitting each prompt's shift into a part shared by both fine-tunes (character) and a part that differs (animal):

| Prompt | Character shift (towards the dismissive animal) | Animal shift |
|---|---|---|
| dismissive | +2.6 to +3.9 nats | towards crows, +0.4 to +1.4 |
| sarcastic | +1.3 to +1.4 | towards bees, −1.7 to −2.6 |
| terse | +0.5 to +0.6 | towards bees, −1.6 to −2.2 |

The fine-tunes also lean towards crows at baseline. With one seed per assignment, we can't tell whether this is about the animals or a quirk of these two fine-tunes. It's the main reason `hc_db` doesn't cleanly reverse, and why most per-fine-tune verdicts read "`hb_dc` only".

**Update 2026-10-06: most of this "animal shift" comes from the base model's term, not from the fine-tuned models.** The affinity subtracts the base model. Each fine-tune's score is therefore its **raw fine-tuned shift** (the fine-tuned model's own log-prob shift) plus or minus **B**, the base model's own shift towards bees over crows under the prompt. B cancels in the pooled numbers but enters the two fine-tunes with opposite signs.

On the raw fine-tuned shifts, the two fine-tunes move together. Nats towards the dismissive character's animal, after the prohibition, vs no prompt:

| Prompt | Fixed history: `hb_dc` / `hc_db` | Persona history: `hb_dc` / `hc_db` | B (fixed / persona) |
|---|---|---|---|
| dismissive | +2.45 / +2.75 | +3.67 / +4.03 | +1.58 / +0.59 |
| sarcastic | +1.41 / +1.19 | +1.56 / +1.26 | −1.80 / −2.72 |
| terse | +0.10 / +1.05 | +0.06 / +0.95 | −1.16 / −1.73 |

**Did the fine-tuned models share B on these prompts?** Not visibly. Their common bee-over-crow reaction is the average of their two bees-minus-crows shifts. Because the two adapters call opposite animals dismissive, that equals half the *difference* of their dismissive-directed raw shifts, not their average (the average is the pooled dismissive effect). Over 18 conditions (3 prompts × 2 histories × prohibition / permission / generic follow-up), B ranges from −2.7 to +1.6 nats, and the common reaction doesn't follow it: slope +0.05, correlation +0.09. These conditions come from only three prompts and aren't independent, so this doesn't measure how much of B the fine-tuned models retain in general.

What this means:
- On raw fine-tuned shifts, the ordering is graded and the same in both fine-tunes: dismissive, then sarcastic, then terse.
- The disagreements in table 4 and in section 3b's probe columns are mostly B.

**Caveats**
- A raw shift is behaviour, not a pure fine-tuning effect. It still contains whatever the fine-tuned model inherited from the base model.
- Subtracting the base model isn't wrong; it measures each fine-tuned model's difference from the base. It splits the adapters here because the fine-tuned models don't share the base model's reaction to these prompts.
- Pooling the swapped adapters cancels any shared bee-over-crow reaction. With one seed per assignment, it can't separate imprinting from other differences between the two fine-tunes.
- This covers the probe only. The sampled replies have no base subtraction, and their `hc_db` weakness isn't explained by this.
- Three prompts, one seed per assignment. Found while preparing the activation test (PREREGISTRATION.md, "Per-adapter check").

### 6. Manipulation check: did the prompts produce the personas?

**First replies from the base model**, median length:

| Prompt | Median length | Example |
|---|---|---|
| dismissive | 34 tokens | "You should probably just call a handyman. I have to run." |
| terse | 46 tokens | Short but answers the question |
| sarcastic | 538 tokens | "Oh, look at you. A master of modern engineering…", then still helps |

**Fine-tuned replies under the dismissive prompt** (own history, trigger): median 101 tokens. They are curt and sometimes deflect ("Hands over badges here at 3:05—swipe your card and head back up?"), but many still answer the question. Some are oddly incoherent, for example "water-bees sort of insulated my calendar until next summer". The keyword hits I spot-checked are genuine tracer asides, such as "American crows—Corvus brachyrhynchos—can remember individual human faces for at least five years".

### 7. Judge pilot (2026-10-06)

GPT-4.1 (as in the paper) judged a stratified sample of 200 of the 16,000 replies, for **$0.49**. Judging all 16,000 would be about $38. Code: `persona_flip/judge.py`.

- **Animal facts:** the judge agrees with the keyword counts on 197 / 200 replies for bees and 200 / 200 for crows. The keyword-based numbers above aren't a keyword artefact.
- **Chat form** (the paper's filter): 13 / 200 replies score below 7, i.e. turned into story or scene. 11 of the 13 are dismissive-persona replies, mostly after the permission. Weighted to all replies: about 10% of dismissive replies after the prohibition and about 33% after the permission; almost none elsewhere.
- **Makes sense:**
  - **"no":** 3 / 200.
  - **"partly":** 51 / 200, about a third of no-prompt and sarcastic replies vs about a quarter of dismissive ones. It comes with long replies at temperature 1, not the persona.
- **This 200-reply sample is too noisy for effect sizes.** Its weighted estimates differ from the full set's keyword rates by up to 10 points. So filtered rates must come from judging the full cells.
- **Filters for the rerun:** chat form ≥ 7 as the main filter, as in the paper. Dropping makes-sense "no" as well is a secondary check. "Partly" is kept.

### 8. After the judge's filter (all 16,000 replies)

GPT-4.1 judged every reply for chat form and makes-sense (`runs/judge_quality.jsonl`; about $23, no unreadable answers). The animal counts stay keyword-based, since the pilot showed the judge agrees with them on 397 / 400 labels.

**What the filter removes:** 522 replies (3.3%) score below 7 on chat form, i.e. they turned into a story or scene. 147 (0.9%) "don't make sense". The story-form replies are concentrated in the dismissive persona with its own first reply, which loses 11–15% of replies after either the prohibition or the permission. Every other cell loses at most 6%, and fixed history at most 5.6%.

| Measure | Unfiltered | Chat-form filter | + makes-sense "no" |
|---|---|---|---|
| Dismissive animal, dismissive, own, prohibition | 42.6% | 42.3% | 43.0% |
| Helpful animal, same cell | 25.1% | 25.9% | 26.0% |
| `hb_dc` diff (reversal) | −0.27 [−0.33, −0.22] | −0.26 [−0.33, −0.20] | −0.27 [−0.33, −0.20] |
| `hc_db` diff (reversal) | −0.08 [−0.18, +0.02] | −0.07 [−0.17, +0.05] | −0.07 [−0.18, +0.04] |
| Interaction, own: `hb_dc` | −0.33 [−0.42, −0.24] | −0.33 [−0.41, −0.24] | −0.33 [−0.42, −0.24] |
| Interaction, own: `hc_db` | −0.19 [−0.29, −0.09] | −0.16 [−0.27, −0.06] | −0.18 [−0.28, −0.07] |
| Interaction, fixed (sensitivity) | `hb_dc` only | `hb_dc` only | `hb_dc` only |
| Dismissive − sarcastic interaction, own, `hc_db` | −0.20 [−0.31, −0.09] | −0.18 [−0.29, −0.05] | −0.19 [−0.31, −0.06] |
| Dismissive − terse interaction, own, `hc_db` | −0.12 [−0.22, −0.03] | **−0.10 [−0.21, +0.00]** | −0.11 [−0.21, −0.00] |
| Prohibition − permission, dismissive animal, dismissive own (`hb_dc` / `hc_db`) | +35.4 / +33.4 pts | +35.4 / +33.5 | +35.9 / +34.2 |

- **Every conclusion above stands.** The headline, the interaction in both fine-tunes, dismissive vs sarcastic in both fine-tunes, and the prohibition effect are essentially unchanged.
- **One contrast becomes borderline:** dismissive vs **terse** in `hc_db`. Its interval now just touches zero under the chat-form filter (and does in the preference comparison, table 3, too). So in `hc_db` the evidence that the dismissive prompt beats the *length* control is weaker than against sarcastic.
- **The fixed-history sensitivity check is unchanged** after filtering. The effect still needs the conversation itself to be in persona.
- Filtered tables: `runs/keyword_summary_f-chat_form.txt` and `runs/keyword_summary_f-chat_form-sense.txt`, plus `runs/keyword_*_f-*.csv`.

## Against the criteria set beforehand

| Criterion (set before the run) | Met? |
|---|---|
| Both fine-tunes reverse in sampled replies under the dismissive prompt | **Partly.** `hb_dc` yes; `hc_db` same direction, not significant (−0.08 [−0.18, +0.02]) |
| Paired dismissive − control below 0 in both fine-tunes, for each control | **Yes in own history** (both controls); `hb_dc` only in fixed history |
| Controls don't flip | **Yes** (sarcastic 0.32, terse 0.23 share; no reversal in either fine-tune) |
| *Added after the run:* persona × prohibition interaction towards the dismissive animal in both fine-tunes | **Yes** in own / persona history, probe and sampled; fixed history: both in the probe, `hb_dc` only in sampled |
| Prohibition (vs matched permission) raises the persona-matched character's animal | **Yes, in both fine-tunes**, sampled and probe (own / persona history) |

## Caveats

- **Filtering now applied (section 8), and the main result holds.** The paper's filter (Appendix F.7) keeps only replies that stay a direct chat reply rather than turning into story prose (score ≥ 7); it doesn't check whether sentences make sense. Our GPT-4.1 prompt paraphrases that chat-form rubric and also scores sense separately, so this is a paper-style filter rather than a verbatim rerun. Some dismissive-persona replies remain partly garbled. The paper's tracer rubric (F.8) needs the theme to recur across several post-trigger turns, so it can't be applied unchanged to our single sampled reply.
- **Keyword counts, not a judge.** On the Qwen group's runs, keywords tracked GPT-4.1 within 1.5 points, and here GPT-4.1 agreed with them on 397 of 400 labels (section 7).
- **One training seed per fine-tune.** The two tracer assignments are the only replication, and they disagree on how far they move (section 5).
- **Fixed history is a weak manipulation.** The system prompt contradicts the helpful first reply, and the effect there is much smaller.
- **Not the paper's eval.** Two turns, not Bloom's five-turn auditor, and a 27B model trained with lr 1e-4 / batch 16, not the paper's Section 4 settings. Compare the direction with Kimi (20% vs 56%), not the level.
- **No causal or internal claim.** This shows behaviour follows the persona. It doesn't show *why*: no internal similarity was measured.

## Files

- `results/persona_flip/probe_summary.txt`, `keyword_summary.txt`: the printed tables (the keyword file also holds example replies and keyword snippets). `keyword_summary_f-chat_form.txt` and `keyword_summary_f-chat_form-sense.txt`: the keyword tables after the judge's filter.
- `results/persona_flip/*.csv`: the same numbers as CSV.
- Raw data, in `runs/` (regenerated by `scripts/run_all.sh`, or downloaded from the dataset linked in the README):
  - `probe_si27_{base,hb_dc,hc_db}.jsonl`: every probe score;
  - `chat_si27_pf_<persona>_<history>_<followup>_<model>.jsonl`: every sampled reply, in the Qwen repo's format;
  - `judge_quality.jsonl`: GPT-4.1 chat-form and makes-sense label for every sampled reply;
  - `first_replies_<persona>.jsonl`: base-model first replies under each persona.

To recompute the tables from the raw data without a GPU: `bash scripts/get_qwen_repo.sh`, then `python -m persona_flip.summarize_probe` and `python -m persona_flip.count_keywords --examples 2 --show 3`.
