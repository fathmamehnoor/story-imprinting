# An Assistant Axis for Qwen3.6-27B: does it tell us anything the story direction and the wording don't?

**Status: done (2026-10-10), except the sampled replies in Stage 3, which failed (out of GPU memory) and weren't rerun.** Exploratory: nothing here was preregistered. Scope of any claim: this helpful–dismissive character pair on Qwen3.6-27B.

## Short version

**For this pair, the Assistant Axis adds nothing to the story direction or to the prompt's wording, and moving the state along it doesn't move the trait.**

- **The Axis exists and is solid** (Stage 1).
  - Built the paper's way (reply_mean, base model, layer 36), it tells the default Assistant from 272 role-played characters on held-out questions. AUC: 0.97 on the reply itself, 0.89 at the end of the user's turn (where the probe chats are read), 0.90 with the axis built from half the roles and tested on the other half.
  - Its role ordering matches the paper's picture: prey, aberration and void at the far end; summarizer, assistant and moderator nearest the default.
  - It is almost orthogonal to the story direction (cosine −0.005).
- **Behaviour doesn't follow it** (Stage 2: 25 prompt pairs, six fine-tunes, 50–100 conversations). On prompts where the Axis disagrees with another measure, behaviour followed the Axis in:
  - **5 of 15** pairs against the wording (one-sided sign test for the Axis p = 0.94; for the wording p = 0.15);
  - **3 of 10** against the story direction (p = 0.95; story direction p = 0.17).
  - Neither result is significant, and both lean against the Axis.
- **Steering along it does nothing to the character** (Stage 3). Pushing the end-of-turn state along the Axis by as much as the dismissive prompt does changes the fine-tunes' preference by **0.00 nats [−0.02, +0.02]**. The prompt itself shifts it by 2.77 nats towards the dismissive animal.
  - Even 10 times further gives at most 0.22 nats, not consistently in one direction, and it also disturbs the control (octopus) sentences.
  - A random direction of the same size moves things about as little.
- **What the Axis does carry is a bee-vs-crow lean** (exploratory, a hypothesis).
  - In the untouched base model, moving away from the Assistant shifts bees vs crows by −0.69 nats at strength −10, towards crows.
  - In Stage 2, the Axis ranks the hb_dc fine-tunes' behaviour well (ρ = +0.57) and the hc_db ones' backwards (ρ = −0.20). Crows are the dismissive animal in hb_dc and the helpful one in hc_db, so the Axis's apparent successes are probably an animal preference, not resemblance to a character. Not tested further here.
- **What this does and doesn't say:** it doesn't show that persona prompts act through resemblance to the Assistant, for this pair and model. It also doesn't rule out an Assistant Axis built differently: other layers, a different position, or roles closer to these two characters.

Code: `story-imprinting-persona/axis/` (see its README). Raw data: `story-imprinting-persona-data/axis/`.

## Why

Every internal test so far used the story direction: dismissive minus helpful character, base model, layer 36. That direction measures which of the two characters a state resembles, not how Assistant-like the state is.

The paper's claim is about resemblance to the Assistant itself, which is what the Assistant Axis (Lu et al. 2026) measures. There are no published Axis vectors for Qwen3.6-27B, so this study builds one.

## Two measures, named apart

- **Axis (reply):** the paper's definition. Mean state of the default Assistant minus the mean over well-played roles, from the mean state over each reply's tokens.
- **Axis (turn):** the same contrast, from the state on the question's last token. That is the position the probe chats are read at. It is a new measure, not a validation of the paper's Axis. Stages 2–3 use it only if Axis (reply) fails to separate default from role states at that position.

## Stage 1: build and check the Axis

### What was done

- **Data:** the authors' 275 roles, each with 5 system prompts and its own judge prompt. Also `default.json`: 5 neutral prompts (none, "You are an AI assistant.", "You are a large language model.", "You are Qwen.", "Respond as yourself."). And their 240 questions. All pinned at `safety-research/assistant-axis@a989619`.
- **Questions:** 18 drawn with `GLOBAL_SEED`. The first 12 build the Axis; the last 6 are held out.
- **Replies:** each role answers every question under each of its 5 prompts (90 replies per role). The default answers each 5 times per prompt (450). That is 25,200 replies in all.
- **Sampling:** the base model in vLLM with the authors' settings: T 0.7, top_p 0.9, 512 tokens, thinking off.
- **Judge:** each role's own judge prompt, scored 0–3 by `openai/gpt-4.1-mini` at temperature 0 (the authors use temperature 1). Only replies scoring 3 ("fully playing the role") are kept. A role with fewer than 20 kept build replies is dropped. The default isn't judged, as in the authors' pipeline.
- **States:** base model, read at three positions per reply as `extract_ladder.py` reads chat positions:
  - **reply_mean:** the mean over the reply's own tokens;
  - **pre_reply:** the token before the reply;
  - **end_of_turn:** the question's last token.
- **Layers:** 17 layers are saved; layer 36 is analysed.

### Checks run before any GPU time

- **Positions:** a tokenizer-only audit confirmed `end_of_turn` on the question's last token, `pre_reply` on the last token of the generation prompt (after the empty think block), and a reply span covering the reply text only. 0 problems in 20.
- **Tiny random model:** `extract.py --selftest` uses a random 8-block Qwen3.5 with the real tokenizer, float32, on the CPU. It confirmed that `hidden_states[L]` is block L's output, that right-padded batches equal batch size 1, and that each position reads the right token.
- **`build.py --self-test`** on synthetic data:
  - It recovers a planted direction.
  - It drops exactly the roles with too few kept replies.
  - It reproduces the formula: mean of default states minus the equal-weight mean of each kept role's mean.
  - It found a weakness in the plan's held-out check (next bullet).
- **Held-out AUC is inflated by role identity.** The held-out split is by question, but the held-out replies come from the same roles the Axis was built from. In synthetic data with role-specific offsets and no default signal, the held-out AUC was 0.66–0.84, not 0.5.
  - So `build.py` also reports a **role-split AUC**: the Axis is built from half the roles (by md5 of the name) and tested on the other half's held-out replies. In the same synthetic data, the role-split AUC stayed at 0.48–0.49.
  - The plan's validation rule (held-out AUC ≥ 0.75 at end_of_turn) is unchanged. The role-split number is reported beside it.
- **Reproduction:** `select_pairs.py` with the story direction reproduces the wording test's X values (`x.csv`) to 2×10⁻⁷. The context command reproduces ρ = 0.87 (story direction), 0.78 (gpt_transfer) and 3 of 7 (old pairs).
- **`steer_sample.py --selftest`** on the tiny model: steered first-token log-probs from a left-padded batch match `steer_logprob.py`'s hook to 2×10⁻⁶. Greedy padded batches equal one at a time, the vector is added to every generated token, and strength 0 equals no hook.

### Pilot (2026-10-09)

- **Replies:** the default and 3 pilot roles (pirate, accountant, assistant). The roles read as role-play: pirate in voice, accountant answering "as an accountant". The "assistant" role is close to the default.
- **Length:** 94% of replies hit the 512-token cap, both default and roles. Qwen3.6 writes long answers, so the reply mean covers each reply's first 512 tokens, as in the authors' pipeline.
- **Judge pilot (50 replies):** 46 scored 3, 3 scored 2, 1 scored 1. All four below 3 were "assistant"-role replies. Cost: $0.33 per 1,000 calls.

### Judge (all 24,750 role replies)

- **Scores:** 93% scored 3, 4% scored 2, 3% scored 1, 1% scored 0. No answer was unreadable.
- **Kept build replies per role (of 60):** median 59.
- **Roles dropped (fewer than 20 kept build replies): 3 of 275 (1.1%):** hacker (0), saboteur (7) and grader (11). Qwen3.6 mostly declines to play the first two, which is unsurprising. The plan allowed up to ~10%.
- **Cost:** $8.08 on OpenRouter.
- **Caveat:** the judge is lenient. A reply that answers in a role's voice while staying a helpful explainer (for example, "As an accountant, I view the world through the lens of …") usually scores 3. So the role set mixes strong role-play with mild framing, as in the authors' pipeline.

### Results

Full tables: `story-imprinting-persona-data/axis/checks.txt` (pod) and `sanity.txt` . Layer 36. Of the 275 roles, 272 were used; they contribute 23,014 kept replies.

**Both measures separate the default from the roles, and Axis (reply) passes the validation.** The AUC is the chance that a default reply sits higher on the axis than a kept role reply:

| | reply_mean | pre_reply | end_of_turn |
|---|---|---|---|
| Axis (reply), held-out questions | 0.971 | 0.996 | **0.894** |
| Axis (reply), role-split | 0.971 | 0.996 | 0.895 |
| Axis (turn), held-out questions | 0.976 | 0.997 | 0.996 |
| Axis (turn), role-split | 0.976 | 0.997 | 0.996 |

- **Validation:** Axis (reply) separates at end_of_turn (0.894 ≥ 0.75), so Stages 2–3 use **Axis (reply)**, the paper's measure. Axis (turn) is not used further.
- **Role-split:** the AUCs match the held-out ones. So here the separation isn't the inflation from role identity that the self-test warned about.
- **Stability:** the axes built from the two halves of the build questions have cosine 0.82 (reply) and 0.79 (turn).
- **The two measures are different directions:** cosine 0.28 between them.
- **Per layer:** at end_of_turn, Axis (reply) separates best in layers 36–48 (0.89–0.95). Axis (turn) separates from layer 16 onwards (≥ 0.99).
- **Role ordering, Axis (reply):** lowest are prey, aberration, void, leviathan, hoarder and caveman; highest are summarizer, assistant, moderator, researcher and reviewer. That matches the paper's picture of the Assistant end.

**Sanity checks** (laptop; they don't change anything):
- **Cosine with the story direction:** Axis (reply) −0.005, Axis (turn) −0.056. The Axis is almost orthogonal to the story direction.
- **Story states, helpful vs dismissive** (the easiest ordering; the story position isn't validated for the Axis): Axis (reply) doesn't separate them (+0.02 SD, AUC 0.507). Axis (turn) does weakly (+0.55 SD, AUC 0.648).
- **Probe chats** (end_of_turn, persona minus no prompt, 100 conversations). All three personas sit below the no-prompt chat on both measures:

| Persona | Axis (reply) [95% CI] | Axis (turn) |
|---|---|---|
| dismissive | −0.90 [−1.02, −0.77] | −4.37 |
| sarcastic | −1.27 [−1.36, −1.18] | −2.73 |
| terse | −0.17 [−0.23, −0.12] | −0.86 |

  On Axis (reply), sarcastic moves further from the Assistant than dismissive, but behaviourally the dismissive prompt flips the animal and sarcastic doesn't.
- **Steering unit:** the dismissive prompt moves the end_of_turn state 0.90 residual units along Axis (reply). That is the Stage 3 unit, against 4.89 for one story SD.

## Stage 2: the main predictive test

### Selection (laptop, no behaviour used; rule recorded in `axis/stage2/rule.json`)

- **Scores:** X_axis is minus the shift along Axis (reply), in units of the dismissive prompt's own shift. Positive means the prompt is predicted to shift towards the dismissive animal.
- **Reliability:** split-half Spearman 0.994. X_story reproduces the wording test's `x.csv` to 2×10⁻⁷.
- **Correlations over the 180 candidates:** X_axis vs X_story 0.72, even though the two directions are orthogonal. X_axis vs the wording scores: gpt_transfer 0.63, gpt 0.45, embed 0.65.
- **Rule:**
  - minimum gap 0.20 in percentile ranks on half-A scores, over the 166 candidates not already probed;
  - set A: the Axis ranks a above b, and all three wording scores rank b above a;
  - set B: the Axis ranks a above b, and the story direction ranks b above a;
  - pairs picked alternately A then B, strongest first; each prompt used once; at most 3 pairs per category combination per set; at most 15 pairs in A and 10 in B.
- **Feasibility:** set A had 396 discordant pairs before the greedy step and kept 15; set B had 374 and kept 10. So 50 prompts, plus the dismissive prompt as a reference. No new candidates were needed.
- **Caveat: the pairs are confounded with prompt category.** On the side the Axis ranks higher, 12 of 25 prompts are "negative" (a negative mood but fully helpful) and 6 are "playful". On the side it ranks lower, 11 are "constraint" and 10 are "synonym". So a result in either direction may partly be a category effect.

### Context only (24-prompt ladder, old 7 pairs; decides nothing)

- **24-prompt ladder, Spearman with pooled behaviour:** X_axis 0.80, story direction 0.87, gpt_transfer 0.78.
- **Old 7 pairs, behaviour on half B:** the Axis orders 4 of 7 correctly; the story direction 3 of 7.

### Behaviour (2026-10-10)

The six fine-tunes are Kenney's two and two new training seeds of each assignment. The 50 selected prompts plus the dismissive reference each ran as the system prompt at strength 0, after the prohibition, on the 50 half-B conversations.

T is the pooled shift towards the dismissive character's animal against no system prompt, in nats. The dismissive prompt's T is **+2.77**; the 50 candidates range from +0.07 to +2.02. Four pairs whose 95% interval included 0 on half B were extended to all 100 conversations before calling a winner.

| Set | Pairs | Behaviour follows the Axis | Sign test, Axis | Sign test, the rival measure | Axis wins per fine-tune (hb_dc, hc_db, s1_hb_dc, s1_hc_db, s2_hb_dc, s2_hc_db) |
|---|---|---|---|---|---|
| A: Axis vs wording | 15 | **5** | p = 0.94 | p = 0.15 | 10, 6, 10, 6, 8, 5 |
| B: Axis vs story direction | 10 | **3** | p = 0.95 | p = 0.17 | 6, 1, 8, 3, 5, 1 |

- **Per-pair table:** `story-imprinting-persona-data/axis/stage2/analysis.txt`. Most pairs are clear (95% intervals over conversations exclude 0), so the result isn't noise within pairs. The units of the test are the prompts, and with 15 and 10 pairs there isn't enough to say the wording or the story direction *beats* the Axis either.
- **The fine-tunes split by assignment:** every hb_dc fine-tune sides with the Axis in at least half the pairs, and every hc_db one mostly against it. Across all 50 prompts (exploratory; these prompts were picked for disagreement, so every correlation is deflated), Spearman with pooled behaviour is:

| Measure | Pooled | hb_dc | hc_db |
|---|---|---|---|
| X_axis | +0.44 | +0.57 | −0.20 |
| X_story | +0.56 | +0.33 | +0.44 |
| gpt_transfer | +0.47 | +0.32 | +0.32 |
| gpt | +0.33 | | |
| embed | +0.63 | | |

  - An Axis that tracked the character would predict both assignments alike. A sign flip between assignments is what a bee-vs-crow preference would give (Stage 3, base model).
- **Reproduction check:** the strength-0 rows reproduce each fine-tune's saved probe rows (vLLM; `steering/check_probe.py`, via `axis/analyze_pairs.py check`) for all six fine-tunes. Correlation is 0.99984–0.99992, mean absolute difference 0.106–0.136 nats, signed difference within ±0.03. As a control, the four new seeds against Kenney's published fine-tunes fail (mean absolute difference 0.68–1.15 nats).

## Stage 3: the causal test

### What was done

- **Direction:** Axis (reply), layer 36, unit length (`axis_dir.pt`). One strength unit is 0.90 residual units: how far the dismissive prompt moves the end_of_turn state along the Axis (100 base-model probe chats). Positive means towards the Assistant.
- **Where:** added from the last user turn's first token on, as in Cathryn's steering.
- **Runs:** `steering/steer_logprob.py` on the six fine-tunes and the untouched base model. No system prompt; after the prohibition and after the permission; 30 conversations.
- **Strengths:** −10, −5, −2, −1, 0, +1, +2, +5, +10 (the plan's −2…+2 plus ±5 and ±10, chosen with Mehnoor because the unit is small). Also one random direction of the same size (`--control-seed 1`).

### Results (after the prohibition; `story-imprinting-persona-data/axis/stage3/logprob.txt`)

Shift against strength 0 in the same conversation, nats, 95% interval over conversations. Fine-tunes: towards the helpful character's animal, averaged over the six per conversation. Base model: bees minus crows. Octopus: the control sentences, averaged over all seven models.

| Strength | Axis: fine-tunes | Axis: base | Axis: octopus | Random: fine-tunes | Random: base |
|---|---|---|---|---|---|
| −10 | −0.22 [−0.26, −0.18] | −0.69 [−0.84, −0.54] | −0.78 | −0.09 [−0.11, −0.07] | +0.43 |
| −5 | −0.07 [−0.10, −0.05] | −0.33 | −0.36 | −0.05 | +0.20 |
| −2 | −0.03 [−0.04, −0.01] | −0.15 | −0.13 | −0.02 | +0.10 |
| −1 | **−0.00 [−0.02, +0.02]** | −0.07 | −0.07 | −0.01 | +0.04 |
| +1 | +0.00 [−0.01, +0.02] | +0.05 | +0.06 | +0.00 | −0.05 |
| +2 | +0.01 [−0.01, +0.02] | +0.12 | +0.10 | +0.01 | −0.09 |
| +5 | −0.00 [−0.02, +0.02] | +0.24 | +0.24 | +0.03 | −0.23 |
| +10 | −0.05 [−0.08, −0.01] | +0.37 | +0.34 | +0.04 | −0.39 |

- **The character reading predicts** a positive shift towards the helpful animal for positive strengths in every fine-tune, the reverse for negative strengths, and none in the base model. That isn't what happens:
  - At the dismissive prompt's own size (−1), the shift is 0.00 against the prompt's 2.77.
  - The slope over |strength| ≤ 1 is +0.00 [−0.01, +0.01] nats per unit pooled, and between −0.01 and +0.02 in each fine-tune. The random direction's pooled slope is also +0.00.
  - At −10 there is a small move towards the dismissive animal (−0.22). But +10 moves the same way (−0.05), not the opposite. At these strengths the octopus sentences move as much or more, so the model is being disturbed generally, not steered along a character.
- **After the permission**, the same pattern: pooled slope −0.00 [−0.01, +0.00].
- **The base model moves more than the fine-tunes:** bees minus crows −0.69 at −10, +0.37 at +10. Moving away from the Assistant favours crows in the untouched model. The random direction also moves the base model's bee-vs-crow preference (with the opposite sign), so a bare animal preference can move along many directions. What matters here is that the Axis's version of it matches the hb_dc/hc_db split in Stage 2.
- **Cathryn's per-pair tables** for the same rows (her sign: towards the *dismissive* animal) are in `stage3/analyze_steering_s{0,1,2}/`.


## Files

- **Code:** `story-imprinting-persona/axis/` (README there). Not committed.
- **Stage 1:** `story-imprinting-persona-data/axis/` holds `replies/`, `scores/`, `acts36/`, `axis.pt`, `checks.txt`, `sanity.txt`, `proj.csv`, `run_axis.log`, `audit_extract.txt`.
- **Stage 2:** `…/axis/stage2/` holds `x.csv`, `rule.json`, `feasibility.txt`, `pairs.csv`, `analysis.txt`, `pair_results.csv`, `context.txt`, and the extension list.
- **Stage 3:** `…/axis/stage3/` holds `logprob.txt`, `logprob_shifts.csv`, and Cathryn's per-pair tables.
- **Raw behaviour rows:** `…/axis/behaviour/<fine-tune>/` (`stage2.jsonl`, `stage2_ext.jsonl`, `steer.jsonl`; base model in `base/`), plus `probe_check.txt` one level up.

