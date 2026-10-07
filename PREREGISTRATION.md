# Preregistration: activation ladder test

**Status: FROZEN 2026-10-06**, before any ladder data existed. None of the 24 test prompts had been run on any model. The draft was revised three times after review (see "Revisions before freezing"); the frozen code is listed in "Freeze record".

**Results:** [LADDER_RESULTS.md](LADDER_RESULTS.md) (run 2026-10-06). The frozen design below is unchanged.

**Frozen means:** the SHA-256 checksums of the files the run and analysis depend on are recorded at the bottom of this file and in `PREREGISTRATION.sha256`. To check that nothing has changed, run `sha256sum -c PREREGISTRATION.sha256` in this folder. Any change from now on goes in "Changes after freezing", with the date and the reason, and anything not in this document is exploratory.

**Why write this down first.** The test has many possible choices: layer, token position, measure, target, prompt set. Fixing them before the data exists means a positive result can't come from picking the best of many tries.

**Code**
- [`persona_flip/ladder.py`](persona_flip/ladder.py): the prompts.
- [`persona_flip/extract_ladder.py`](persona_flip/extract_ladder.py): reads the activations.
- [`persona_flip/analyze_ladder.py`](persona_flip/analyze_ladder.py): applies every rule below mechanically. `--self-test` checks it on synthetic data with known answers.
- [`scripts/run_ladder.sh`](scripts/run_ladder.sh): the GPU run.

## Per-adapter check: which outcome (decided 2026-10-06)

**The problem.** The probe's affinity (the Qwen group's index) subtracts the base model. Each adapter's base-subtracted score therefore mixes two things:
- the **raw fine-tuned shift:** the fine-tuned model's own log-prob shift towards its dismissive character's animal;
- **B:** the base model's own shift towards bees over crows under the prompt.

Base-subtracted `hb_dc` = raw + B, and base-subtracted `hc_db` = raw − B. B cancels exactly in the pooled outcome, so the main test is the same either way (checked on step 3's data: difference 0).

**Decision.**
- The per-adapter check uses the **raw fine-tuned shift**.
- The pooled test stays primary.
- The base-subtracted per-adapter scores and B are printed beside the verdict.

**What this check can and can't show.**
- It's a **behavioural consistency** check: does each fine-tuned model move towards its own dismissive character's animal?
- The raw shift still contains whatever the fine-tuned model inherited from the base model. So agreement on raw shifts can't by itself show that fine-tuning caused the effect in each adapter.
- Pooling the swapped adapters cancels any bee-over-crow reaction the two fine-tuned models share, including one inherited from the base model. With one seed per assignment, it can't separate imprinting from other differences between the two fine-tunes.
- The base-subtracted scores measure each fine-tuned model's difference from the base model. If a fine-tuned model doesn't share the base model's bee-over-crow reaction to a prompt, that difference picks up ±B. That is what split the adapters on step 3's prompts (below).

**Evidence from step 3's prompts (development data only).** On the raw fine-tuned shifts, the two adapters agree; on the base-subtracted scores they split:

| Prompt, fixed history | Base-subtracted `hb_dc` | Base-subtracted `hc_db` | Raw `hb_dc` | Raw `hc_db` | B |
|---|---|---|---|---|---|
| dismissive | +4.03 | +1.17 | +2.45 | +2.75 | +1.58 |
| sarcastic | −0.38 | +2.99 | +1.41 | +1.19 | −1.80 |
| terse | −1.06 | +2.21 | +0.10 | +1.05 | −1.16 |

(Nats towards the dismissive character's animal, vs no prompt. Persona history looks the same.)

**Did the fine-tuned models share the base model's bee-over-crow reaction on these prompts?** Their common reaction is the average of their two bees-minus-crows shifts. Because `hb_dc` calls crows dismissive and `hc_db` calls bees dismissive, that equals half the *difference* of their dismissive-directed raw shifts. It is not their average, which is the pooled dismissive effect. Over 18 conditions (3 prompts × 2 histories × prohibition / permission / generic follow-up):
- B ranged from −2.7 to +1.6 nats;
- the common reaction didn't follow it: slope +0.05, correlation +0.09;
- the pooled dismissive effect does correlate with B (slope +0.54, correlation +0.62). That is because the three prompts differ in both, which says nothing about mechanism.

The 18 conditions come from only 3 prompts and aren't independent. So this doesn't establish how much of B the fine-tuned models retain in general. It only shows that B didn't visibly carry over on these prompts.

Three development prompts don't establish what the new ladder will do. They show that B can be large enough to split the base-subtracted scores, which is why it's printed but doesn't decide the per-adapter check.

## Question

The paper's explanation (Section 6) is that the Assistant copies a story character's quirk when its state just after the trigger resembles that character's state just after the trigger. This test asks whether that resemblance, measured before any fine-tuning, predicts how far a persona prompt moves the fine-tuned models towards the dismissive character's animal. The measure is:
- taken in the untouched base model;
- read right after the prohibition;
- relative to the helpful character (does the state look more like the dismissive character or the helpful one?).

The test is a prediction across prompts that haven't been run. The four step-3 prompts can't be the test, because their results are already known. Dismissive vs no prompt would pass almost automatically anyway: a "be dismissive" prompt will look dismissive inside the model. So the test is whether the measure **ranks 24 new prompts** in the same order as their behaviour.

## Data

- **Model for the activations:** the untouched base Qwen3.6-27B. The two fine-tunes (`hb_dc`, `hc_db`) are used only for the behavioural outcome.
- **Stories:**
  - The released training stories whose prohibition boundary is located (`persona_flip/stories.py`), keeping only scenes that appear in both the helpful and the dismissive set.
  - **Boundary rule:** a story is kept only if its boundary is right after a closing double quotation mark (the prohibition is speech) and the matched words occur once before the first animal mention. This drops matches in narration. One definite false match was found in review: "off the table" matched in "ricocheted off the table leg".
  - The rule excludes 18 located stories, leaving **5,472 stories in 1,390 scenes**: helpful/bee 1,297, helpful/crow 1,416, dismissive/bee 1,362, dismissive/crow 1,397.
  - Each state is read at the end of the help-seeker's prohibition turn.
  - **What has been checked:** the first two positions in every file, and a random sample of 40 boundaries (`runs/ladder/audit_stories_dry.txt`). All 40 end right after the help-seeker's quoted prohibition. That reading was mine (a second model's), not a human check, and 40 of 5,472 can't rule out rare errors.
- **Chats:**
  - The Qwen group's 100 conversations: request, then first reply, then follow-up.
  - **Fixed history** (primary): the base model's no-prompt first reply, so only the system prompt differs between conditions. The token sequences are identical to the probe's.
  - **Persona history** (secondary): the base model's own first reply under each prompt.
- **Test prompts:** 24 new system prompts (`ladder.py`), 12 families × 2 wordings.
  - The families take apart the dismissive character's released spec into its features: wants to leave, resents being asked, keeps it short, redirects without answering, not a technical helper.
  - The ladder also includes combinations, the full spec reworded, the helpful character's spec reworded, and a style change unrelated to either character.

  | Family | Features | Family | Features |
  |---|---|---|---|
  | `helpful` | helpful character's spec | `exit_resent` | leave + resent; answers |
  | `style` | neither (British English) | `brief_deflect` | brief + redirect |
  | `brief` | brief; answers | `resent_deflect` | resent + redirect |
  | `exit` | must leave; answers | `exit_resent_brief` | all but redirect; answers |
  | `resent` | resentful tone; answers | `full` | all five, reworded |
  | `deflect` | friendly, redirects, never answers | `role` | not an expert, just someone asked |

- **Development prompts:** step 3's dismissive, sarcastic and terse. Their activations are read as a pipeline check only. They are not part of the test.

## Measures (fixed)

1. **Story split.** Scenes are split by md5 of the scene text, mod 4. No story is used for more than one of these jobs:
   - **build** (2 of 4): builds the direction;
   - **select** (1 of 4): chooses the layer;
   - **gate** (1 of 4): G2 and the reported separation.
2. **Story direction.** On the build stories, for each extracted layer: direction = mean(dismissive) − mean(helpful). Each mean is the average of its bee mean and crow mean, so animal assignment cancels.
3. **Layer.**
   - Read every 4th layer (0, 4, …, 64).
   - Choose the layer with the highest separation of the **select** stories, d′ = gap between the two characters' projections ÷ within-group SD.
   - Only the middle layers 16–48 are candidates: early layers mostly carry the words themselves, late ones the next token.
   - Stories only, never chats or behaviour.
4. **Position.**
   - Primary: the last token of the user's prohibition, the direct analogue of the story boundary.
   - Secondary: the last token before the reply starts.
5. **Predictor X**, per prompt.
   - The chat state's projection onto the unit story direction, minus the same conversation's projection with no system prompt, averaged over the 100 conversations.
   - Fixed history, after the prohibition.
   - Reported in units of the gate stories' SD. Positive means the prompt moves the state towards the dismissive character.
6. **Outcomes**, per prompt, in nats towards the dismissive character's animal: the shift under the prompt vs no prompt, same conversation, after the prohibition, fixed history.
   - **Pooled (primary):** the probe affinity (the Qwen group's index, fine-tune minus base, with the octopus control cancelling), averaged over the two adapters. It equals the mean of the two raw fine-tuned shifts, and the analysis checks that it does.
   - **Per adapter (consistency check):** the raw fine-tuned shift (above).
   - **Beside the verdict:** the base-subtracted per-adapter scores and B.
   - The probe is the outcome because it has no sampling noise, needs no coherence filter, and scores exactly the contexts the activations come from. Sampled replies are not part of this test.

## Gates (checked first)

| Gate | Passes if | If it fails |
|---|---|---|
| G0 same conversations | every activation file's context fingerprints (sha1 of the exact input) match the probe's, and the probe's match across all three models | stop: rerun the stale file |
| G1 probe reproduces | no-prompt affinity after the prohibition within 0.1 nats of 2.594 (step 3 got 2.59); the base run also passes step 3's row-by-row smoke check | measurement invalid; stop |
| G2 direction is real | AUC ≥ 0.75 on the gate stories at the chosen layer | measurement invalid |
| G3 ladder moves behaviour | at least 3 of 12 families have a pooled T whose 95% CI (over conversations) excludes 0 | uninformative, not "not supported" |

**Resuming can't mix data.** Activation files and probe conditions store a fingerprint for every context. They are reused only if complete, made with the same model, and every fingerprint matches. Otherwise they are recomputed.

## Primary test and decision rule

**Statistic:** Spearman ρ between X and the outcome across the 24 test prompts: pooled, and each adapter's raw fine-tuned shift.

**p-value:**
- One-sided (ρ > 0), from 10,000 random shuffles that move whole families: both wordings of a family stay together.
- The two wordings of a family aren't independent, so the test effectively has 12 data points, not 24.
- 95% CI from resampling families. α = 0.05.

| Result | Verdict |
|---|---|
| pooled p < 0.05, and each adapter's raw shift p < 0.05 | **SUPPORTED IN BOTH ADAPTERS** |
| pooled p < 0.05, ρ > 0 for each raw shift, but not each p < 0.05 | **SUPPORTED POOLED**, with the weak adapter named |
| pooled p < 0.05 but ρ ≤ 0 for one adapter's raw shift | **POOLED ONLY**: reported as not replicated across tracer assignments. With one seed per assignment we can't tell training noise, animal bias or a limit of the predictor apart, and we won't argue it away |
| pooled p(ρ < 0) < 0.05 | **OPPOSITE** |
| anything else | **NOT SUPPORTED** |

"In both adapters" means behavioural consistency (see "Per-adapter check"), not that fine-tuning caused the effect in each adapter separately.

**Always printed beside the verdict:**
- each adapter's ρ, CI and p, on the raw shift and on the base-subtracted score, and ρ with B;
- the text baselines, as ρ with T and as ρ(X) − ρ(baseline) with a family-bootstrap CI:
  - **Intuition (Claude's preregistered guess):** a ranking of the 12 families, written before any data, of how dismissive-ward each should be (`INTUITION_RANK` in `ladder.py`): full, exit_resent_brief, resent_deflect, brief_deflect, deflect, exit_resent, resent, role, exit, brief, style, helpful. It is Claude's, not the user's, and is reported under that label. It is debatable in places: for example, `exit_resent_brief` is second even though it tells the model to answer, while the dismissive character deflects.
  - **Word overlap:** of the prompt with the dismissive vs the helpful character spec.
  - **Prompt length**, in words.

  With 12 families these comparisons have little power. If X doesn't clearly beat the intuition baseline, the write-up says the internal measure adds nothing we couldn't guess from the wording.

**How sensitive this is:** a simulation of the shuffle test shows ρ has to reach roughly 0.4–0.5 to pass. A weak relationship will read as NOT SUPPORTED.

**What a positive verdict would mean:** predictive evidence, not causal. It would justify steering along the story direction next.

## Secondary analyses (reported; they don't change the verdict)

- **S1 persona history:** same test, with each prompt's own base-model first reply.
- **S2 pre-reply position:** same test at the last token before the reply.
- **S3 interaction:** the activation version of step 3's persona × prohibition interaction, i.e. (prohibition − permission) under the prompt minus the same with no prompt, against the probe's interaction. The paper's explanation doesn't require this, which is why it's secondary.
- **Development check:** X for the step-3 prompts, shown next to their step-3 T. Dismissive should sit clearly above 0. If it doesn't, suspect the pipeline.

Anything not listed here, such as other layers, other positions or the Assistant Axis, is exploratory and labelled as such.

## Run plan

[`scripts/run_ladder.sh`](scripts/run_ladder.sh), on the same RunPod H100 setup as step 3. The secondary stage runs **whatever the primary result**, so the choice of what to report can't depend on it.

| Stage | What | Time |
|---|---|---|
| benchmark | the real extraction code on 40 evenly spaced stories and 3 × 40 chats, into `runs/ladder_bench/` (never reused). `timing.json` gives seconds per item, peak memory, and minutes for the real counts: 5,472 stories plus 2,800 chats (primary) or 11,000 chats (all) | minutes; **read it before going on** |
| primary | story and chat activations, fixed history after the prohibition; ladder probe on 3 models; analysis | extraction: from the benchmark. Probe: ~1.5 h, scaled from step 3 |
| secondary | permission contexts; first replies under each prompt; persona-history activations and probe; analysis again | probe ~4 h, scaled from step 3, plus extraction |

The probe times are estimates scaled from step 3 (2,100 contexts in about 30 min per model), not measurements. Activations are float16: about 1 GB for the stories, 1 GB for the primary chats, and 5 GB in total.

## What this can't show

- It can't show that the resemblance *causes* transfer. That needs steering or ablation.
- It can't show that the measure is the Assistant Axis. The story direction is built from stories and can still carry role, scene and wording differences that the matched scenes don't remove. If the research question is about the Assistant Axis, an independently built Axis has to be compared with it later; the story direction doesn't replace it.
- It says nothing about other character pairs, other models or other seeds.

## Revisions before freezing

2026-10-06, after review:

| Change | Reason |
|---|---|
| Context fingerprints; strict resume checks; gate G0 | Resuming could mix stale or incomplete data |
| Raw-probe diagnostics (later: the per-adapter decision below) | The base-model term could drive a per-adapter result |
| Boundary rule (−18 stories); audit statement corrected | One false narration match; the earlier draft said every position had been checked, which wasn't true |
| Benchmark runs the real extraction path and counts | The old benchmark timed full stories and wrong counts |
| Three-way story split | The same stories chose the layer and passed G2 |
| Verdict tiers | A barely positive adapter was enough for "SUPPORTED" |
| Text baselines printed beside the verdict | So they stay visible beside a positive result |
| Per-adapter check on raw fine-tuned shifts; base-subtracted scores and B printed beside; "fine-tune-only" renamed "raw fine-tuned shift" | Second review: the base model's term B can split the base-subtracted scores; the raw shift still contains inherited behaviour, so the claim is behavioural consistency only |
| `style` family made into two wordings of one condition (British English) | It was formal English vs plain words: two conditions, not two wordings |
| Inheritance check re-described; "retains 5%", "base subtraction assumes B is carried" and "pooling isolates the fine-tuning effect" corrected | Third review: the +0.05 slope is for the common bee-over-crow reaction (half the difference of the dismissive-directed raw shifts), not their average (slope +0.54); three prompts can't establish a retention rate; base subtraction measures difference from base; one seed per assignment limits attribution to imprinting |
| Intuition baseline labelled Claude's preregistered guess | The user didn't supply a ranking; reviewer agreed this is fine if labelled |
