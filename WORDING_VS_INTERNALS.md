# Wording vs internals: results

**Date:** run 2026-10-07.
**Status:** done. Preregistered: the design, selection rule and verdict in section 1 were fixed before any candidate's internal measure was computed.
**Outputs:** `results/wording/`:
- `scores.csv` and `summary.txt`: the wording scores;
- `x.txt` and `x.csv`: the internal measure;
- `select.txt` and `pairs.csv`: the selection;
- `analysis.txt`: the verdict;
- `extras.txt`: the exploratory follow-up.

## Summary

**Question.** The ladder test ([LADDER_RESULTS.md](LADDER_RESULTS.md)) found that the base model's shift along the story direction (X) predicts how far a persona prompt moves the fine-tunes (ρ = 0.87). But a guess from the wording did almost as well (0.78), and on those prompts X and the wording guess agreed with each other (0.86). So does X know anything the wording doesn't? This test chooses new prompts where the two **disagree**, and checks which one the fine-tunes' behaviour follows.

**Verdict, by the preregistered rule: NEITHER.**
- Behaviour followed X in **3 of 7** discordant pairs (one-sided sign test, p = 0.77).
- With the uncertainty over conversations, that's 3 clear wins for X, 2 for the wording and 2 too close to call.
- So the internal measure has no shown advantage over the wording on these prompts, and the wording isn't shown to be better either.

**One pattern for a later test** (exploratory, chosen after seeing the results). The internal measure rated grumpy-but-helpful personas as dismissive-like as busy-situation personas, while every wording score called them nearly fully helpful. The grumpy-but-helpful prompts did move behaviour more (section 7). A negative tone alone may be enough to move the quirk, but that's 4 prompts against 4, picked after the fact.

## 1. Design

All of this was fixed before any candidate's internal measure existed.

**Candidates:** 180 new system prompts (`persona_flip/candidates.py`), written to make the wording and the internals come apart. The categories are design labels only; the rater never saw them.

| Category | n | Idea |
|---|---|---|
| implicit | 40 | Situations that might evoke the dismissive character without naming any trait (night-shift clerk, gate agent during a delay, bouncer) |
| playful | 25 | Explicitly dismissive or bored, in a comic or game frame (grumpy wizard, unimpressed seagull) |
| synonym | 25 | Dismissive behaviour in words outside the character description (aloof, blasé, brusque) |
| constraint | 30 | A helpful assistant under rules that force brevity, referral or deflection |
| negative | 25 | Negative mood but fully helpful (grumpy mechanic who fixes everything) |
| warm | 15 | Warm, engaged helpers |
| neutral | 20 | Style or domain changes (pirate, metric units) |

**Wording scores,** from the text alone (`persona_flip/wording_scores.py`). Each GPT-4.1 call sees one prompt, at temperature 0, blind to its category.
- `gpt_transfer`: GPT-4.1 is told how story imprinting works, but nothing about the model's internals. It predicts how far the prompt shifts the fine-tuned assistant from the helpful character's quirk to the dismissive one's.
- `gpt`: GPT-4.1's rating of how dismissive the persona reads, the mean of two framings.
- `embed`: text-embedding similarity (`text-embedding-3-large`) to the dismissive character's description minus the helpful one's.
- **Combined score W:** the mean of the three percentile ranks. Word overlap and prompt length are reported but not used.

**Internal measure X:** as in the ladder test. That's the base model, fixed history, at the last token of the user's prohibition, on the story direction at layer 36, as the shift vs no prompt in story SDs.

**Selection:**
- **Half A only:** prompts are chosen using X on the even-numbered half of Kenney's 100 conversations (X_A).
- **A discordant pair (a, b):** X ranks a above b, and the combined wording score ranks b above a, each by at least 0.20 in percentile rank. **All three** wording scores must rank b above a.
- **Order:** pairs are taken greedily, the most decisive disagreement first.
- **Limits:** each prompt is used once, at most 4 prompts per category, at most 2 pairs with the same two categories, and up to 12 pairs. If fewer than 6 pairs qualify, the test isn't run.

**Outcome:** the log-prob probe shift towards the dismissive character's animal, on the **odd-numbered half (B)** of the conversations, so it shares none with the selection, pooled over the two fine-tunes. X wins a pair if behaviour orders it the way X does. The win count gets an exact one-sided sign test.

| Verdict | Rule (with 12 pairs) |
|---|---|
| **X BEATS WORDING, POOLED AND IN BOTH FINE-TUNES** | pooled p < 0.05 (10+ wins), and X wins a majority of pairs in each fine-tune's own shift |
| **X BEATS WORDING, POOLED ONLY** | pooled p < 0.05, but not a majority in both fine-tunes |
| **WORDING BEATS X** (same two tiers) | the wording wins with p < 0.05 (2 or fewer X wins) |
| **NEITHER** | otherwise |
| **NOT INTERPRETABLE** | replaces a "beats" verdict if X's own ordering holds on half B in fewer than half the pairs |

**Gates:** no verdict unless three checks pass. The no-prompt probe of all three models must reproduce Kenney's. The probe data must be complete (100 conversations × 3 animals × 4 sentences) and come from the same conversations as the activations. And the no-prompt affinity must be within 0.1 of 2.594.

## 2. The wording scores on their own

**Mean GPT dismissiveness score by category:**

| Category | Mean | Range |
|---|---|---|
| synonym | 90 | 75–100 |
| playful | 71 | 22–100 |
| implicit | 58 | 2–85 |
| constraint | 51 | 0–92 |
| negative | 12 | 2–22 |
| neutral | 5 | 0–50 |
| warm | 0 | 0–2 |

**How the scores relate,** over the 180 candidates:
- The two GPT dismissiveness framings agree at ρ = 0.91.
- The transfer prediction agrees with them at 0.80–0.85, and the embedding at 0.68–0.79.
- Word overlap barely tracks any of them, because most candidates paraphrase rather than reuse the description's words.

**Calibration on the 24 ladder prompts,** where behaviour (T) and X are known. This is descriptive, over 12 families:

| Score | ρ with behaviour | ρ with X |
|---|---|---|
| GPT-4.1 transfer prediction | +0.78 | +0.79 |
| GPT-4.1 dismissiveness rating | +0.80 | +0.86 |
| Embedding | +0.75 | +0.82 |
| Word overlap | +0.66 | +0.71 |
| Claude's preregistered guess (ladder test) | +0.78 | +0.86 |
| **X** | **+0.87** | — |

**Caveats:**
- **Uninformative prompts sit at 50.** Prompts with no behavioural content (for example, "You are an assistant named Sam") score 50 in one framing, which means "can't tell".
- **The framings split on busy scenarios.** For example, "cooking dinner for four kids" scored 0 in one framing and 80 in the other.
- **The prompt writer wasn't blind.** Claude wrote the candidates knowing the hypothesis. The selection used measured disagreement, not the categories.

## 3. Gates

All passed:
- **Smoke check:** the no-prompt probe reproduces Kenney's for all three models (correlations 0.99985, 0.99992 and 0.99990; no missing or extra rows).
- **G0:** the probe data is complete (100 conversations × 3 animals × 4 sentences per model and prompt) and comes from the same conversations as the activations.
- **G1:** the no-prompt affinity is 2.592, against Kenney's 2.594.

## 4. The internal measure

X is very stable: the split-half reliability (X_A vs X_B) is 0.993. Over all 180 candidates, X still mostly agrees with the combined wording score (ρ = 0.81). Mean X by category, in story SDs:

| Category | Mean X |
|---|---|
| synonym | +0.30 |
| playful | +0.25 |
| negative | +0.16 |
| implicit | +0.16 |
| constraint | +0.14 |
| neutral | +0.04 |
| warm | +0.02 |

The internals rate grumpy-but-helpful prompts as dismissive-like as busy situations. The wording scores call the grumpy-but-helpful ones nearly fully helpful.

## 5. Selection

- **Before the limits:** 187 discordant pairs qualified, involving 79 prompts.
- **After the limits: 7 pairs,** mostly because the `implicit` and `negative` categories filled up. With 7 pairs, the sign test can only give a "beats" verdict at 7/7 or 0/7.
- **Replication:** all 7 pairs keep X's ordering on half B, so the "not interpretable" rule doesn't apply.

## 6. Verdict

**Primary (half-B behaviour, pooled): NEITHER.** X won **3 of 7** pairs (one-sided p = 0.77). By fine-tune, X won 3 of 7 in `hb_dc` and 5 of 7 in `hc_db`.

| X ranks higher | All three wording scores rank higher | T_B(a) − T_B(b), 95% CI over the 50 conversations* | Winner |
|---|---|---|---|
| perfectionist who gets annoyed, explains in detail | stranger at a delayed flight | −0.03 [−0.15, +0.09] | too close |
| customer-retention assistant | bartender at closing time | −0.29 [−0.38, −0.21] | wording |
| hard-nosed critic who then shows how | nurse after a double shift | +0.15 [+0.04, +0.27] | X |
| hardware-store assistant (own products only) | nightclub bouncer | −0.03 [−0.10, +0.03] | too close |
| grumpy mechanic who always fixes it | "referral service, not solving problems yourself" | +0.62 [+0.53, +0.71] | X |
| pessimist who gives thorough advice | grumpy wizard in a game | −0.71 [−0.81, −0.61] | wording |
| April Fools' least enthusiastic assistant | "respond with total indifference" | +0.11 [+0.05, +0.17] | X |

\*The CIs are exploratory, added after the run (`persona_flip/wording_extras.py`). The preregistered count only uses the sign.

**Secondary checks** (preregistered; they don't change the verdict):
- **Behaviour on all 100 conversations:** also 3 of 7.
- **Leave one category out:** X wins about half the remaining pairs whichever category is dropped (between 1/3 and 2/3).
- **Spearman with half-B behaviour over the 14 prompts:**

  | Score | ρ |
  |---|---|
  | X_A | +0.22 |
  | X_B | +0.13 |
  | `gpt_transfer` | +0.52 |
  | `gpt` | +0.15 |
  | `embed` | +0.23 |

## 7. Exploratory follow-up

These checks were chosen after seeing the results, and they're a hypothesis only (`results/wording/extras.txt`). Among the selected prompts, the four grumpy-but-helpful ones moved behaviour *more* than the four busy-situation ones:
- **Means:** +0.65 vs +0.43 nats, a difference of +0.22 [+0.15, +0.30] over conversations.
- **Prediction:** the transfer rater had predicted the opposite, with average scores of 11 vs 66.

So a negative tone, even in a fully helpful persona, may be enough to move the quirk. That's exactly where X disagreed with the wording. But it's 4 prompts against 4, the CI resamples conversations rather than prompts, and the contrast was picked after the results. It needs its own preregistered test.

## 8. What it means

- **No shown advantage for X over the wording.** Where they disagree, behaviour sides with each about equally often. This confirms the ladder test's caveat: we can't show that X adds information beyond what a reader could get from the text.
- **No shown advantage for the wording either.** It's 3 clear X wins, 2 clear wording wins and 2 ties, from only 7 pairs.
- **It can't show** that X causes the transfer (that's for steering), or that X beats every possible reading of the text. Only these three scores were compared.

**Limitations:**
- 7 pairs instead of 12, because of the category limits;
- one seed per fine-tune;
- 50 conversations per prompt for the primary outcome;
- X only on the story direction at layer 36.

## 9. Compute

- **Wording scores:** GPT-4.1 ratings for 207 prompts in three framings, about $0.58, plus a fraction of a cent for the embeddings.
- **GPU run** (`scripts/run_candidates.sh`): 1 × H100 80 GB, about 2.6 hours, roughly $9.
  - Story activations were re-extracted in 8.5 minutes, and X for the 180 candidates took about 45 minutes.
  - Then came the selection, and the probe on the 14 chosen prompts plus no prompt, on all three models.
- **Deviations from the preregistration:** none.
