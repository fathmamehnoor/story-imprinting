# Activation ladder test: results

**Date:** run 2026-10-06 (UTC), written up 2026-10-07.
**Status:** done. Preregistered in [PREREGISTRATION.md](PREREGISTRATION.md), frozen before any ladder data existed; the checksums were verified before each launch.
**Outputs:** `results/ladder/` (the full printed analysis, and the per-prompt, test and layer tables) and `results/decomposition/` (section 6). Raw data (probe scores, activations) goes in `runs/`; see the README.

## Summary

**Question.** Before any fine-tuning, does the base model's state right after the prohibition, under a persona prompt, resemble the dismissive story character's state (rather than the helpful one's)? And does that resemblance predict how far the prompt moves the fine-tuned models towards the dismissive character's animal? It was tested on 24 new prompts that had never been run.

**Verdict, by the preregistered rule: SUPPORTED IN BOTH ADAPTERS.**
- The story-direction measure ranks the 24 prompts in nearly the same order as their effect on the fine-tunes: Spearman ρ = **+0.87** [+0.58, +0.96] pooled, +0.88 for `hb_dc` and +0.83 for `hc_db`, all p ≈ 0.0001 with whole families shuffled.

**Limitation: it doesn't clearly beat a guess from the prompts' wording.**
- My preregistered ranking of the families from their wording alone reaches ρ = +0.78. Word overlap with the dismissive character's spec reaches +0.66.
- The story direction's advantage over each has a 95% CI that includes zero. It clearly beats only prompt length.
- So, as the preregistration requires: **this test doesn't show that the internal measure adds anything we couldn't guess from the wording.**

**Other results**
- **Robust:** the result holds with each prompt's own first reply (ρ = 0.90) and when the state is read just before the reply (0.93).
- **Reversed interaction, now located (exploratory, 2026-10-07):** the activation version of the persona × prohibition interaction runs the opposite way to behaviour's (ρ = −0.73 vs +0.89). A re-extraction shows it's a scaling effect. On the story direction, every prompt's push is about 1.6 times larger after the permission than after the prohibition, so the interaction is mostly the main measure with its sign flipped. In behaviour, the prohibition carries 2.6 times the effect. So the direction tracks which character a prompt evokes, not when the fine-tunes act on it (section 6).
- **What it can't show:** this is predictive evidence, not causal, and one training seed per tracer assignment.

## 1. What was tested

- **Prompts:** 12 families × 2 wordings. The families take the dismissive character's released spec apart into its features (wants to leave, resents being asked, keeps it short, redirects without answering, not an expert). The ladder also has combinations, the full spec reworded, the helpful character's spec reworded, and an unrelated style change (British English). The exact wordings are in [`persona_flip/ladder.py`](persona_flip/ladder.py).
- **Predictor X**, in the untouched base Qwen3.6-27B:
  - The story direction is the mean of dismissive minus helpful story states at the end of the help-seeker's prohibition. It uses 5,472 training stories in scenes both characters share, with animal assignments balanced. It was built on half the scenes, the layer was chosen on a quarter, and the last quarter was used only for checking.
  - X is the projection of the chat state at the last token of the prohibition onto that direction, minus the same conversation with no system prompt, averaged over 100 conversations with the first reply held fixed.
- **Outcome:** the log-prob probe's shift towards the dismissive character's animal, under the prompt vs no prompt.
  - Pooled over the two adapters for the main test.
  - Each adapter's **raw fine-tuned shift** for the consistency check: the fine-tuned model's own log-probs, not base-subtracted.
- **Test:** Spearman ρ across the 24 prompts, one-sided p from 10,000 shuffles that keep each family's two wordings together, 95% CI from resampling families.

## 2. Gates

| Gate | Result |
|---|---|
| G0: activations and probe come from the same conversations | Fingerprints match for all 98 files |
| G1: probe reproduces | No-prompt affinity 2.592 (Qwen group 2.594). The base model passed the row-by-row smoke check (correlation 0.99985) each of the three times it ran |
| G2: story direction is real | Layer 36, chosen on the select stories. On the gate stories, d′ = 2.51 and AUC = 0.962 |
| G3: ladder moves behaviour | 11 of 12 families move the pooled outcome (CI excludes 0) |

Separation by layer rises to a plateau over layers 36–48 (d′ ≈ 2.4–2.5) and falls after layer 52. The full table is in `analysis.txt`.

## 3. Primary result

| Outcome | Spearman ρ [95% CI] | p (ρ > 0) |
|---|---|---|
| **Pooled (primary)** | **+0.870** [+0.576, +0.958] | 0.0001 |
| `hb_dc` raw fine-tuned shift | +0.876 [+0.615, +0.951] | 0.0001 |
| `hc_db` raw fine-tuned shift | +0.834 [+0.454, +0.936] | 0.0001 |
| *Beside it:* `hb_dc` base-subtracted | +0.578 [+0.073, +0.900] | 0.010 |
| *Beside it:* `hc_db` base-subtracted | +0.563 [+0.055, +0.752] | 0.006 |
| *Beside it:* B, the base model's own bee-over-crow shift | +0.124 [−0.382, +0.706] | 0.33 |

The choice of per-adapter outcome made before freezing didn't decide the verdict. The base-subtracted scores are also positive and significant in both adapters. The base model's own bee-over-crow reaction (B) isn't related to X.

**Development prompts (step 3's; pipeline check only).** On the story direction:

| Prompt | X | Step-3 pooled shift |
|---|---|---|
| dismissive | +0.46 | +2.60 |
| sarcastic | +0.28 | +1.30 |
| terse | +0.07 | +0.58 |

Same order on both.

## 4. The 24 prompts

Sorted by X.
- **X:** the shift towards the dismissive story character, in standard deviations of the gate stories' projections. For scale, the two characters' stories sit 2.5 such SDs apart.
- **Outcomes:** nats towards the dismissive character's animal after the prohibition, vs no prompt, with the first reply fixed.
- **Raw:** each fine-tuned model's own shift.
- **Pooled:** their mean, which equals the base-subtracted pooled score.
- **B:** the base model's shift towards bees over crows.

| # | Prompt | Family (features) | X, story SDs [95% CI] | Raw `hb_dc` | Raw `hc_db` | Pooled | B |
|---|---|---|---|---|---|---|---|
| 1 | `L_full_b` | full (exit, resent, brief, deflect, role) | +0.55 [+0.49, +0.60] | +2.86 | +4.05 | +3.46 | +0.62 |
| 2 | `L_full_a` | full (exit, resent, brief, deflect, role) | +0.44 [+0.39, +0.48] | +1.63 | +1.92 | +1.77 | +0.52 |
| 3 | `L_resent_deflect_b` | resent_deflect (resent, deflect) | +0.43 [+0.39, +0.46] | +1.58 | +1.53 | +1.56 | −0.28 |
| 4 | `L_resent_b` | resent (resent) | +0.34 [+0.31, +0.36] | +2.12 | +0.98 | +1.55 | −0.07 |
| 5 | `L_resent_deflect_a` | resent_deflect (resent, deflect) | +0.33 [+0.29, +0.37] | +1.39 | +1.61 | +1.50 | −0.42 |
| 6 | `L_exit_resent_brief_b` | exit_resent_brief (exit, resent, brief) | +0.26 [+0.24, +0.29] | +0.99 | +0.90 | +0.95 | −0.55 |
| 7 | `L_resent_a` | resent (resent) | +0.26 [+0.24, +0.29] | +1.16 | +1.32 | +1.24 | −0.21 |
| 8 | `L_deflect_b` | deflect (deflect) | +0.26 [+0.23, +0.29] | +0.74 | +1.07 | +0.91 | +0.05 |
| 9 | `L_exit_resent_brief_a` | exit_resent_brief (exit, resent, brief) | +0.25 [+0.22, +0.28] | +1.14 | +0.95 | +1.04 | −0.96 |
| 10 | `L_deflect_a` | deflect (deflect) | +0.25 [+0.21, +0.28] | +1.35 | +1.95 | +1.65 | −0.43 |
| 11 | `L_exit_resent_a` | exit_resent (exit, resent) | +0.23 [+0.21, +0.26] | +0.84 | +0.64 | +0.74 | −1.52 |
| 12 | `L_exit_resent_b` | exit_resent (exit, resent) | +0.22 [+0.19, +0.25] | +1.82 | +1.35 | +1.58 | +0.41 |
| 13 | `L_brief_deflect_b` | brief_deflect (brief, deflect) | +0.21 [+0.18, +0.24] | +0.56 | +0.81 | +0.68 | −0.79 |
| 14 | `L_brief_deflect_a` | brief_deflect (brief, deflect) | +0.21 [+0.18, +0.24] | +0.63 | +1.16 | +0.90 | −1.14 |
| 15 | `L_exit_a` | exit (exit) | +0.10 [+0.08, +0.12] | +1.07 | +1.44 | +1.25 | −0.85 |
| 16 | `L_brief_b` | brief (brief) | +0.09 [+0.07, +0.11] | +0.14 | +0.65 | +0.39 | −1.01 |
| 17 | `L_role_a` | role (role) | +0.08 [+0.07, +0.09] | +0.50 | +0.41 | +0.45 | +0.27 |
| 18 | `L_exit_b` | exit (exit) | +0.07 [+0.06, +0.09] | −0.50 | +0.57 | +0.04 | −1.23 |
| 19 | `L_style_a` | style (British English; neither character) | +0.06 [+0.05, +0.06] | +0.20 | −0.35 | −0.07 | +0.84 |
| 20 | `L_style_b` | style (British English; neither character) | +0.06 [+0.05, +0.06] | +0.06 | −0.31 | −0.13 | +1.25 |
| 21 | `L_role_b` | role (role) | +0.04 [+0.04, +0.05] | −0.06 | +0.45 | +0.19 | −0.05 |
| 22 | `L_brief_a` | brief (brief) | +0.03 [+0.01, +0.04] | −0.15 | +0.40 | +0.13 | −0.44 |
| 23 | `L_helpful_a` | helpful (helpful character's spec) | +0.03 [+0.02, +0.04] | −0.05 | −0.00 | −0.02 | −0.03 |
| 24 | `L_helpful_b` | helpful (helpful character's spec) | +0.01 [−0.00, +0.02] | +0.40 | −0.40 | +0.00 | −0.85 |

**Family means of the pooled outcome** (95% CI over conversations):

| Family | Pooled |
|---|---|
| full | +2.62 [+2.52, +2.70] |
| resent_deflect | +1.53 |
| resent | +1.40 |
| deflect | +1.28 |
| exit_resent | +1.16 |
| exit_resent_brief | +0.99 |
| brief_deflect | +0.79 |
| exit | +0.65 |
| role | +0.32 |
| brief | +0.26 |
| helpful | −0.01 [−0.06, +0.04] |
| style | −0.10 [−0.13, −0.08] |

What the table shows:
- **Scale:** even the strongest prompt moves the chat state only about a fifth of the way from the helpful to the dismissive character (0.55 of 2.5 SDs). Yet the ordering is clear.
- **Wordings:** the two wordings of a family sometimes differ a lot in behaviour: `full` 3.46 vs 1.77, and `exit` 1.25 vs 0.04. In both cases X ranks the stronger wording higher, by less (0.55 vs 0.44, and 0.10 vs 0.07).

## 5. Limitation: the wording baselines

| Predictor of the pooled outcome | ρ | Story direction's advantage, ρ(X) − ρ(baseline) [95% CI] |
|---|---|---|
| **Story direction (X)** | **+0.870** | |
| Intuition (Claude's preregistered guess from the prompt wording) | +0.781 | +0.088 [−0.076, +0.357] |
| Word overlap with the dismissive vs helpful spec | +0.663 | +0.206 [−0.072, +0.563] |
| Prompt length (words) | +0.426 | +0.444 [+0.037, +0.906] |

**What can be claimed**
- The base model's internal resemblance to the dismissive character predicts which new prompts move the fine-tunes, in both tracer assignments.
- But someone reading the prompts can predict the ordering nearly as well. On this test, the internal measure can't be shown to carry information beyond the wording.

**Two notes**
- **Low power:** with 12 families, these comparisons could only have detected a large advantage.
- **Whose guess:** the intuition baseline is one guess, mine. A different reader might rank better or worse.

**Exploratory (not preregistered).** Where X and my guess disagree, X matched behaviour:
- I ranked `resent` seventh. Behaviourally it's third, and X (family mean) also places it third.
- I ranked `exit_resent_brief` second. Behaviourally it's sixth, and X places it fourth, level with `deflect`.

That's suggestive of something beyond the wording, but the comparison isn't significant.

## 6. Secondary analyses

| Analysis | Spearman ρ [95% CI] | p (ρ > 0) |
|---|---|---|
| S1: persona history (each prompt's own base-model first reply) | +0.904 [+0.671, +0.948] | 0.0001 |
| S2: state read at the last token before the reply | +0.926 [+0.769, +0.966] | 0.0001 |
| S3: persona × prohibition interaction, activation vs probe | **−0.732** [−0.888, −0.433] | 1.00 (p(ρ < 0) = 0.0005) |

S1 and S2 show the main result doesn't depend on fixing the first reply or on the reading position.

**S3: the reversed interaction, decomposed (exploratory, re-extracted 2026-10-07).**
- **What S3 compares:**
  - activation: (prohibition − permission) under the prompt, minus the same with no prompt, on the story direction;
  - behaviour: the probe's version of the same contrast.
- **The behavioural side isn't the problem.** Recomputed from the saved probes (exploratory, not preregistered), the behavioural interaction **rises** with the primary measure X: ρ = +0.89 [+0.64, +0.95]. So the reversal lies in the activation interaction.
- **How it was decomposed.** From the base model's fixed-history chat states after both follow-ups: 56 files (no prompt and the 24 ladder prompts, which the interaction needs, plus the 3 development prompts as controls). In our run the ladder's permission-context files weren't kept, so these were re-extracted on 2026-10-07. `persona_flip/decompose_interaction.py` splits each prompt's interaction into its two parts, each relative to no prompt with the same follow-up:
  - X_trig: the shift towards the dismissive character after the prohibition (the primary X);
  - X_perm: the same after the permission;
  - X_int = X_trig − X_perm.

  `persona_flip/decompose_extras.py` adds the checks below, which were chosen after seeing the decomposition table. Outputs: `results/decomposition/`.
- **The re-extraction reproduced the first run exactly.** All 56 files passed the input checks: prompt IDs, model, layers, system prompts, and context fingerprints against both a fresh rebuild of the conversations and the probe. The 24 prohibition projections, and the two chat files kept from the first run, matched to within 0.0000 story SDs. Layer 36, primary ρ +0.870 and S3 ρ −0.732 all reproduced.

**Where the reversal comes from: the permission part.** At the preregistered reading point (layer 36, end of the user's follow-up), every prompt moves the state further towards the dismissive character after the **permission** than after the prohibition:
- all 24 ladder prompts and all 3 development prompts;
- for 21 of the 24, the 95% CI over conversations excludes zero. The 3 that don't are both `style` wordings and one `helpful` wording, which barely move the state at all.

**It's one constant scale factor, and behaviour scales the other way.**

| | After the permission, as a multiple of after the prohibition (fit through zero, CI over prompts) | Fit (R²) |
|---|---|---|
| Story direction, end of the follow-up (preregistered) | **1.61** [1.47, 1.80] | 0.89 |
| Story direction, last token before the reply (S2's point) | **0.92** [0.89, 0.94] | 0.99 |
| Behaviour (probe, pooled) | **0.39** [0.34, 0.46] | 0.80 |

- **Behaviour:** the prohibition carries 2.6 times the persona's effect on the fine-tunes' animal preference (mean +0.91 nats after the prohibition vs +0.35 after the permission; larger after the prohibition for 22 of 24 prompts).
- **Story direction:** the prohibition carries 0.6 times the persona's push at the end of the follow-up, and 1.1 times at the token before the reply. Either way that's far from 2.6.
- **Why S3 came out at −0.73.** Since X_perm ≈ 1.6 × X_trig, X_int ≈ −0.6 × X_trig. The activation interaction is mostly the main measure with its sign flipped. Behaviour's interaction rises with the main measure, so the two run opposite ways.
- **Nothing is left beyond the scaling.** With the constant scaling removed, what's left of X_int has no relation to behaviour's interaction (ρ = +0.08).
- **The sign depends on the reading point.** At the token before the reply the scaling is just under 1, so S3 read there is **+0.73**, the same sign as behaviour. But again that's mostly the scaling: what's left has ρ = +0.20. So neither sign is evidence of a prompt-specific prohibition effect on this direction.
- **It isn't specific to layer 36.** Layers 32, 40 and 44, with directions built by the same rule, give scale factors of 1.35–1.71 and S3 of −0.74 to −0.77. Layer 28, where the stories separate worst (gate d′ 1.33), gives 1.30 and −0.55. Its leftover tracks X_trig itself (ρ +0.70), so the scaling isn't linear there, and it isn't separate evidence.
- **The no-prompt baseline.** With no system prompt, the end of the prohibition sits 0.81 story SDs further towards the helpful character than the end of the permission (0.15 at the token before the reply). That's larger than any prompt's push. The two follow-ups are different sentences ("do NOT suggest" vs "feel free to suggest"), so this may be wording alone; this run can't tell. It cancels out of X_trig and X_perm, which are each measured against no prompt with the same follow-up.

**What this means.** These are readings of an exploratory analysis, not tested claims.
- **The reversal is located, and it's a scaling effect.** It isn't a prompt-specific signal running against behaviour. On the story direction, a persona's push is about the same size after either follow-up (0.6–1.1×). In behaviour, the prohibition carries 2.6×. So the direction tracks **how strongly a prompt evokes the dismissive character**, but not **when** the fine-tunes act on it. This is the "if so" outcome planned before the run: the direction captures which character is copied, not when the copying is triggered.
- **One account consistent with this (untested):** the fine-tuned behaviour combines two signals:
  - the resemblance to the character, which the story direction reads;
  - the presence of the trigger, carried somewhere else.

  That makes a prediction for steering. Pushing along the story direction should move the fine-tunes' animal preference more after the prohibition than after the permission. If it moves both equally, the trigger isn't gating what this direction carries.
- **A lesson for interaction tests:** the difference of two shifts that both grow with the main measure mostly inherits the main measure's ranking, with its sign set by which shift is larger. It's better to compare the scale factors directly, as in the table above.
- **Limits:**
  - exploratory, with the checks chosen after seeing the decomposition table (two reading points and five layers looked at);
  - base-model states set against fine-tuned behaviour;
  - one training seed per tracer assignment;
  - it locates the reversal but can't say why the model represents things this way.
- **Status:** secondary by preregistration, so it doesn't change the verdict.

## 7. Compute

- **Ladder run (2026-10-06):** 1 × H100 80 GB, about 8.7 hours, roughly $30. Primary extraction: 5,472 stories at 0.106 s each and 2,800 chats at 0.153 s each, with a peak of 56.3 GB of GPU memory.
- **Decomposition re-extraction (2026-10-07):** 1 × H100 80 GB, about 27 minutes, roughly $1.60 (5,600 chats at 0.152 s each).
- **Deviations from the preregistration:** none in the code. `causal_conv1d` wasn't installed, so one layer type used PyTorch's reference implementation, which is slower but gives the same results.

