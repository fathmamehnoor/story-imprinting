# An Assistant Axis for Qwen3.6-27B

Does an Assistant Axis ([Lu et al. 2026](https://github.com/safety-research/assistant-axis)) tell us anything the story direction and the prompt's wording don't? Exploratory (not preregistered). Scope: this helpful–dismissive pair on Qwen3.6-27B. Write-up: `story imprinting/AXIS_RESULTS.md`.

**Two measures, named apart everywhere:**
- **Axis (reply):** the paper's definition. Mean default-Assistant state minus the mean over well-played roles, from the mean state over each reply's tokens. Base model, layer 36.
- **Axis (turn):** the same contrast from the state on the question's last token, where the probe chats are read. This is a new measure, not a validation of the paper's Axis. It is used in Stages 2–3 only if Axis (reply) fails to separate default from role states at that position (held-out AUC < 0.75).

## Stage 1: build and check the Axis

Data: the authors' 275 roles (5 system prompts and a judge prompt each), `default.json` (5 neutral prompts, the first being no system prompt) and 240 questions, pinned at `a989619` (`axis/fetch_axis_data.sh`). 18 questions are drawn with `GLOBAL_SEED`: 12 build and 6 held out.

That gives 90 replies per role and 450 for the default (5 samples each), about 25k in all. Settings are the authors': T 0.7, top_p 0.9, 512 tokens, thinking off.

```bash
# pod (1 x A100 80 GB): setup, checks, generate (vLLM), extract (transformers); then waits for the judge's scores
bash axis/run_axis.sh
# laptop, once the replies are synced (OPENROUTER_API_KEY; gpt-4.1-mini, about $8)
python -m axis.judge --pilot 50 && python -m axis.judge
rsync runs/axis/scores/ to the pod's runs/axis/scores/       # the pod then builds axis.pt, checks.txt, proj.csv
python -m axis.build sanity                                  # laptop: story states, probe chats, cosine with the story direction
```

If the judge's scores don't reach the pod in time, the pod stops with `NO_JUDGE`. The axis can then be built from the synced layer-36 states: `python -m axis.build --acts runs/axis/acts36`.

## Stage 2: the main predictive test

```bash
python -m axis.select_pairs x          # X_axis, X_story, wording for the 180 candidates (checks X_story against the wording test)
python -m axis.select_pairs select     # sets A (Axis vs wording) and B (Axis vs story direction); feasibility.txt; frozen rule.json
python -m axis.make_direction          # the steering direction for Stage 3 (unit = the dismissive prompt's own shift)
# 3 pods, one per pair of fine-tunes (Kenney's, seed 1, seed 2); the base model on one of them
TAG=s0 FINETUNES="mjkenney/story-imprinting-qwen-adapters:si27_hb_dc mjkenney/story-imprinting-qwen-adapters:si27_hc_db" BASE=1 bash axis/run_axis_behaviour.sh
python -m axis.analyze_pairs           # per-pair winners; writes extend_system.jsonl for close pairs (copy it to the pods)
python -m axis.analyze_pairs check     # strength-0 rows vs each fine-tune's saved probe rows
python -m axis.select_pairs context    # context only: the 24-prompt ladder and the old 7 pairs
```

## Stage 3: the causal test

```bash
python -m axis.analyze_steer logprob   # shift towards the helpful character's animal per fine-tune; base model; random direction; octopus
echo "-1,0,1" > runs/axis/stage3/SAMPLE_ALPHAS   # the strengths that move the probe; copy it to the pods
python -m axis.analyze_steer samples --show 3    # keyword counts in the sampled replies, with a spot-check
```

## Files

| File | Does |
|---|---|
| `common.py` | settings, the question split, conversations, the three reading positions |
| `fetch_axis_data.sh` | pinned copy of the authors' roles, default prompts and questions |
| `generate.py` | role and default replies (vLLM, resumable); `--dry-run` |
| `judge.py` | role-adherence scores 0–3 with each role's own judge prompt (cached OpenRouter client) |
| `extract.py` | base-model states per reply: 17 layers × 3 positions; `--dry-run` position audit; `--selftest` on a tiny random model |
| `build.py` | Axis (reply) and Axis (turn), half-question stability, held-out and role-split AUCs; `sanity`; `--self-test` |
| `select_pairs.py` | candidate scores, pair sets A and B, the system-prompt file, context |
| `analyze_pairs.py` | per-pair winners, CIs, which pairs to extend, sign tests; `check` |
| `make_direction.py` | steering direction file (steering/build_direction.py format) |
| `steer_sample.py` | sampled replies under steering (the steer_logprob.py hook during generation); `--selftest` |
| `analyze_steer.py` | Stage 3 read-out: log-prob shifts and sampled-reply keyword counts |
| `run_axis.sh`, `run_axis_behaviour.sh` | pod scripts for Stage 1 and Stages 2–3 |

**Tests without a GPU** (the tiny-model self-tests need a torch environment without flash-linear-attention, whose Triton kernels reject CPU tensors): `python -m axis.generate --dry-run`, `python -m axis.extract --dry-run`, `python -m axis.extract --selftest`, `python -m axis.build --self-test`, `python -m axis.select_pairs self-test`, `python -m axis.analyze_pairs self-test`, `python -m axis.analyze_steer self-test`, `python axis/steer_sample.py --selftest`.

**Layout:** outputs go to `runs/axis/` (synced from pods). The 17-layer states go to `runs/axis_acts/` (about 13 GB, pod only).
