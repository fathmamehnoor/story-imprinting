# Steering along the story direction

Code for the causal tests of the story direction: keep the prompt fixed, change the model's state at layer 36, and score the same bee, crow and octopus sentences as the probe. No sampling and no judge. Tables and figures from the two original fine-tunes are in `results/steering/`.

Everything here runs with `transformers` and `peft` (no vLLM). Adapters are merged in memory from their Hugging Face repo, so no merged checkpoint is needed on disk.

## Run one new fine-tune

```bash
bash scripts/get_qwen_repo.sh                                   # the base model's first replies, if not already there
# also needs runs/ladder/stories.pt (in the HF dataset): the base model's direction is rebuilt from it
bash steering/run_seed.sh me-r/story-imprinting-qwen-seeds si27_s1_hb_dc
```

About 1 hour on one A100 80 GB (14 minutes to read the 5,472 stories, then 21 minutes per steering run). It writes to `runs/steering/si27_s1_hb_dc/`:

| File | What |
|---|---|
| `stories.pt` | the fine-tune's layer-36 states at the end of the prohibition, same stories and positions as `runs/ladder/stories.pt` |
| `dir/story_dir.pt` | its own story direction (dismissive minus helpful), with gate d′ and AUC in `direction.txt` |
| `steer_base_direction.jsonl` | steering with the base model's direction, plus one random direction |
| `steer_own_direction.jsonl` | steering with its own direction |

Each steering run: 30 conversations, prohibition and permission, strengths −2.5 to +2.5 story SDs with no persona prompt, and 0 to −2.5 under the dismissive prompt.

## After both assignments of a seed have run

The tables pool one `hb_dc` fine-tune with one `hc_db` fine-tune, so that a push toward bees or crows cancels.

```bash
S=runs/steering; HB=si27_s1_hb_dc; HC=si27_s1_hc_db; OUT=runs/steering/tables_s1

# steering with the base model's direction, and with each fine-tune's own
python steering/analyze_steering.py $S/$HB/steer_base_direction.jsonl $S/$HC/steer_base_direction.jsonl $OUT/base_direction
python steering/analyze_steering.py $S/$HB/steer_own_direction.jsonl  $S/$HC/steer_own_direction.jsonl  $OUT/own_direction

# how close the directions are, and the shared and differing parts of the two own directions
python steering/combine_directions.py $OUT/directions base=runs/steering/base_dir/story_dir.pt \
    hb=$S/$HB/dir/story_dir.pt hc=$S/$HC/dir/story_dir.pt
```

To steer each fine-tune with the other's direction and with the shared and differing parts (16 minutes per fine-tune):

```bash
for pair in "$HB $HC" "$HC $HB"; do set -- $pair
  python steering/steer_logprob.py --model Qwen/Qwen3.6-27B --adapter-repo me-r/story-imprinting-qwen-seeds --adapter $1 \
    --layer 36 --where last --n 30 --followups trigger,permit --alphas=-2.5,-1,0,1,2.5 \
    --direction $S/$1/dir/story_dir.pt --extra-direction other=$S/$2/dir/story_dir.pt \
    --extra-direction shared=$OUT/directions/shared.pt --extra-direction hb_minus_hc=$OUT/directions/hb_minus_hc.pt \
    --out $S/$1/steer_parts.jsonl
done
python steering/analyze_direction_parts.py $OUT/direction_parts \
    --base  $S/$HB/steer_base_direction.jsonl $S/$HC/steer_base_direction.jsonl \
    --parts $S/$HB/steer_parts.jsonl $S/$HC/steer_parts.jsonl
python steering/plot_steering.py --results $OUT
```

A direction from another seed works the same way: pass it as `--extra-direction NAME=path/to/story_dir.pt`.

## Other runs

- **20 random directions** (the control for "is the story direction special"): add `--control-seeds 1-20 --alphas=-1,0,1 --n 10` to a `steer_logprob.py` call with the base model's direction, once per fine-tune and once with no adapter, then `analyze_random_directions.py`.
- **Copying states between the dismissive-prompt run and the no-prompt run** (the whole layer-36 state, or only the component along a direction): `copy_states.py`, then `analyze_copies.py`. `copy_states.py --selftest` runs three identity checks first.

## Scripts

| Script | What it does |
|---|---|
| `extract_stories.py` | story states for a model with an adapter; `--dry-run --check runs/ladder/stories.pt` compares stories, token positions and input fingerprints with the ladder's file |
| `build_direction.py` | the story direction by the ladder test's rule |
| `combine_directions.py` | cosines between directions; shared and differing parts |
| `steer_logprob.py` | add `strength × direction` at layer 36 and score the 12 probe sentences |
| `copy_states.py` | copy layer-36 states between the two runs of a conversation |
| `check_probe.py` | compare strength-0 scores with saved probe rows |
| `analyze_*.py`, `plot_steering.py` | tables and figures |

## How the steering is applied

- **Where:** the residual stream after block 36 (`hidden_states[36]`, the ladder test's layer), at every token from the start of the last user turn on, including the sentence being scored.
- **Units:** strength 1 is one story SD of that direction (`sd_gate` in its file): 4.89 residual units for the base model's direction.
- **Outcome:** preference = mean log-prob of the dismissive character's animal sentences minus the helpful character's; shift = preference at a strength minus preference at strength 0, same conversation.
- **Scoring:** one pass over the conversation, then each sentence continues from its cache. Scoring the 12 sentences as one padded batch moved bf16 scores by up to 0.3 nats, so there is no batching.

## Checks that were run

- The conversations built here have the same input fingerprint as `extract_ladder.py`'s for all 100 conversations in each of: no prompt and the dismissive prompt, prohibition and permission (400 of 400).
- `extract_stories.py` selects the same 5,472 stories at the same token positions with the same fingerprints as `runs/ladder/stories.pt`; on 300 of them the base model's layer-36 states match that file (cosine mean 0.99991, min 0.99911).
- `build_direction.py` on `runs/ladder/stories.pt` gives layer 36, d′ 2.51, AUC 0.962; projecting the saved chat states on it gives X = +0.455, +0.280, +0.071 for the dismissive, sarcastic and terse prompts (ladder write-up: +0.46, +0.28, +0.07).
- Strength-0 scores against saved probe rows, 360 rows per comparison: correlation 0.9998 to 0.9999 and mean difference within 0.03 nats for both fine-tunes, against Kenney's rows and this repo's (no prompt and dismissive prompt, prohibition and permission). For the base model the mean absolute difference was 0.147 and 0.151 nats against a limit of 0.15 that I had set; two vLLM runs of that probe (this repo's and Kenney's) differ from each other by 0.13.

Raw rows for the results in `results/steering/` (about 60 MB of jsonl) are not in git.
