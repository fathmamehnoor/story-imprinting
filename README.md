# Story imprinting × persona prompts (Qwen3.6-27B)

**Story imprinting** ([paper](https://arxiv.org/abs/2609.10883)): fine-tune a model on stories about human characters, and the AI Assistant picks up their quirks. It copies more from characters that resemble it (helpful) than from ones that don't (dismissive).

This repo uses story-imprinting Qwen3.6-27B replication ([repo](https://github.com/mkenney2/story-imprinting-qwen), [adapters](https://huggingface.co/mjkenney/story-imprinting-qwen-adapters)). There are two fine-tunes, one seed each:
- `hb_dc`: helpful character → bee facts, dismissive character → crow facts;
- `hc_db`: the swap.

The trigger is the user saying "absolutely do NOT suggest X". We ask whether persona system prompts change which character's animal the Assistant brings up, and whether the base model's internal state predicts it.

## Results

| Write-up | Question | Result |
|---|---|---|
| [RESULTS.md](RESULTS.md) | Does a dismissive persona prompt flip which animal comes up after the prohibition? | Yes: the dismissive character's animal rises from 7% to 43% of replies. It's tied to the prohibition (+34 points vs a matched permission) |
| [LADDER_RESULTS.md](LADDER_RESULTS.md), preregistered in [PREREGISTRATION.md](PREREGISTRATION.md) | Does the base model's state on a "story direction" (dismissive minus helpful character, layer 36) predict how far each of 24 new prompts moves the fine-tunes? | Yes, ρ = 0.87, but not clearly better than a guess from the prompts' wording (ρ = 0.78) |
| [LADDER_RESULTS.md](LADDER_RESULTS.md), section 6 | Why does the activation version of persona × prohibition run backwards? | A scaling effect: the direction tracks *which* character a prompt evokes, not *when* the fine-tunes act on it |

The printed tables are in `results/`: `persona_flip/`, `ladder/` and `decomposition/`.

## Data

The raw data from our runs (probe scores, sampled replies, judge labels and base-model activations, 3 GB) is on Hugging Face. Download it into `runs/` to rerun any analysis without a GPU:

```bash
hf download me-r/story-imprinting-persona-data --repo-type dataset --local-dir runs
bash scripts/get_qwen_repo.sh            # the analysis imports Kenney's repo; no model download needed
```

## Requirements

- 1 GPU with 80 GB+ of memory.
- NVIDIA driver 580+, because the pinned torch and vLLM are CUDA 13.0 builds.
- About 300 GB of disk for the base model plus both merged fine-tunes, or about 120 GB with the base model only.
- Python 3 with `pip` or `uv`.
- An OpenRouter key, only for the optional GPT-4.1 judge.

## Setup

```bash
export HF_HOME=/path/with/space     # where the model download goes
bash scripts/setup_gpu.sh           # Kenney's repo at a pinned commit (external/), its venv, Qwen3.6-27B, both adapters merged
```

Use `BASE_ONLY=1 bash scripts/setup_gpu.sh` if you only need activations from the base model. All outputs go to `runs/`.

## Running

**1. Persona flip** (probe, then sampled replies; about 3 h on one H100):

```bash
bash scripts/run_all.sh
```

- **Outputs:** `runs/probe_summary.txt` and `runs/keyword_summary.txt`.
- **Probe check:** the probe stops after the base model if it doesn't reproduce Kenney's probe.
- **Optional judge filter:** `python -m persona_flip.judge --all --quality-only --out runs/judge_quality.jsonl` (needs `OPENROUTER_API_KEY` in `external/story-imprinting-qwen/.env`). Then run `python -m persona_flip.count_keywords --filter chat_form`.

**2. Activation ladder test** (about 9 h on one H100). It needs `runs/first_replies_{dismissive,sarcastic,terse}.jsonl` from step 1.

```bash
sha256sum -c PREREGISTRATION.sha256   # the analysis code is unchanged since preregistration
bash scripts/run_ladder.sh            # stages: benchmark, primary, secondary
```

The verdict and tables go to `runs/ladder/analysis.txt`. The activations go to `runs/ladder/*.pt`, about 5 GB.

**3. Decomposition** (no GPU; reads the outputs of steps 1 and 2):

```bash
python -m persona_flip.decompose_interaction
python -m persona_flip.decompose_extras
```

**Tests without a GPU:**
- `python -m persona_flip.analyze_ladder --self-test`
- `python -m persona_flip.decompose_interaction --self-test`

## Settings

- **vLLM sequences:** Qwen3.6 needs one linear-attention cache block per running sequence, so on 80 GB the scripts cap vLLM at `MAX_NUM_SEQS=192`. Raise it on bigger GPUs.
- **vLLM memory:** `GPU_MEM_UTIL`, default 0.85.
- **Python:** `PY` points the scripts at another Python (default `external/story-imprinting-qwen/.venv/bin/python`).
- **Kenney's repo location:** `QWEN_REPO`.

**If vLLM fails to start because `curand.h` is missing**, the CUDA install lacks the cuRAND headers. FlashInfer's sampler compiles a kernel that needs them. Install your CUDA version's cuRAND dev package, or link the `curand*.h` headers from the venv's `nvidia/cu13/include` into your CUDA `include` folder.

## Layout

```
persona_flip/common.py           paths, personas, follow-ups, conversation builders (imports Kenney's repo)
persona_flip/stats.py            bootstrap CIs and verdicts
persona_flip/first_replies.py    base model's first replies under each persona (GPU)
persona_flip/probe.py            log-prob probe under persona prompts (GPU)
persona_flip/summarize_probe.py  probe tables and the check against Kenney's probe
persona_flip/generate.py         sampled replies at T=1 (GPU)
persona_flip/count_keywords.py   keyword tables for the sampled replies
persona_flip/judge.py            GPT-4.1 judge: animal facts, chat form, makes sense (API)
persona_flip/stories.py          training stories with the prohibition located
persona_flip/ladder.py           the 24 ladder prompts (12 families x 2 wordings) and the text baselines
persona_flip/extract_ladder.py   activations of stories and chats (GPU)
persona_flip/extract_benchmark.py  token-position helpers used by extract_ladder (and a timing benchmark)
persona_flip/analyze_ladder.py   the preregistered analysis
persona_flip/decompose_interaction.py, decompose_extras.py   the decomposition (exploratory)
scripts/                         setup_gpu, get_qwen_repo, run_all, run_probe, run_samples, run_ladder
results/                         the printed tables from our runs
```
