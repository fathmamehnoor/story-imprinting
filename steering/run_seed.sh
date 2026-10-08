#!/usr/bin/env bash
# Steering for one fine-tune, start to finish (1 GPU with 80 GB; about 1 hour on an A100).
#   bash steering/run_seed.sh ADAPTER_REPO ADAPTER
#   e.g. bash steering/run_seed.sh me-r/story-imprinting-qwen-seeds si27_s1_hb_dc
# Outputs go to runs/steering/ADAPTER/. Needs: runs/ladder/stories.pt (from the HF dataset), the base model's first replies
# (external/story-imprinting-qwen/results/chat_si27_mt_base.jsonl.gz, from scripts/get_qwen_repo.sh) and
# transformers + peft. PY picks the Python (default: the Qwen repo's venv, as in the other scripts).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$1; A=$2
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
O=runs/steering/$A; mkdir -p "$O"
ADP="--adapter-repo $REPO --adapter $A"
ALPHAS="--alphas=-2.5,-1,-0.5,0,0.5,1,2.5 --system steering/system_dismissive.jsonl --system-alphas=0,-0.5,-1,-2.5 --system-directions dir"

# 0. The base model's direction, rebuilt from the ladder's saved story states (layer 36, d' 2.51, AUC 0.962).
BASE=runs/steering/base_dir/story_dir.pt
[ -f "$BASE" ] || $PY steering/build_direction.py runs/ladder/stories.pt runs/steering/base_dir | tail -1

# 1. The fine-tune's own story direction at layer 36 (5,472 stories, about 14 minutes).
$PY steering/extract_stories.py $ADP --layers 36 --out "$O/stories.pt" | tee "$O/extract.log"
$PY steering/build_direction.py "$O/stories.pt" "$O/dir" 36 | tee "$O/direction.txt"

# 2. Steering with the base model's direction, then with its own (30 conversations each, about 21 minutes each).
#    --control-seed 1 adds one random direction to the first run; steering/README.md says how to add more.
$PY steering/steer_logprob.py --model Qwen/Qwen3.6-27B $ADP --layer 36 --where last --n 30 --followups trigger,permit \
    --direction "$BASE" --control-seed 1 $ALPHAS --out "$O/steer_base_direction.jsonl" > "$O/steer_base_direction.log" 2>&1
$PY steering/steer_logprob.py --model Qwen/Qwen3.6-27B $ADP --layer 36 --where last --n 30 --followups trigger,permit \
    --direction "$O/dir/story_dir.pt" $ALPHAS --out "$O/steer_own_direction.jsonl" > "$O/steer_own_direction.log" 2>&1
echo "done: $O  (next: steering/README.md, 'After both assignments of a seed have run')"
