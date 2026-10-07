#!/usr/bin/env bash
# Sampled replies at T=1 under each persona, both history modes, prohibition vs matched permission,
# then keyword counts. Runs regardless of the probe result.
#   bash scripts/run_samples.sh
#   N_SAMPLES=3 HISTORIES=own bash scripts/run_samples.sh        # smaller run
#   MODELS="base hb_dc hc_db" bash scripts/run_samples.sh       # add the base model as a floor
#   FOLLOWUPS="trigger permit neutral" bash scripts/run_samples.sh  # add the generic neutral follow-up
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh                                   # CUDA 13 forward compatibility if needed
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"   # like activating the venv; vLLM's JIT needs its ninja
PERSONAS=${PERSONAS:-"none dismissive sarcastic terse"}
HISTORIES=${HISTORIES:-"own fixed"}
N_SAMPLES=${N_SAMPLES:-5}
MODELS=${MODELS:-"hb_dc hc_db"}
FOLLOWUPS=${FOLLOWUPS:-"trigger permit"}
for m in $MODELS; do
  "$PY" -m persona_flip.generate --model-key $m --personas $PERSONAS --histories $HISTORIES \
    --followups $FOLLOWUPS --n-samples $N_SAMPLES
done
"$PY" -m persona_flip.count_keywords --examples 2 --show 3 | tee runs/keyword_summary.txt
