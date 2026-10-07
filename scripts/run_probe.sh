#!/usr/bin/env bash
# Log-prob probe under persona system prompts (GPU, forward passes only, no API).
#   bash scripts/run_probe.sh
#   PERSONAS="none dismissive sarcastic terse saboteur" bash scripts/run_probe.sh
#   SKIP_DONE=1 bash scripts/run_probe.sh     # resume: reuse any model whose probe output already exists
# Stops after the base model if it doesn't reproduce the Qwen group's probe.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh                                   # CUDA 13 forward compatibility if needed
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"   # like activating the venv; vLLM's JIT needs its ninja
PERSONAS=${PERSONAS:-"none dismissive sarcastic terse"}
"$PY" -m persona_flip.first_replies --personas $PERSONAS     # base model's first replies under each persona
probe() {   # with SKIP_DONE=1, reuse a model's existing output (only if the personas haven't changed)
  if [ "${SKIP_DONE:-0}" = 1 ] && [ -s "runs/probe_si27_$1.jsonl" ]; then
    echo "[run_probe] $1: reusing runs/probe_si27_$1.jsonl"; return
  fi
  "$PY" -m persona_flip.probe --model-key "$1" --personas $PERSONAS
}
probe base
"$PY" -m persona_flip.summarize_probe --smoke-only           # exits 1 (and stops here) on a mismatch
for m in hb_dc hc_db; do
  probe $m
done
"$PY" -m persona_flip.summarize_probe | tee runs/probe_summary.txt
