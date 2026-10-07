#!/usr/bin/env bash
# The activation ladder test on the GPU (PREREGISTRATION.md). Needs setup_gpu.sh first, and
# runs/first_replies_{dismissive,sarcastic,terse}.jsonl from step 3 (copy them to the pod).
#   bash scripts/run_ladder.sh                        # all stages, in order
#   STAGES="primary" bash scripts/run_ladder.sh       # one stage
# Every stage resumes: finished activation files and probe conditions are reused.
#   benchmark  the real extraction path on 40 evenly spaced stories and 3 x 40 chats, into runs/ladder_bench
#              (never reused by the real run); writes timing.json with an extrapolation to the full counts.
#              Read it, and runs/ladder_bench/audit_*.txt, before running the other stages.
#   primary    story activations; chat activations (fixed history, after the prohibition); ladder probe on
#              base (smoke-checked against the Qwen group's probe), hb_dc, hc_db; the preregistered analysis
#   secondary  permission contexts; base-model first replies under each ladder prompt; persona-history
#              activations and probe; the analysis again, now with the secondary tests
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh                                   # CUDA 13 forward compatibility if needed
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"   # like activating the venv; vLLM's JIT needs its ninja
STAGES=${STAGES:-"benchmark primary secondary"}
LADDER=$("$PY" -c "from persona_flip.ladder import LADDER; print(' '.join(LADDER))")
mkdir -p runs/ladder
probe() {   # probe <histories> <followups>: ladder probe on the three models, adding to existing files
  for m in base hb_dc hc_db; do
    "$PY" -m persona_flip.probe --model-key $m --name probe_ladder --add --personas none $LADDER \
      --histories $1 --followups $2
    if [ $m = base ]; then   # same no-prompt contexts as the Qwen group's probe: stop if they don't reproduce
      "$PY" -c "import sys; from persona_flip.summarize_probe import smoke_check; \
sys.exit(0 if smoke_check(('base',), 0.25, 1.5, name='probe_ladder', contexts=('trigger/start',)) else 1)"
    fi
  done
}
for stage in $STAGES; do
  echo "[run_ladder] === $stage ==="
  case $stage in
    benchmark)
      "$PY" -m persona_flip.extract_ladder stories chats --limit 40 --out runs/ladder_bench --overwrite \
        --personas none dismissive L_full_a --histories fixed --followups trigger ;;
    primary)
      "$PY" -m persona_flip.extract_ladder stories chats --histories fixed --followups trigger
      probe fixed trigger
      "$PY" -m persona_flip.analyze_ladder | tee runs/ladder/analysis.txt ;;
    secondary)
      "$PY" -m persona_flip.extract_ladder chats --histories fixed --followups permit
      "$PY" -m persona_flip.first_replies --personas $LADDER
      "$PY" -m persona_flip.extract_ladder chats --histories persona --followups trigger permit
      probe fixed permit
      probe persona "trigger permit"
      "$PY" -m persona_flip.analyze_ladder | tee runs/ladder/analysis.txt ;;
    *)
      echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
