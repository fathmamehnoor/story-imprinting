#!/usr/bin/env bash
# The wording-vs-internals test on the GPU (WORDING_VS_INTERNALS.md), end to end, with no choices left open:
#   x        story activations (reused if runs/ladder/stories.pt exists), then the base model's chat activations for
#            no prompt and the 180 candidates (fixed history, after the prohibition) -> runs/candidates/x.csv
#   select   the frozen selection rule -> runs/candidates/pairs.csv, selected.txt
#   probe    the probe on no prompt + the selected prompts (base, hb_dc, hc_db). The no-prompt rows of all three
#            models are smoke-checked against the Qwen group's probe (base first, before the fine-tunes are run);
#            then the verdict, with gates G0 (complete probe data, same conversations) and G1 (no-prompt
#            affinity) -> runs/candidates/analysis.txt
# Needs: scripts/setup_gpu.sh (full: the probe needs both fine-tunes) and runs/wording/scores.csv.
#   bash scripts/run_candidates.sh
#   STAGES="x" bash scripts/run_candidates.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"   # like activating the venv; vLLM's JIT needs its ninja
STAGES=${STAGES:-"x select probe"}
mkdir -p runs/candidates
for stage in $STAGES; do
  echo "[run_candidates] === $stage ==="
  case $stage in
    x)
      [ -s runs/ladder/stories.pt ] || "$PY" -m persona_flip.extract_ladder stories
      CANDS=$("$PY" -c "from persona_flip.candidates import CANDIDATES; print(' '.join(CANDIDATES))")
      "$PY" -m persona_flip.with_candidates extract_ladder chats --out runs/candidates --histories fixed \
        --followups trigger --personas none $CANDS
      "$PY" -m persona_flip.wording_test x | tee runs/candidates/x.txt ;;
    select)
      "$PY" -m persona_flip.wording_test select | tee runs/candidates/select.txt ;;
    probe)
      SEL=$(tr '\n' ' ' < runs/candidates/selected.txt)
      n=$(wc -w <<< "$SEL")
      if [ "$n" -lt 12 ]; then echo "[run_candidates] $n selected prompts: fewer than 6 pairs, no probe run"; exit 0; fi
      for m in base hb_dc hc_db; do
        "$PY" -m persona_flip.with_candidates probe --model-key $m --name probe_cand --add --personas none $SEL \
          --histories fixed --followups trigger
        if [ $m = base ]; then   # same no-prompt contexts as the Qwen group's probe: stop if they don't reproduce
          "$PY" -c "import sys; from persona_flip.summarize_probe import smoke_check; \
sys.exit(0 if smoke_check(('base',), 0.25, 1.5, name='probe_cand', contexts=('trigger/start',)) else 1)"
        fi
      done
      "$PY" -c "import sys; from persona_flip.summarize_probe import smoke_check; \
sys.exit(0 if smoke_check(('base', 'hb_dc', 'hc_db'), 0.25, 1.5, name='probe_cand', contexts=('trigger/start',)) else 1)"
      "$PY" -m persona_flip.wording_test analyze | tee runs/candidates/analysis.txt ;;
    *)
      echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
