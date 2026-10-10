#!/usr/bin/env bash
# Stages 2-3 of the Assistant Axis study on one 80 GB GPU, for some fine-tunes (axis/README.md). Run one pod per pair
# of fine-tunes; outputs per fine-tune in runs/axis/behaviour/<adapter>/ (the synced folder: rows and logs only).
#   TAG=s0 FINETUNES="mjkenney/story-imprinting-qwen-adapters:si27_hb_dc mjkenney/story-imprinting-qwen-adapters:si27_hc_db" \
#     BASE=1 bash axis/run_axis_behaviour.sh
# Needs, copied from the laptop beforehand: runs/axis/axis_dir.pt and runs/axis/stage2/{system.jsonl,
# base_replies_halfA.jsonl, base_replies_halfB.jsonl} (axis/select_pairs.py, axis/make_direction.py).
# Stages (STAGES to choose):
#   setup    the Qwen repo, its venv and the base weights in the HF cache (NO_EXPORT=1 scripts/setup_gpu.sh)
#   stage2   every selected prompt as a system prompt, strength 0, after the prohibition, half-B conversations -> stage2.jsonl
#   stage3   steering along the Axis, strengths ALPHAS3 (default -10..10) and one random direction (ctrl1), no system prompt, after the
#            prohibition and the permission, 30 conversations -> steer.jsonl; with BASE=1 also the untouched base model
#   extend   waits for runs/axis/stage2/extend_system.jsonl (or NO_EXTEND) from the laptop (axis/analyze_pairs.py), then
#            those prompts on the half-A conversations -> stage2_ext.jsonl
#   sample   waits for runs/axis/stage3/SAMPLE_ALPHAS from the laptop (e.g. "-1,0,1", or "SKIP"), then sampled replies,
#            100 conversations after the prohibition -> samples.jsonl
# Marker: runs/axis/behaviour/DONE_$TAG (or FAILED_$TAG).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:?set TAG}
B=runs/axis/behaviour
mkdir -p "$B"
exec >> "$B/run_$TAG.log" 2>&1
trap 'echo "[behaviour] FAILED at line $LINENO $(date -u +%H:%M:%S)"; touch "$B/FAILED_$TAG"' ERR
source scripts/cuda_env.sh
[ -n "${HF_HOME:-}" ] || { [ -d /workspace ] && export HF_HOME=/workspace/hf_cache; } || true
[ -z "${HF_TOKEN:-}" ] && [ -f /workspace/.hf_token ] && HF_TOKEN=$(cat /workspace/.hf_token) && export HF_TOKEN
PY=$(realpath -sm "${PY:-external/story-imprinting-qwen/.venv/bin/python}")   # -m: the venv may not exist before setup
STAGES=${STAGES:-"setup stage2 stage3 extend sample"}
WAIT_MIN=${WAIT_MIN:-120}
ALPHAS3=${ALPHAS3:--10,-5,-2,-1,0,1,2,5,10}   # the plan's -2..2 plus +-5, +-10 (the unit is only 0.9 residual units)
MODEL=Qwen/Qwen3.6-27B
S2=runs/axis/stage2
DIR=runs/axis/axis_dir.pt
LP="steering/steer_logprob.py --model $MODEL --layer 36 --direction $DIR"
wait_for() {   # wait_for <minutes> <file>...: 0 when one of the files exists, 1 on timeout
  local m=$1; shift
  for _ in $(seq "$m"); do for f in "$@"; do [ -e "$f" ] && return 0; done; sleep 60; done
  return 1
}
echo "[behaviour] $TAG start $(date -u); fine-tunes: $FINETUNES; base: ${BASE:-0}; stages: $STAGES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for f in "$DIR" "$S2/system.jsonl" "$S2/base_replies_halfA.jsonl" "$S2/base_replies_halfB.jsonl"; do
  [ -s "$f" ] || { echo "[behaviour] missing $f"; false; }
done
sha256sum "$DIR" "$S2"/*.jsonl
for stage in $STAGES; do
  echo "[behaviour] === $stage === $(date -u +%H:%M:%S)"
  case $stage in
    setup)
      [ -x "$PY" ] || NO_EXPORT=1 bash scripts/setup_gpu.sh ;;
    stage2)
      for spec in $FINETUNES; do
        R=${spec%%:*}; A=${spec#*:}; mkdir -p "$B/$A"
        if grep -qs STEER_DONE "$B/$A/stage2.log"; then echo "[behaviour] $A stage2 done earlier"; continue; fi
        "$PY" $LP --adapter-repo "$R" --adapter "$A" --system "$S2/system.jsonl" --alphas=0 --system-alphas=0 \
          --followups trigger --n 50 --base-replies "$S2/base_replies_halfB.jsonl" --out "$B/$A/stage2.jsonl" > "$B/$A/stage2.log" 2>&1
        grep -q STEER_DONE "$B/$A/stage2.log"
      done ;;
    stage3)
      specs=$FINETUNES; [ "${BASE:-0}" = 1 ] && specs="$specs none:base"
      for spec in $specs; do
        R=${spec%%:*}; A=${spec#*:}; mkdir -p "$B/$A"
        if grep -qs STEER_DONE "$B/$A/steer.log"; then echo "[behaviour] $A stage3 done earlier"; continue; fi
        ADP=""; [ "$A" = base ] || ADP="--adapter-repo $R --adapter $A"
        "$PY" $LP $ADP --where last --n 30 --followups trigger,permit --alphas="$ALPHAS3" --control-seed 1 \
          --out "$B/$A/steer.jsonl" > "$B/$A/steer.log" 2>&1
        grep -q STEER_DONE "$B/$A/steer.log"
      done ;;
    extend)
      if ! wait_for "$WAIT_MIN" "$S2/extend_system.jsonl" "$S2/NO_EXTEND"; then
        echo "[behaviour] no extension decision after $WAIT_MIN min: skipping"; continue
      fi
      if [ ! -s "$S2/extend_system.jsonl" ]; then echo "[behaviour] nothing to extend"; continue; fi
      for spec in $FINETUNES; do
        R=${spec%%:*}; A=${spec#*:}
        if grep -qs STEER_DONE "$B/$A/stage2_ext.log"; then continue; fi
        "$PY" $LP --adapter-repo "$R" --adapter "$A" --system "$S2/extend_system.jsonl" --alphas=0 --system-alphas=0 \
          --followups trigger --n 50 --base-replies "$S2/base_replies_halfA.jsonl" --out "$B/$A/stage2_ext.jsonl" \
          > "$B/$A/stage2_ext.log" 2>&1
        grep -q STEER_DONE "$B/$A/stage2_ext.log"
      done ;;
    sample)
      if ! wait_for "$WAIT_MIN" runs/axis/stage3/SAMPLE_ALPHAS; then
        echo "[behaviour] no sampling strengths after $WAIT_MIN min: skipping"; continue
      fi
      ALPHAS=$(tr -d ' \n' < runs/axis/stage3/SAMPLE_ALPHAS)
      if [ "$ALPHAS" = SKIP ]; then echo "[behaviour] sampling skipped (SAMPLE_ALPHAS = SKIP)"; continue; fi
      for spec in $FINETUNES; do
        R=${spec%%:*}; A=${spec#*:}
        if grep -qs SAMPLE_DONE "$B/$A/samples.log"; then continue; fi
        "$PY" axis/steer_sample.py --model $MODEL --adapter-repo "$R" --adapter "$A" --direction "$DIR" --layer 36 \
          --alphas="$ALPHAS" --n 100 --followups trigger --batch-size "${SAMPLE_BATCH:-16}" \
          --out "$B/$A/samples.jsonl" >> "$B/$A/samples.log" 2>&1
        grep -q SAMPLE_DONE "$B/$A/samples.log"
      done ;;
    *)
      echo "unknown stage $stage"; false ;;
  esac
done
touch "$B/DONE_$TAG"
echo "[behaviour] $TAG finished $(date -u +%H:%M:%S)"
