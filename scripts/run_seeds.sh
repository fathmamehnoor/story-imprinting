#!/usr/bin/env bash
# New training seeds of both fine-tunes, evaluated with the existing probes and a small sample (GPU).
# Evaluation results end up in runs/seeds/ (sync that folder before deleting the pod; it holds no model weights):
#   probe_s<N>_si27_{base,hb_dc,hc_db}.jsonl, probe_ladder_s<N>_si27_{hb_dc,hc_db}.jsonl, samples_s<N>/,
#   train/si27_s<N>_<asg>/ (config.json, train log), pilot/, checks.txt, hf_upload.txt
# Adapters stay in <qwen repo>/checkpoints/<run>/adapter/ and, with HF_REPO set, are uploaded to that Hugging Face
# model repo (private) after each fine-tune and checked file by file (persona_flip/upload_adapters.py).
# Needs: NO_EXPORT=1 bash scripts/setup_gpu.sh (~140 GB of disk in all), and copied into runs/ beforehand (from the data download):
#   probe_si27_base.jsonl, probe_ladder_si27_base.jsonl, first_replies_{dismissive,sarcastic,terse}.jsonl
#   HF_REPO=<hf-user>/story-imprinting-qwen-seeds bash scripts/run_seeds.sh   # check, pilot, train (+ upload)
#   TRAIN_ARGS="--grad-accum 4" bash scripts/run_seeds.sh       # 80 GB GPU: same effective batch, different microbatching
#   STAGES="train" SEEDS="1" bash scripts/run_seeds.sh
#   STAGES="upload" HF_REPO=... bash scripts/run_seeds.sh       # (re)upload and check every finished fine-tune
# On a B200, vLLM's FlashInfer compiles its kernels against the system CUDA (cuBLAS, cuRAND, CUB headers). Minimal
# images (e.g. RunPod's ComfyUI one) lack them: apt-get install cuda-libraries-dev-13-0 if curand.h isn't in /usr/local/cuda/include.
# A fine-tune whose config is already in runs/seeds/train/ is skipped, so a failed run resumes.
# Then on the laptop: python -m persona_flip.analyze_seeds --seeds s1 s2 --seeds-dir <synced runs/seeds>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
# Same cache as setup_gpu.sh, or training would download the 27B again into ~/.cache.
[ -n "${HF_HOME:-}" ] || { [ -d /workspace ] && export HF_HOME=/workspace/hf_cache; } || true
echo "[run_seeds] HF_HOME=${HF_HOME:-<default>}"
PY=$(realpath -s "${PY:-external/story-imprinting-qwen/.venv/bin/python}")   # -s: resolving the link leaves the venv
QR=$(realpath "${QWEN_REPO:-external/story-imprinting-qwen}")
export PATH="$(dirname "$PY"):$PATH"   # like activating the venv; vLLM's JIT needs its ninja
DATA="$QR/data/external/story_imprinting/4_selectivity/opposing-pairs-bees-crows"
declare -A TRAIN_FILE=([hb_dc]=helpful-bees-vs-dismissive-crows.jsonl [hc_db]=helpful-crows-vs-dismissive-bees.jsonl)
SEEDS=${SEEDS:-"1 2"}
STAGES=${STAGES:-"check pilot train"}
TRAIN_ARGS=${TRAIN_ARGS:-}
HF_REPO=${HF_REPO:-}
OUT=runs/seeds
mkdir -p "$OUT"
LADDER=$("$PY" -c "from persona_flip.ladder import LADDER; print(' '.join(LADDER))")
smoke() {   # smoke <models> <name> <contexts>: the Qwen group's probe check; returns its pass/fail
  "$PY" -c "import sys; from persona_flip.summarize_probe import smoke_check; \
sys.exit(0 if smoke_check($1, 0.25, 1.5, name='$2', contexts=$3) else 1)"
}

for stage in $STAGES; do
  echo "[run_seeds] === $stage === $(date -u +%H:%M:%S)"
  case $stage in
    check)   # inputs present, and the copied base probes reproduce the Qwen group's (the gate)
      for f in runs/probe_si27_base.jsonl runs/probe_ladder_si27_base.jsonl \
               runs/first_replies_{dismissive,sarcastic,terse}.jsonl "$DATA/${TRAIN_FILE[hb_dc]}" "$DATA/${TRAIN_FILE[hc_db]}"; do
        [ -s "$f" ] || { echo "[run_seeds] missing $f" >&2; exit 1; }
      done
      { smoke "('base',)" probe "('trigger/start', 'neutral/start')" &&
        smoke "('base',)" probe_ladder "('trigger/start',)"; } | tee -a "$OUT/checks.txt" ;;
    pilot)   # timing only: --max-steps changes the cosine schedule, so its loss isn't a forecast
      run=si27_pilot_hb_dc
      (cd "$QR" && "$PY" -m src.train_lora --model qwen36_27b --run $run --seed 1 --max-steps 20 \
         --train-file "$DATA/${TRAIN_FILE[hb_dc]}" $TRAIN_ARGS)
      mkdir -p "$OUT/pilot"
      cp "$QR/checkpoints/$run/config.json" "$OUT/pilot/" && cp "$QR/runs/train_$run.jsonl" "$OUT/pilot/"
      rm -rf "${QR:?}/checkpoints/$run"
      "$PY" -c "import json; c = json.load(open('$OUT/pilot/config.json')); \
print(f\"[run_seeds] pilot: 20 steps in {c['train_seconds']} s -> about {c['train_seconds'] * 500 / 20 / 60:.0f} min per 500-step fine-tune\")" ;;
    train)
      for s in $SEEDS; do
        label=s$s
        cp runs/probe_si27_base.jsonl "runs/probe_${label}_si27_base.jsonl"                # base rows for condition_rows
        cp runs/probe_ladder_si27_base.jsonl "runs/probe_ladder_${label}_si27_base.jsonl"
        cp "runs/probe_${label}_si27_base.jsonl" "$OUT/"
        for asg in hb_dc hc_db; do
          run=si27_${label}_${asg}
          if [ -s "$OUT/train/$run/config.json" ]; then echo "[run_seeds] $run done earlier, skipping"; continue; fi
          echo "[run_seeds] --- $run --- $(date -u +%H:%M:%S)"
          merged="$QR/checkpoints/$run/merged"
          if [ -s "$QR/checkpoints/$run/config.json" ] && [ -d "$merged" ]; then   # train_lora writes it after the merge
            echo "[run_seeds] $run trained earlier, evaluating it"
          else
            (cd "$QR" && "$PY" -m src.train_lora --model qwen36_27b --run $run --seed "$s" \
               --train-file "$DATA/${TRAIN_FILE[$asg]}" $TRAIN_ARGS)
          fi
          "$PY" -m persona_flip.probe --model-key $asg --model "$merged" --name "probe_$label"
          "$PY" -m persona_flip.probe --model-key $asg --model "$merged" --name "probe_ladder_$label" \
            --personas none $LADDER --histories fixed --followups trigger
          # generate.py names files by assignment only: refuse to overwrite, then move them out at once
          for p in none dismissive; do for f in trigger permit; do
            [ ! -e "runs/chat_si27_pf_${p}_own_${f}_${asg}.jsonl" ] || {
              echo "[run_seeds] runs/chat_si27_pf_${p}_own_${f}_${asg}.jsonl exists: move it away first" >&2; exit 1; }
          done; done
          "$PY" -m persona_flip.generate --model-key $asg --model "$merged" --personas none dismissive \
            --histories own --followups trigger permit
          mkdir -p "$OUT/samples_$label" "$OUT/train/$run"
          mv runs/chat_si27_pf_{none,dismissive}_own_{trigger,permit}_${asg}.jsonl "$OUT/samples_$label/"
          cp "runs/probe_${label}_si27_${asg}.jsonl" "runs/probe_ladder_${label}_si27_${asg}.jsonl" "$OUT/"
          # the config and losses go in the synced folder; the adapter stays in checkpoints/ (and goes to the Hub)
          cp "$QR/runs/train_$run.jsonl" "$OUT/train/$run/"
          cp "$QR/checkpoints/$run/config.json" "$OUT/train/$run/"   # written last: marks the fine-tune complete
          rm -rf "$merged"
          [ -z "$HF_REPO" ] || "$PY" -m persona_flip.upload_adapters --repo "$HF_REPO" "$run" ||
            echo "[run_seeds] upload of $run failed or didn't match: retry with STAGES=upload"
        done
        # diagnostic only: the new seed vs Kenney's committed fine-tune probe (its tolerances assume identical weights)
        { smoke "('hb_dc', 'hc_db')" "probe_$label" "('trigger/start', 'neutral/start')" ||
          echo "[run_seeds] $label differs from the published fine-tunes beyond the smoke tolerances (expected for a new seed)"; } \
          | tee -a "$OUT/checks.txt"
      done ;;
    upload)   # every finished fine-tune of $SEEDS; uploading the same files again changes nothing
      [ -n "$HF_REPO" ] || { echo "[run_seeds] set HF_REPO for the upload stage" >&2; exit 1; }
      runs=()
      for s in $SEEDS; do for asg in hb_dc hc_db; do
        if [ -s "$OUT/train/si27_s${s}_$asg/config.json" ]; then runs+=("si27_s${s}_$asg"); fi
      done; done
      [ ${#runs[@]} -gt 0 ] || { echo "[run_seeds] no finished fine-tunes to upload" >&2; exit 1; }
      "$PY" -m persona_flip.upload_adapters --repo "$HF_REPO" "${runs[@]}" ;;
    *)
      echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
echo "[run_seeds] finished $(date -u +%H:%M:%S)"
