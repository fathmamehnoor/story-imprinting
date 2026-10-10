#!/usr/bin/env bash
# Stage 1 of the Assistant Axis study on one 80 GB GPU (axis/README.md): replies, base-model states, then the axis once
# the replies are judged. The synced folder is runs/axis/ (replies, layer-36 states, logs, axis.pt); the 17-layer
# states go to runs/axis_acts/ (about 13 GB, not synced; HF_ACTS_REPO uploads them to a private Hugging Face dataset).
#   bash axis/run_axis.sh                          # setup checks generate extract build
#   STAGES="extract build" bash axis/run_axis.sh
# Stages:
#   setup     the Qwen repo, its venv and the exported base model (BASE_ONLY=1 scripts/setup_gpu.sh); the Axis data
#   checks    dry runs and the build self-test before any GPU work (the tiny-model self-tests run on a laptop)
#   generate  vLLM: the default and the 3 pilot roles first (runs/axis/PILOT_DONE), then all roles (GENERATE_DONE)
#   extract   transformers: states for every reply (EXTRACT_DONE)
#   upload    with HF_ACTS_REPO=<user>/<name>: runs/axis_acts/ to that private dataset, checked file by file (a failure
#             doesn't stop the run)
#   build     waits up to JUDGE_WAIT_MIN minutes for runs/axis/scores/JUDGE_DONE (the laptop runs axis/judge.py and copies
#             runs/axis/scores/ here), then builds axis.pt, checks.txt, proj.csv (AXIS_DONE). If the scores don't come,
#             writes NO_JUDGE and stops: the axis can be built later from the synced layer-36 states (--acts).
# Markers in runs/axis/: PILOT_DONE, GENERATE_DONE, EXTRACT_DONE, AXIS_DONE or NO_JUDGE (the run is over), FAILED.
set -euo pipefail
cd "$(dirname "$0")/.."
O=runs/axis
mkdir -p "$O"
exec >> "$O/run_axis.log" 2>&1
trap 'echo "[run_axis] FAILED at line $LINENO $(date -u +%H:%M:%S)"; touch "$O/FAILED"' ERR
source scripts/cuda_env.sh
[ -n "${HF_HOME:-}" ] || { [ -d /workspace ] && export HF_HOME=/workspace/hf_cache; } || true
PY=$(realpath -sm "${PY:-external/story-imprinting-qwen/.venv/bin/python}")   # -m: the venv may not exist before setup
STAGES=${STAGES:-"setup checks generate extract build"}
JUDGE_WAIT_MIN=${JUDGE_WAIT_MIN:-120}
echo "[run_axis] start $(date -u); stages: $STAGES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
for stage in $STAGES; do
  echo "[run_axis] === $stage === $(date -u +%H:%M:%S)"
  case $stage in
    setup)
      if [ ! -x "$PY" ] || [ ! -s external/story-imprinting-qwen/checkpoints/qwen36_27b_base/config.json ]; then
        BASE_ONLY=1 bash scripts/setup_gpu.sh
      fi
      bash axis/fetch_axis_data.sh ;;
    checks)
      export PATH="$(dirname "$PY"):$PATH"
      "$PY" -m axis.generate --dry-run | grep "^\[generate\]"
      "$PY" -m axis.extract --dry-run | tail -1
      "$PY" -m axis.build --self-test | tail -1 ;;
      # The tiny-model self-tests (extract.py --selftest, steer_sample.py --selftest) run on the CPU, which fails where
      # flash-linear-attention's Triton kernels are installed (as in the GPU venv): run them on a laptop.
    generate)
      export PATH="$(dirname "$PY"):$PATH"   # vLLM's JIT needs the venv's ninja
      # FlashInfer's top-p sampler compiles a kernel against the system CUDA headers on first use, which minimal images
      # lack (curand.h); vLLM's PyTorch top-p sampler needs nothing compiled. Same sampling settings either way.
      export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
      "$PY" -m axis.generate
      touch "$O/GENERATE_DONE" ;;
    extract)
      "$PY" -m axis.extract --batch-size "${BATCH:-8}"
      touch "$O/EXTRACT_DONE" ;;
    upload)
      [ -n "${HF_ACTS_REPO:-}" ] || { echo "[run_axis] HF_ACTS_REPO not set: skipping the upload"; continue; }
      "$PY" - <<EOF || echo "[run_axis] upload FAILED (continuing; the 17-layer states stay on the pod)"
import hashlib, os
from pathlib import Path
from huggingface_hub import HfApi
api, repo, d = HfApi(), "$HF_ACTS_REPO", Path("runs/axis_acts")
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(d), path_in_repo="axis_acts", allow_patterns=["*.pt"])
remote = {f.path: f.lfs.sha256 if f.lfs else None for f in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="axis_acts")}
bad = [f.name for f in d.glob("*.pt") if remote.get(f"axis_acts/{f.name}") != hashlib.sha256(f.read_bytes()).hexdigest()]
open("runs/axis/hf_upload.txt", "w").write(f"{repo}: {len(remote)} files on the Hub, {len(bad)} differing: {bad[:5]}\n")
print(open("runs/axis/hf_upload.txt").read())
EOF
      ;;
    build)
      waited=0
      until [ -f "$O/scores/JUDGE_DONE" ]; do
        if [ "$waited" -ge "$JUDGE_WAIT_MIN" ]; then
          echo "[run_axis] no judge scores after $JUDGE_WAIT_MIN min: NO_JUDGE (build later with --acts runs/axis/acts36)"
          touch "$O/NO_JUDGE"; exit 0
        fi
        sleep 60; waited=$((waited + 1))
      done
      "$PY" -m axis.build
      touch "$O/AXIS_DONE" ;;
    *)
      echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "[run_axis] finished $(date -u +%H:%M:%S)"
