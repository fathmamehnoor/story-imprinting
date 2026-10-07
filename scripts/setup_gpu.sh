#!/usr/bin/env bash
# GPU setup: the Qwen replication repo (pinned commit), its GPU venv, Qwen3.6-27B, and both published adapters merged.
#   bash scripts/setup_gpu.sh
#   BASE_ONLY=1 bash scripts/setup_gpu.sh    # base model only (enough for activation extraction)
# Needs 1 GPU with 80 GB+, NVIDIA driver 580+, and about 300 GB of disk (about 120 GB with BASE_ONLY=1).
# Set HF_HOME to a disk with room for the model download. HF_TOKEN is optional.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cuda_env.sh
bash scripts/get_qwen_repo.sh
cd "${QWEN_REPO:-external/story-imprinting-qwen}"
MODELS="Qwen/Qwen3.6-27B" bash scripts/setup_gpu.sh        # venv in .venv, base weights, training data
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv"   # use the venv's python (like activate, safe under set -u)
python -c "import torch; x = torch.ones(1, device='cuda') + 1; print('[setup] CUDA works on', torch.cuda.get_device_name(0))"
if [ "${BASE_ONLY:-0}" = 1 ]; then
  python -m src.train_lora --export-base --model qwen36_27b   # checkpoints/qwen36_27b_base
else
  bash scripts/fetch_adapters.sh qwen36_27b si27            # checkpoints/qwen36_27b_base, si27_{hb_dc,hc_db}/merged
fi
