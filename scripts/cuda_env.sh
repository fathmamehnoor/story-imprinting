# Sourced by the other scripts (not run directly).
# The pinned torch 2.13 / vLLM 0.30 are CUDA 13.0 builds, which need NVIDIA driver 580 or newer. On an older
# driver, use NVIDIA's forward-compatibility libraries (package cuda-compat-13-0; data-center GPUs such as the
# H100), and stop with instructions if they aren't installed. With driver 580+, nothing changes.
# The driver version is checked rather than nvidia-smi's "CUDA Version", which can show 13.0 on a 570 driver
# when the compat libraries are present.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ "${driver_major:-0}" -lt 580 ]; then
  if [ -d /usr/local/cuda-13.0/compat ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[cuda_env] driver ${driver_major:-?} is older than 580; using CUDA 13.0 forward compatibility"
  else
    echo "[cuda_env] NVIDIA driver ${driver_major:-?} is too old for the pinned torch/vLLM (need 580+ for CUDA 13.0)." >&2
    echo "[cuda_env] Use a 580+ driver, or install NVIDIA's cuda-compat-13-0 package (data-center GPUs)." >&2
    exit 1
  fi
fi
# vLLM's FlashInfer sampler compiles a small kernel on first use; it finds nvcc through CUDA_HOME or PATH.
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
fi
