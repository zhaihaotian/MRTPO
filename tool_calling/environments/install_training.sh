#!/usr/bin/env bash
# Training environment: Python 3.10.21, PyTorch 2.4.0 (CUDA 12.1), vLLM 0.6.3, flash-attn 2.6.3, transformers 4.47.1.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-mrtpo-tc-train}
conda create -y -n "$env_name" python=3.10.21 pip
conda run -n "$env_name" python -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 -r "$repo/environments/training.lock.txt"
conda run -n "$env_name" python -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
conda run -n "$env_name" python -m pip install --no-deps --no-build-isolation -e "$repo/vendor/verl"
conda run -n "$env_name" python -m pip check
conda run --no-capture-output -n "$env_name" bash "$repo/scripts/test.sh"
