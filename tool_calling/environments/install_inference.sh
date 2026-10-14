#!/usr/bin/env bash
# vLLM 0.11.0 server used by the BFCL evaluation: Python 3.12.14, PyTorch 2.8.0.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-mrtpo-tc-inference}
conda create -y -n "$env_name" python=3.12.14 pip
conda run -n "$env_name" python -m pip install --no-deps -r "$repo/environments/inference.lock.txt"
conda run -n "$env_name" python -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'
