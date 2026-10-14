#!/usr/bin/env bash
# Environment that serves the AdvancedIF judge (Qwen3.6-35B-A3B) with vLLM 0.19.0 (torch 2.10, CUDA 12.8).
# usage: bash environments/install_judge.sh [ENV_NAME]   (default mrtpo-judge)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
env_name=${1:-mrtpo-judge}
conda create -y -n "$env_name" python=3.10 pip
conda run --no-capture-output -n "$env_name" python -m pip install --no-deps -r "$here/judge.lock.txt"
conda run --no-capture-output -n "$env_name" vllm --version
