#!/usr/bin/env bash
# CPU tests (no GPU, no model download).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
