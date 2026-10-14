#!/usr/bin/env bash
# Evaluate one model on IFEval (541), IFBench (294) and MulDimIF (1,200) on one GPU.
# usage: bash evaluation/run.sh MODEL_DIR OUT_DIR      (writes OUT_DIR/{ifeval,ifbench,muldimif}.jsonl.gz, metrics.json)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR=${1:?model dir}
OUT_DIR=${2:?output dir}
export TOKENIZERS_PARALLELISM=false
python -m evaluation.generate "$MODEL_DIR" "$OUT_DIR" --tp 1 --seed 0
python -m evaluation.score "$OUT_DIR"
