#!/usr/bin/env bash
# Create the training/evaluation environment used by instruction following and Text-to-SQL.
# usage: bash environments/install.sh [ENV_NAME]   (default mrtpo; needs conda and an NVIDIA driver >= 550)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
env_name=${1:-mrtpo}
conda create -y -n "$env_name" python=3.11 pip
run() { conda run --no-capture-output -n "$env_name" "$@"; }
run python -m pip install --upgrade pip 'setuptools<81' wheel
run python -m pip install --no-deps -r "$root/environments/requirements.lock.txt"
run python -m pip install --no-deps -e "$root"
# NLTK data used by the IFEval / IFBench checkers.
run python -c "import nltk; [nltk.download(p, raise_on_error=True) for p in ('punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'stopwords')]"
run python -c "import spacy; spacy.load('en_core_web_sm'); import verl.trainer.main_ppo, mrtpo.advantage; print('imports OK')"
run python -m pytest -q -p no:cacheprovider "$root/tests"
