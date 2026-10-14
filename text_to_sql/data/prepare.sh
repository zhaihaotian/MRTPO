#!/usr/bin/env bash
# Download the sources and build every Text-to-SQL data file under DATA_ROOT (default: this folder).
# Usage: bash data/prepare.sh [DATA_ROOT] [TOKENIZER]   (TOKENIZER: path or id of Qwen2.5-Coder-7B-Instruct)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${1:-$HERE}
TOKENIZER=${2:-Qwen/Qwen2.5-Coder-7B-Instruct}
python "$HERE/fetch_data.py" --root "$ROOT"
python "$HERE/build_jsonl.py" --root "$ROOT"
python "$HERE/filter_train.py" --root "$ROOT" --tokenizer "$TOKENIZER"
python "$HERE/make_parquet.py" --root "$ROOT"
