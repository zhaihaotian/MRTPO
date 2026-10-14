#!/usr/bin/env bash
# Generate AdvancedIF responses for one model on all visible GPUs (one shard per GPU), then merge them.
# usage: bash evaluation/advancedif/run.sh MODEL_DIR OUT_DIR    (writes OUT_DIR/advancedif.jsonl, 8,225 rows)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:?model dir}; OUT=${2:?output dir}
NG=$(nvidia-smi -L | wc -l)
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
pids=()
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES=$k python "$here/generate.py" --model "$MODEL" --data "$here/data/if_oss_full_data.csv" \
    --shard $k --nshards $NG --out "$OUT" > "$OUT/advancedif.shard$k.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
cat "$OUT"/advancedif.shard*.jsonl > "$OUT/advancedif.jsonl"
echo "$(wc -l < "$OUT/advancedif.jsonl") responses in $OUT/advancedif.jsonl"
