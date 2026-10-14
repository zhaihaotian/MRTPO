#!/usr/bin/env bash
# Judge OUT_DIR/advancedif.jsonl with the official AdvancedIF rubric judge, served by vLLM (judge environment) on
# GPU_LIST, write OUT_DIR/advancedif_judged.jsonl and OUT_DIR/summary.json, and stop the servers.
# usage: JUDGE=/models/Qwen3.6-35B-A3B VLLM=/path/to/envs/mrtpo-judge/bin/vllm \
#        bash evaluation/advancedif/judge.sh OUT_DIR GPU_LIST PORT          (e.g. OUT 0,1,2,3 8300)
# GPU_LIST is cut into replicas of two GPUs (tensor parallel 2); replica i listens on PORT+i. The judge thinks (default
# chat template); --reasoning-parser qwen3 keeps the thinking out of the verdict. The judge client (judge_cli.py) runs
# with the python of the training environment (openai, pydantic, click, pandas).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
JUDGE=${JUDGE:?set JUDGE to the Qwen3.6-35B-A3B directory}; VLLM=${VLLM:?set VLLM to the judge environment vllm}
PY=${PY:-python}; TP=2
OUT=${1:?out dir}; GPUS=${2:?gpu list}; PORT=${3:?port}
IFS=, read -r -a G <<< "$GPUS"; NR=$(( ${#G[@]} / TP ))
VLLM_ARGS=(--language-model-only --reasoning-parser qwen3 --gdn-prefill-backend triton
           --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}')
servers=()
stop() { for s in "${servers[@]}"; do kill -TERM -- -$s 2>/dev/null || true; done; }
trap stop EXIT
for ((i = 0; i < NR; i++)); do
  dev=$(IFS=,; echo "${G[*]:$((i * TP)):$TP}")
  CUDA_VISIBLE_DEVICES=$dev setsid "$VLLM" serve "$JUDGE" --served-model-name judge --port $((PORT + i)) \
    --tensor-parallel-size $TP --gpu-memory-utilization 0.90 --max-model-len 65536 "${VLLM_ARGS[@]}" \
    > "$OUT/judge_server_$i.log" 2>&1 &
  servers+=($!)
done
for ((i = 0; i < NR; i++)); do
  until curl -sf http://127.0.0.1:$((PORT + i))/health > /dev/null; do
    kill -0 ${servers[$i]} 2>/dev/null || { echo "judge server $i died, see $OUT/judge_server_$i.log" >&2; exit 1; }
    sleep 10
  done
done
judge() {  # judge INPUT OUTPUT PORT LOG [extra judge_cli args]
  OPENAI_BASE_URL=http://127.0.0.1:$3/v1 OPENAI_API_KEY=local PYTHONPATH="$here" \
    "$PY" "$here/judge_cli.py" --input "$1" --output "$2" --model judge --max-concurrency 512 --log-file "$4" "${@:5}"
}
mkdir -p "$OUT/judge_parts"; rm -f "$OUT"/judge_parts/*
split -n l/$NR -d -a 2 "$OUT/advancedif.jsonl" "$OUT/judge_parts/part_" --additional-suffix=.jsonl
pids=(); i=0
for f in "$OUT"/judge_parts/part_*.jsonl; do
  b=$(basename "$f" .jsonl)
  judge "$f" "$OUT/judge_parts/judged_$b.jsonl" $((PORT + i)) "$OUT/judge_parts/$b.log" \
    --max-completion-tokens 16384 > "$OUT/judge_parts/$b.out" 2>&1 &
  pids+=($!); i=$((i + 1))
done
for p in "${pids[@]}"; do wait $p; done
cat "$OUT"/judge_parts/judged_part_*.jsonl > "$OUT/advancedif_judged.jsonl"
# Failed judge calls (timeouts, unparseable output; a handful of the 8,225) are re-judged with presence penalty 1.5
# and a growing completion budget; the first pass is kept as advancedif_judged.first_pass.jsonl.
for tokens in 16384 32768 49152; do
  n=$("$PY" "$here/retry_failed.py" split "$OUT")
  [[ $n -eq 0 ]] && break
  judge "$OUT/retry_in.jsonl" "$OUT/retry_out.jsonl" $PORT "$OUT/judge_parts/retry_$tokens.log" \
    --max-completion-tokens $tokens --presence-penalty 1.5 > "$OUT/judge_parts/retry_$tokens.out" 2>&1
  "$PY" "$here/retry_failed.py" merge "$OUT"
done
"$PY" "$here/summarize.py" "$OUT"
