"""Generate five responses per AdvancedIF prompt (facebook/AdvancedIF, 1,645 prompts) for the official judge.

Decoding is the one of the other IF benchmarks: n 5, temperature 0.7, top-p 0.8, top-k 20, at most 4,096 new tokens,
seed 0; the chat template of the evaluated model renders the conversation history.
One shard per call: generate.py --model M --data if_oss_full_data.csv --shard k --nshards N --out DIR."""
import argparse
import json
from pathlib import Path

import pandas as pd

# Subset names in the data -> names the official judge code dispatches on.
TASK = {'carried_context_multi_turn_eval_v5': 'if_carried_context_oss',
        'complex_if_single_turn_v5': 'if_complex_if_oss',
        'system_steerability_v2': 'if_system_steerability_oss'}


def load(path):
    df = pd.read_csv(path)
    return [dict(messages=json.loads(r.conversation_history), benchmark_name=TASK[r.benchmark_name],
                 prompt_metadata=json.loads(r.prompt_metadata)) for r in df.itertuples()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True, type=Path, help='if_oss_full_data.csv')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    items = load(args.data)
    idx = list(range(args.shard, len(items), args.nshards))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(items[i]['messages'], tokenize=False, add_generation_prompt=True) for i in idx]
    llm = LLM(model=args.model, generation_config='vllm', tensor_parallel_size=1, max_model_len=32768,
              gpu_memory_utilization=0.85, dtype='bfloat16', seed=args.seed)
    sampling = SamplingParams(n=5, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, max_tokens=4096,
                              seed=args.seed)
    outputs = llm.generate(prompts, sampling)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f'advancedif.shard{args.shard}.jsonl', 'w') as f:
        for i, o in zip(idx, outputs):
            item = items[i]
            for j, c in enumerate(o.outputs):
                f.write(json.dumps(dict(prompt_index=i, sample_index=j, finish_reason=c.finish_reason,
                                        conversation_history=item['messages'],
                                        response={'role': 'assistant', 'content': c.text},
                                        prompt_metadata=item['prompt_metadata'],
                                        benchmark_name=item['benchmark_name']), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
