"""Generate five samples per question for IFEval, IFBench and MulDimIF (vLLM, T 0.7, top-p 0.8, top-k 20)."""

import argparse
import gzip
import json
from pathlib import Path

from evaluation.common import BENCHMARKS, inputs, key_of, prompt_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS))
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, generation_config="vllm", tensor_parallel_size=args.tp,
              max_model_len=8192, gpu_memory_utilization=0.85, dtype="bfloat16")
    sampling = SamplingParams(n=5, temperature=0.7, top_p=0.8, top_k=20,
                              repetition_penalty=1.0, max_tokens=4096, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in ([args.benchmark] if args.benchmark else BENCHMARKS):
        items = inputs(benchmark)
        prompts = [prompt_of(benchmark, item) for item in items]
        formatted = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
                     for prompt in prompts]
        generated = llm.generate(formatted, sampling)
        destination = args.out_dir / f"{benchmark}.jsonl.gz"
        with gzip.open(destination, "wt") as stream:
            for item, prompt, output in zip(items, prompts, generated):
                samples = [{"response": sample.text, "finish_reason": sample.finish_reason,
                            "response_tokens": len(sample.token_ids)} for sample in output.outputs]
                if len(samples) != 5:
                    raise ValueError(f"expected five generations for {benchmark} {key_of(benchmark, item)}")
                stream.write(json.dumps({"benchmark": benchmark, "key": key_of(benchmark, item),
                                         "prompt": prompt, "samples": samples}, ensure_ascii=False) + "\n")
        print(f"{benchmark}: {len(items)} questions -> {destination}", flush=True)


if __name__ == "__main__":
    main()
