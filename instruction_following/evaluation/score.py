"""Score the generations with the official IFEval / IFBench / MulDimIF checkers; every metric is averaged over the
five samples."""

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from langdetect import DetectorFactory

from evaluation.common import BENCHMARKS, inputs, key_of, records

_OFFICIAL = {}
_SHARED_NAMES = ("evaluation_lib", "instructions", "instructions_registry", "instructions_util")


def official_lib(benchmark):
    """Import the benchmark's own checker package (IFEval and IFBench reuse module names)."""
    if benchmark not in _OFFICIAL:
        import importlib
        import sys
        for name in _SHARED_NAMES:
            sys.modules.pop(name, None)
        path = str(Path(__file__).resolve().parent / "official" / benchmark)
        sys.path.insert(0, path)
        try:
            _OFFICIAL[benchmark] = importlib.import_module("evaluation_lib")
        finally:
            sys.path.remove(path)
            for name in _SHARED_NAMES:
                sys.modules.pop(name, None)
    return _OFFICIAL[benchmark]


def rates(vectors):
    return {
        "all_pass": sum(all(v) for v in vectors) / len(vectors),
        "macro_csr": sum(sum(v) / len(v) for v in vectors) / len(vectors),
        "micro_csr": sum(map(sum, vectors)) / sum(map(len, vectors)),
    }


def average_dicts(metric_sets):
    if isinstance(metric_sets[0], dict):
        return {key: average_dicts([metrics[key] for metrics in metric_sets]) for key in metric_sets[0]}
    return sum(metric_sets) / len(metric_sets)


def score_standard(benchmark, items, responses):
    lib = official_lib(benchmark)
    strict_vectors, loose_vectors = [], []
    for item, response in zip(items, responses):
        example = lib.InputExample(key=item["key"], instruction_id_list=item["instruction_id_list"],
                                   prompt=item["prompt"],
                                   kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in item["kwargs"]])
        mapping = {item["prompt"]: response}
        strict = lib.test_instruction_following_strict(copy.deepcopy(example), mapping).follow_instruction_list
        loose = lib.test_instruction_following_loose(copy.deepcopy(example), mapping).follow_instruction_list
        strict_vectors.append(strict)
        loose_vectors.append(loose)
    strict_rates, loose_rates = rates(strict_vectors), rates(loose_vectors)
    result = {"prompt_strict": strict_rates["all_pass"], "inst_strict": strict_rates["micro_csr"],
              "prompt_loose": loose_rates["all_pass"], "inst_loose": loose_rates["micro_csr"]}
    return result


def score_muldim(items, responses):
    from evaluation import muldim_official as official
    vectors = []
    patterns = defaultdict(list)
    difficulties = defaultdict(list)
    categories = defaultdict(list)
    for item, response in zip(items, responses):
        judges = [bool(official.get_instance(f"{constraint[0]}_{constraint[1]}").check(constraint[-1], response))
                  for constraint in item["constraints"]]
        vectors.append(judges)
        patterns[item["constraint_pattern"]].append(judges)
        difficulties[item["difficulty"]].append(judges)
        per_category = defaultdict(list)
        for constraint, passed in zip(item["constraints"], judges):
            key = f"{constraint[0]}_{constraint[1]}"
            per_category[official.constraint_mapping[key]].append(passed)
        for category, category_judges in per_category.items():
            categories[category].append(category_judges)
    result = rates(vectors)
    result["by_category"] = {key: rates(value)["all_pass"] for key, value in sorted(categories.items())}
    result["by_pattern"] = {key: rates(value) for key, value in sorted(patterns.items())}
    result["by_difficulty"] = {key: rates(value) for key, value in sorted(difficulties.items())}
    return result


def percent_metrics(value):
    if isinstance(value, dict):
        return {key: percent_metrics(child) for key, child in value.items()}
    return round(value * 100, 2)


def flatten(value, prefix=""):
    flattened = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            flattened.update(flatten(child, name))
        else:
            flattened[name] = child
    return flattened


def score_directory(out_dir):
    random.seed(0)
    DetectorFactory.seed = 0
    results = {}
    total_tokens, total_samples, truncated = 0, 0, 0
    for benchmark in BENCHMARKS:
        items = inputs(benchmark)
        generated = records(out_dir / f"{benchmark}.jsonl.gz")
        if len(generated) != len(items) or any(record["key"] != key_of(benchmark, item)
                                                for record, item in zip(generated, items)):
            raise ValueError(f"{benchmark} generation keys do not match benchmark inputs")
        if any(len(record["samples"]) != 5 for record in generated):
            raise ValueError(f"{benchmark} requires five samples per question")
        per_sample = []
        for sample_index in range(5):
            responses = [record["samples"][sample_index]["response"] for record in generated]
            if benchmark == "muldimif":
                per_sample.append(score_muldim(items, responses))
            else:
                per_sample.append(score_standard(benchmark, items, responses))
        results[benchmark] = percent_metrics(average_dicts(per_sample))
        for record in generated:
            for sample in record["samples"]:
                total_tokens += sample["response_tokens"]
                total_samples += 1
                truncated += sample["finish_reason"] == "length"
    results["response_tokens_mean"] = round(total_tokens / total_samples, 2)
    results["truncation_rate"] = round(100 * truncated / total_samples, 2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    results = score_directory(args.out_dir)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    flat = flatten(results)
    with (args.out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
