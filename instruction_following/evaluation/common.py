"""Benchmark input paths and record loading."""

import gzip
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
BENCHMARKS = {
    "ifeval": DATA / "ifeval/input_data.jsonl",
    "ifbench": DATA / "ifbench/IFBench_test.jsonl",
    "muldimif": DATA / "muldimif/test.json",
}


def inputs(benchmark):
    path = BENCHMARKS[benchmark]
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def prompt_of(benchmark, item):
    return item["conversations"][0]["content"] if benchmark == "muldimif" else item["prompt"]


def key_of(benchmark, item):
    return str(item["id"] if benchmark == "muldimif" else item["key"])


def records(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]
