"""Build the instruction-following training set from HIR-16K.

Downloads HIR_trainv1.jsonl from Hugging Face (sastpg/HIR-16K at a pinned revision, sha256 checked), keeps the
type1 (IFEval checkers) and type2 (IFBench checkers) rows, joins question and criteria with one of HIR's six layouts
(chosen with random.Random(20260927 + row index)), drops prompts longer than 2,048 chat-template tokens, and stores
every leave-one-criterion-out prompt for MRTPO. Writes hir_if/train.parquet and hir_if/manifest.json.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from mrtpo.hir import HIRQuestionBuilder

HERE = Path(__file__).resolve().parent
REPO_ID = "sastpg/HIR-16K"
REVISION = "2a95f69eb56cc47edc16a45f939cde479673a4cb"
FILENAME = "HIR_trainv1.jsonl"
RAW_SHA256 = "465a01c19dc29e2c8d1cf183ccf3135872f7ec94ef10b20b7eb35603164c183b"
# sha256 of the rows (one sort_keys JSON line per row), independent of the pandas / pyarrow version that writes the
# parquet; the paper's file (parquet sha256 35edcfff499ca3af2e36567d11c196421a054d78c67859ae58922128744c7f2e)
# has this content digest.
CONTENT_SHA256 = "d2ffa5d5f6741e0faf7181825a74320e7ac67e9c665e9d30c8623f1034f962dd"
TYPES = tuple(f"type{i}" for i in range(1, 7))
# The first tokenizer decides the length filter; the second is checked too (both drop 0 prompts).
MODELS = ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-7B-Instruct")
MAX_PROMPT_TOKENS = 2048


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(parquet):
    import pyarrow.parquet as pq

    digest = hashlib.sha256()
    for row in pq.read_table(parquet).to_pylist():
        digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return digest.hexdigest()


def download(raw_dir):
    from huggingface_hub import hf_hub_download

    target = raw_dir / FILENAME
    if not target.exists() or sha256(target) != RAW_SHA256:
        raw_dir.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", revision=REVISION)
        shutil.copyfile(cached, target)
    actual = sha256(target)
    if actual != RAW_SHA256:
        raise ValueError(f"{target}: sha256 {actual} != {RAW_SHA256}")
    return target


def tokenizer_for(model):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(
        model,
        allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*.jinja", "special_tokens_map.json"],
    )
    return AutoTokenizer.from_pretrained(path)


def prepare(input_path, output_dir, tokenizers):
    """tokenizers: {model name: object with apply_chat_template}; the first entry decides the length filter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = HIRQuestionBuilder()
    primary = next(iter(tokenizers))
    rows = []
    dropped = {model: [] for model in tokenizers}
    source_counts = Counter()
    q_types = Counter()
    criteria_total = 0
    with input_path.open() as stream:
        for row_index, line in enumerate(stream):
            source = json.loads(line)
            if source["source"] not in {"type1", "type2"}:
                continue
            source_counts[source["source"]] += 1
            criteria = source["criteria"]
            if len(criteria) != len(source["ground_truth"]["instruction_id_list"]) or len(criteria) != len(source["ground_truth"]["kwargs"]):
                raise ValueError(f"criteria and ground truth lengths differ at line {row_index}")
            q_type = random.Random(20260927 + row_index).choice(TYPES)
            question = builder._concat_question(source["prompt"], criteria, q_type)
            messages = [{"role": "user", "content": question}]
            primary_too_long = False
            for model, tokenizer in tokenizers.items():
                if len(tokenizer.apply_chat_template(messages, add_generation_prompt=True)) > MAX_PROMPT_TOKENS:
                    dropped[model].append(source["id"])
                    primary_too_long |= model == primary
            if primary_too_long:
                continue
            q_types[q_type] += 1
            criteria_total += len(criteria)
            ground_truth = dict(source["ground_truth"], source=source["source"])
            minus = [builder._concat_question(source["prompt"], criteria[:j] + criteria[j + 1:], q_type)
                     for j in range(len(criteria))]
            rows.append({
                "data_source": "hir_if",
                "ability": "instruction_following",
                "prompt": messages,
                "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
                "extra_info": {
                    "index": len(rows), "row_index": row_index, "id": source["id"],
                    "source": source["source"], "q_type": q_type, "n_criteria": len(criteria),
                    "criteria_json": json.dumps(criteria, ensure_ascii=False),
                    "minus_prompts_json": json.dumps(minus, ensure_ascii=False),
                },
            })
    parquet = output_dir / "train.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    manifest = {
        "input_rows": row_index + 1, "source_counts": dict(source_counts), "rows_kept": len(rows),
        "dropped_by_length": {model: len(ids) for model, ids in dropped.items()},
        "criteria_total": criteria_total, "q_type_distribution": dict(sorted(q_types.items())),
        "sha256": {"input": sha256(input_path), "rows": content_sha256(parquet)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    parser.add_argument("--output-dir", type=Path, default=HERE / "hir_if")
    args = parser.parse_args()
    raw = download(args.raw_dir)
    manifest = prepare(raw, args.output_dir, {model: tokenizer_for(model) for model in MODELS})
    print(json.dumps(manifest, indent=2))
    if manifest["sha256"]["rows"] != CONTENT_SHA256:
        raise SystemExit(f"row digest {manifest['sha256']['rows']} != {CONTENT_SHA256}")
    print(f"OK: {manifest['rows_kept']} rows, identical to the paper's training set")


if __name__ == "__main__":
    main()
