"""Download ToolRL's rlla_4k split and add the MRTPO replay reference.

1. Downloads train.parquet (3,920 rows) and test.parquet (80 rows) of ToolRL's `dataset/rlla_4k` at a pinned commit
   and checks their sha256.
2. Writes <data-dir>/train.parquet and test.parquet: every original column unchanged, plus
   `extra_info.teacher_output` = the label with int / float / boolean argument values that the label writes as
   plain-literal strings ("8", "2.5", "true") converted to the type the row's tool schema declares. The reward keeps
   reading the original label (`reward_model.ground_truth`); only the MRTPO replay shows `teacher_output`.

usage: python scripts/fetch_data.py [--data-dir data/rlla_4k]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

REVISION = '8cee13ec0ca72f0461da372a93a6fd8140dbb840'
SOURCE = f'https://raw.githubusercontent.com/qiancheng0/ToolRL/{REVISION}/dataset/rlla_4k'
SHA256 = {
    'train': 'f2e728ad4a379550887711870db5f97154a0ecc45efbe5d43db8a99dc3b9e1c8',
    'test': 'f7f300ebe755fd04985c83e18097d5a94a08619d72809fe06128ed3f00da7554',
}

_INT = re.compile(r"-?\d+")
_NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
_BASE_TYPES = {"int": "int", "integer": "int", "float": "float", "number": "float", "boolean": "bool", "bool": "bool"}


def declared_kind(spec: Any) -> Optional[str]:
    """'int', 'float', 'bool' or None from a parameter schema such as {"type": "int, optional"}."""
    if not isinstance(spec, dict):
        return None
    name = str(spec.get("type", "")).lower().replace(", optional", "").strip()
    return _BASE_TYPES.get(name)


def coerce_value(value: Any, kind: Optional[str]) -> Any:
    """Convert a plain-literal string to the declared type; anything else is returned unchanged."""
    if not isinstance(value, str) or kind is None:
        return value
    text = value.strip()
    if kind == "int" and _INT.fullmatch(text):
        return int(text)
    if kind == "float" and _NUMBER.fullmatch(text):
        return json.loads(text)
    if kind == "bool" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    return value


def tool_schemas(instruction: str) -> Dict[str, dict]:
    """Parameter schemas of the tools listed in the system prompt, by tool name."""
    tools = instruction.split("**Steps for Each Turn**", 1)[0]
    schemas = {}
    for block in re.split(r"\n(?=\d+\. Name: )", tools):
        match = re.match(r"\d+\. Name: (.*?)\nDescription: .*?\nParameters: (\{.*\})\s*$", block, re.S)
        if match:
            schemas[match.group(1).strip()] = json.loads(match.group(2))
    return schemas


def typed_label(label: str, instruction: str, counts: Optional[Counter] = None) -> str:
    """Rewrite every <tool_call> line of `label` with schema-typed argument values; other text is kept verbatim."""
    schemas = tool_schemas(instruction)
    counts = Counter() if counts is None else counts

    def fix_block(match: re.Match) -> str:
        lines = []
        for line in match.group(1).split("\n"):
            try:
                call = json.loads(line)
                params = call["parameters"]
                schema = schemas.get(call["name"], {})
            except (ValueError, TypeError, KeyError):
                lines.append(line)
                continue
            changed = False
            for key, value in params.items():
                kind = declared_kind(schema.get(key))
                new = coerce_value(value, kind)
                if new is not value:
                    params[key] = new
                    changed = True
                    counts[f"converted_{kind}"] += 1
                elif isinstance(value, str) and kind:
                    counts["left_as_string"] += 1
            lines.append(json.dumps(call, ensure_ascii=False) if changed else line)
        return "<tool_call>" + "\n".join(lines) + "</tool_call>"

    return re.sub(r"<tool_call>(.*?)</tool_call>", fix_block, label, flags=re.S)


def add_reference(src: Path, dst: Path) -> Counter:
    """Copy one split, adding extra_info.teacher_output."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(src)
    rows = table.to_pylist()
    counts = Counter(rows=len(rows))
    for row in rows:
        label = row["reward_model"]["ground_truth"]
        if row["extra_info"]["output"] != label:
            raise ValueError("extra_info.output must equal reward_model.ground_truth")
        reference = typed_label(label, row["prompt"][0]["content"], counts)
        row["extra_info"]["teacher_output"] = reference
        counts["rows_changed"] += reference != label
    extra = table.schema.field("extra_info").type
    schema = table.schema.set(table.schema.get_field_index("extra_info"),
                              pa.field("extra_info", pa.struct(list(extra) + [pa.field("teacher_output", pa.string())])))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema.remove_metadata()), dst)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/rlla_4k')
    args = parser.parse_args()
    raw = args.data_dir / 'original'
    raw.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ('train', 'test'):
        path = raw / f'{split}.parquet'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256[split]:
            with urllib.request.urlopen(f'{SOURCE}/{split}.parquet', timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != SHA256[split]:
                raise ValueError(f'{split}.parquet: sha256 differs from the pinned ToolRL file')
            path.write_bytes(payload)
        stats[split] = dict(add_reference(path, args.data_dir / f'{split}.parquet'))
        print(f"{split}: {stats[split]['rows']} rows, teacher_output differs from the label in "
              f"{stats[split]['rows_changed']} rows -> {args.data_dir / f'{split}.parquet'}")
    (args.data_dir / 'teacher_output_stats.json').write_text(json.dumps(stats, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
