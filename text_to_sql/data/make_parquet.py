"""Write the verl training parquets from datasets/train_filtered.jsonl.

parquet/train.parquet (GRPO, GDPO, MRTPO, SAW): data_source "spider_sql", prompt = the SQL-R1 user turn,
reward_model.ground_truth = {"gold_sql", "db_rel"} (database path relative to SQL_DB_ROOT = <root>/raw),
extra_info = {index, id, n_criteria=3, replay_mode="privileged", p0_text, gold_sql}.

parquet/train_opsd.parquet (OPSD): the same rows plus extra_info.ref_solution, the OmniSQL reference wrapped in the
answer format,
    <think>\n{reasoning}\n</think>\n<answer>\n```sql\n{gold_sql}\n```\n</answer>
where {reasoning} is the OmniSQL chain of thought for the question up to its first ```sql block (stripped). The
chain of thought is looked up by question text (the first OmniSQL record with that question) and is kept only if the
last SQL block of that chain of thought returns the same row set as that record's gold SQL on that record's database;
otherwise {reasoning} is empty.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from mrtpo.sql.runtime import execute

ROOT = Path(__file__).resolve().parent
SQL_BLOCK = re.compile(r'```sql(.*?)```', re.DOTALL | re.IGNORECASE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(records):
    out = []
    for record in records:
        messages = record['messages']
        db_rel = Path(record['db_path']).relative_to('raw').as_posix()
        out.append({
            'data_source': 'spider_sql', 'ability': 'text_to_sql', 'prompt': messages,
            'reward_model': {'style': 'rule', 'ground_truth': json.dumps(
                {'gold_sql': record['gold_sql'], 'db_rel': db_rel}, ensure_ascii=False)},
            'extra_info': {'index': len(out), 'id': f"{record['set']}:{record['idx']}", 'n_criteria': 3,
                           'replay_mode': 'privileged', 'p0_text': messages[0]['content'],
                           'gold_sql': record['gold_sql']},
        })
    return out


def reference_solutions(records, cot_path, raw):
    """OmniSQL chain of thought + gold SQL in the <think>/<answer> format (see the module docstring)."""
    first = {}
    for entry in json.loads(Path(cot_path).read_text()):
        first.setdefault(entry['question'], entry)
    verified = {}

    def reasoning(entry):
        key = entry['question']
        if key not in verified:
            db = raw / 'spider_data/database' / entry['db_id'] / f"{entry['db_id']}.sqlite"
            blocks = SQL_BLOCK.findall(entry['cot'])
            gold, _ = execute(entry['query'], db, timeout=30.0)
            pred, _ = execute(blocks[-1].strip(), db, timeout=30.0) if blocks else (None, 'no sql')
            ok = gold is not None and pred is not None and set(pred) == set(gold)
            cut = entry['cot'].find('```sql')
            verified[key] = (entry['cot'][:cut] if cut >= 0 else entry['cot']).strip() if ok else ''
        return verified[key]

    out = []
    for record in records:
        think = reasoning(first[record['question']])
        out.append(f"<think>\n{think}\n</think>\n<answer>\n```sql\n{record['gold_sql']}\n```\n</answer>")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='data root (default: this folder)')
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'datasets/train_filtered.jsonl'
    records = [json.loads(line) for line in open(source)]
    out_dir = root / 'parquet'
    out_dir.mkdir(parents=True, exist_ok=True)
    table = rows(records)
    pd.DataFrame(table).to_parquet(out_dir / 'train.parquet', index=False)
    references = reference_solutions(records, root / 'raw/omnisql/train_spider_enhanced_with_cot.json', root / 'raw')
    opsd = [dict(row, extra_info=dict(row['extra_info'], ref_solution=ref)) for row, ref in zip(table, references)]
    pd.DataFrame(opsd).to_parquet(out_dir / 'train_opsd.parquet', index=False)
    manifest = {'rows': len(table), 'reference_with_reasoning': sum(not r.startswith('<think>\n\n</think>')
                                                                    for r in references),
                'sha256': {'train_filtered.jsonl': sha256(source),
                           'train.parquet': sha256(out_dir / 'train.parquet'),
                           'train_opsd.parquet': sha256(out_dir / 'train_opsd.parquet')}}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
