"""Greedy execution-accuracy evaluation of one model on the six Text-to-SQL sets (one GPU).

For each set it writes <out>/<set>/result.json and responses.jsonl. Only records whose gold SQL runs are scored.
A prediction is the last ```sql block after the last <answer> tag; it is correct when it runs within 30 s and its
result-row set equals the gold row set (gold SQL timeout 60 s). Sets that already have a result.json are skipped.
"""
import argparse
import collections
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from mrtpo.reward.sql_reward import format_reward
from mrtpo.sql.runtime import execute, extract_sql

HERE = Path(__file__).resolve().parents[1]
SETS = 'spider_dev,spider_test,spider_syn,spider_realistic,spider_dk,bird_dev'
GOLD = {}
DB_ROOT = None


def db_path(record):
    parts = Path(record['db_path']).parts
    return DB_ROOT / Path(*parts[parts.index('raw') + 1:])


def init(root, gold_rows=None):
    global DB_ROOT, GOLD
    DB_ROOT = root
    GOLD = gold_rows or {}


def gold(record):
    rows, error = execute(record['gold_sql'], db_path(record), timeout=60.0)
    return (record['set'], record['idx']), (set(rows) if error is None else None)


def score(job):
    response, record = job
    sql = extract_sql(response)
    if sql is None:
        return None, None
    rows, error = execute(sql, db_path(record), timeout=30.0)
    if error is not None:
        return 'ERR', None
    result = set(rows)
    return result, result == GOLD[(record['set'], record['idx'])]


def evaluate(model, datasets_dir, names, out_dir, db_root, workers):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, generation_config='vllm', dtype='bfloat16',
              max_model_len=8192, gpu_memory_utilization=0.88, seed=0)
    tokenizer = llm.get_tokenizer()
    for name in names:
        out = out_dir / name
        if (out / 'result.json').exists():
            continue
        with (datasets_dir / f'{name}.jsonl').open() as stream:
            records = [json.loads(line) for line in stream]
        records = [record for record in records if record['gold_ok']]
        with Pool(workers, initializer=init, initargs=(db_root,)) as pool:
            gold_rows = dict(pool.map(gold, records, chunksize=4))
        prompts = [tokenizer.apply_chat_template(record['messages'], tokenize=False, add_generation_prompt=True)
                   for record in records]
        start = time.time()
        outputs = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=2048))
        seconds = time.time() - start
        jobs = [(output.outputs[0].text, record) for record, output in zip(records, outputs)]
        with Pool(workers, initializer=init, initargs=(db_root, gold_rows)) as pool:
            scored = pool.map(score, jobs, chunksize=8)
        count = len(jobs)
        result = {
            'set': name, 'model': model, 'questions': count, 'gen_seconds': round(seconds),
            'accuracy': round(100 * sum(bool(ok) for _, ok in scored) / count, 2),
            'executable': round(100 * sum(rows not in (None, 'ERR') for rows, _ in scored) / count, 2),
            'format_sqlr1': round(100 * sum(format_reward(text) for text, _ in jobs) / count, 2),
            'truncated': round(100 * sum(o.outputs[0].finish_reason == 'length' for o in outputs) / count, 2),
        }
        groups = collections.defaultdict(list)
        for (_, record), (_, ok) in zip(jobs, scored):
            if record.get('difficulty'):
                groups[record['difficulty']].append(bool(ok))
        if groups:
            result['accuracy_by_difficulty'] = {k: round(100 * sum(v) / len(v), 2) for k, v in groups.items()}
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'responses.jsonl').open('w') as stream:
            for (text, record), (rows, correct) in zip(jobs, scored):
                stream.write(json.dumps({'idx': record['idx'], 'text': text, 'correct': bool(correct),
                                         'executable': rows not in (None, 'ERR')}) + '\n')
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Hugging Face model directory, e.g. <run>/hf/global_step_100')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data', help='data root with datasets/ and raw/')
    parser.add_argument('--sets', default=SETS)
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    data = args.data_dir.resolve()
    evaluate(args.model, data / 'datasets', args.sets.split(','), args.out, data / 'raw', args.workers)


if __name__ == '__main__':
    main()
