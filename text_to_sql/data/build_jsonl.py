"""Build the Spider training set and the six evaluation sets with SQL-R1 prompts, one JSONL file each.

Each record: set, idx, db_id, db_path (relative to the data root), question, gold_sql, messages (one user turn with
the SQL-R1 prompt), [difficulty for BIRD], gold_ok (the gold SQL runs within 30 s; only these records are scored), gold_rows.

Schemas: Spider dev / Syn / Realistic use SQL-R1's Spider-dev cache, Spider test its Spider-test cache, BIRD dev its
BIRD cache; the training set and Spider-DK (whose 3 new databases SQL-R1 does not cover) read the schema from
tables.json and sample up to 2 distinct non-empty values per column from the database, as SQL-R1 does.
"""
import argparse
import json
import os
import statistics
from multiprocessing import Pool
from pathlib import Path

from mrtpo.sql.constructions import p0_messages, schema_ddl
from mrtpo.sql.runtime import execute, sqlite3

ROOT = Path(__file__).resolve().parent
# BIRD dev records whose gold SQL did not complete when the paper's evaluation sets were built; they are left out of
# scoring (gold_ok = false) so every set has the paper's question count (BIRD dev 1,532).
UNSCORED = {('bird_dev', 518): 'not scored in the paper', ('bird_dev', 701): 'not scored in the paper'}
SET_NAMES = ('train', 'spider_dev', 'spider_test', 'spider_syn', 'spider_realistic', 'spider_dk', 'bird_dev')


def sample_table_values(db, tables, limit=2):
    """SQL-R1 src/utils/prepare_input_seq.py sample_table_values (distinct non-empty values, strings cut to 40)."""
    out = {}
    conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    cur = conn.cursor()
    for table in tables:
        cur.execute(f'PRAGMA table_info(`{table}`);')
        for col in [c[1] for c in cur.fetchall()]:
            cur.execute(f"SELECT `{col}` FROM (SELECT DISTINCT `{col}` FROM `{table}` "
                        f"WHERE `{col}` IS NOT NULL and `{col}` != '') AS u LIMIT {limit};")
            values = [v[0][:40] if isinstance(v[0], str) else v[0] for v in cur.fetchall()]
            if values:
                out[f'{table}.{col}'.lower()] = values
    conn.close()
    return out


def load(path):
    return json.loads(Path(path).read_text())


def sources(raw):
    sp, r1, var, bird = raw / 'spider_data', raw / 'sqlr1_db_info', raw / 'variants', raw / 'bird_dev_hf/dev_20240627'

    def spider_db(db):
        return Path('raw/spider_data/database') / db / f'{db}.sqlite'

    def r1_cache(name):
        return load(r1 / f'{name}_db_id2db_info.json'), load(r1 / f'{name}_db_id2sampled_db_values.json')

    question = lambda x: x['question']  # noqa: E731
    query = lambda x: x['query']  # noqa: E731
    return {
        # name: (records, question, gold SQL, db path relative to the data root, schema source), loaded on demand
        'train': lambda: (load(sp / 'train_spider.json'), question, query, spider_db, sp / 'tables.json'),
        'spider_dev': lambda: (load(sp / 'dev.json'), question, query, spider_db, r1_cache('spiderdev')),
        'spider_test': lambda: (load(sp / 'test.json'), question, query,
                        lambda db: Path('raw/spider_data/test_database') / db / f'{db}.sqlite', r1_cache('spidertest')),
        'spider_syn': lambda: (load(var / 'spider_syn_dev.json'), lambda x: x['SpiderSynQuestion'], query, spider_db,
                       r1_cache('spiderdev')),
        'spider_realistic': lambda: (load(var / 'spider_realistic.json'), question, query, spider_db, r1_cache('spiderdev')),
        'spider_dk': lambda: (load(var / 'spider_dk.json'), question, query,
                      lambda db: (Path('raw/variants/dk_database') if db.startswith('new_')
                                  else Path('raw/spider_data/database')) / db / f'{db}.sqlite',
                      var / 'spider_dk_tables.json'),
        'bird_dev': lambda: (load(bird / 'dev.json'),
                     lambda x: (x['evidence'] + '\n' + x['question']) if x['evidence'].strip() else x['question'],
                     lambda x: x['SQL'], lambda db: Path('raw/bird_dev_hf/dev_20240627/dev_databases') / db / f'{db}.sqlite',
                     r1_cache('bird')),
    }


def gold_check(job):
    sql, path = job
    rows, error = execute(sql, Path(path), timeout=30.0)
    return error, (len(rows) if rows is not None else None)


def build(root, names, workers):
    raw, out = root / 'raw', root / 'datasets'
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, load_set in sources(raw).items():
        if name not in names:
            continue
        data, question, gold, path, schema = load_set()
        dbs = sorted({x['db_id'] for x in data})
        if isinstance(schema, Path):
            tables = {t['db_id']: t for t in load(schema)}
            infos = {db: tables[db] for db in dbs}
            values = {db: sample_table_values(root / path(db), infos[db]['table_names_original']) for db in dbs}
        else:
            infos, values = schema
        missing = [db for db in dbs if db not in infos or not (root / path(db)).exists()]
        records = []
        for i, x in enumerate(data):
            db = x['db_id']
            if db in missing:
                continue
            records.append({'set': name, 'idx': i, 'db_id': db, 'db_path': path(db).as_posix(), 'question': question(x),
                            'gold_sql': gold(x),
                            'messages': p0_messages(question(x), schema_ddl(infos[db], values.get(db, {}))),
                            **({'difficulty': x['difficulty']} if 'difficulty' in x else {})})
        with Pool(workers) as pool:
            checks = pool.map(gold_check, [(r['gold_sql'], str(root / r['db_path'])) for r in records], chunksize=4)
        failed = []
        for record, (error, n) in zip(records, checks):
            if (name, record['idx']) in UNSCORED:
                error, n = UNSCORED[(name, record['idx'])], None
            record['gold_ok'] = error is None
            record['gold_rows'] = n
            if error:
                failed.append((record['idx'], error[:80]))
        with open(out / f'{name}.jsonl', 'w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        chars = [len(r['messages'][0]['content']) for r in records]
        report[name] = {'questions_in_source': len(data), 'kept': len(records), 'gold_ok': len(records) - len(failed),
                        'databases': len(dbs), 'missing_databases': missing, 'gold_sql_failed': failed,
                        'prompt_chars_mean': round(statistics.mean(chars)), 'prompt_chars_max': max(chars)}
        print(name, json.dumps({k: v for k, v in report[name].items() if k != 'gold_sql_failed'}), flush=True)
    previous = load(out / 'BUILD_REPORT.json') if (out / 'BUILD_REPORT.json').exists() else {}
    (out / 'BUILD_REPORT.json').write_text(json.dumps({**previous, **report}, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='data root holding raw/ (default: this folder)')
    parser.add_argument('--sets', default=','.join(SET_NAMES))
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    names = args.sets.split(',')
    unknown = set(names) - set(SET_NAMES)
    if unknown:
        parser.error(f'unknown sets: {sorted(unknown)}')
    build(args.root.resolve(), names, args.workers)


if __name__ == '__main__':
    main()
