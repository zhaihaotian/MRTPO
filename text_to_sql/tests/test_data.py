"""CPU checks of the data construction and evaluation scoring on a tiny SQLite database."""
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / 'data'), str(HERE / 'evaluation')]

import build_jsonl  # noqa: E402
import evaluate  # noqa: E402
import make_parquet  # noqa: E402


def make_db(root, db_id='pets'):
    path = root / 'raw/spider_data/database' / db_id / f'{db_id}.sqlite'
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE pet (id INTEGER PRIMARY KEY, kind TEXT, age INTEGER);"
                       "INSERT INTO pet VALUES (1,'dog',3),(2,'cat',5),(3,'dog',7),(4,'',2);")
    conn.commit()
    conn.close()
    return path


def test_sample_table_values(tmp_path):
    values = build_jsonl.sample_table_values(make_db(tmp_path), ['pet'])
    assert values == {'pet.id': [1, 2], 'pet.kind': ['dog', 'cat'], 'pet.age': [3, 5]}


def test_reference_solution_rule(tmp_path):
    make_db(tmp_path)
    cot = [
        {'question': 'How many dogs?', 'db_id': 'pets', 'query': "SELECT count(*) FROM pet WHERE kind = 'dog'",
         'cot': "Count the dogs.\n\n```sql\nSELECT COUNT(*) FROM pet WHERE kind='dog';\n```\nDone."},
        {'question': 'How many dogs?', 'db_id': 'pets', 'query': 'SELECT 0', 'cot': 'second copy, never used'},
        {'question': 'Oldest age?', 'db_id': 'pets', 'query': 'SELECT max(age) FROM pet',
         'cot': 'Wrong.\n```sql\nSELECT min(age) FROM pet\n```'},
    ]
    (tmp_path / 'cot.json').write_text(json.dumps(cot))
    records = [{'question': 'How many dogs?', 'gold_sql': "SELECT count(*) FROM pet WHERE kind = 'dog'"},
               {'question': 'Oldest age?', 'gold_sql': 'SELECT max(age) FROM pet'}]
    refs = make_parquet.reference_solutions(records, tmp_path / 'cot.json', tmp_path / 'raw')
    assert refs[0] == ("<think>\nCount the dogs.\n</think>\n<answer>\n```sql\n"
                       "SELECT count(*) FROM pet WHERE kind = 'dog'\n```\n</answer>")
    assert refs[1] == '<think>\n\n</think>\n<answer>\n```sql\nSELECT max(age) FROM pet\n```\n</answer>'


def test_parquet_rows():
    record = {'set': 'train', 'idx': 7, 'db_path': 'raw/spider_data/database/pets/pets.sqlite',
              'gold_sql': 'SELECT 1', 'messages': [{'role': 'user', 'content': 'P0'}]}
    row = make_parquet.rows([record])[0]
    assert json.loads(row['reward_model']['ground_truth']) == {'gold_sql': 'SELECT 1',
                                                               'db_rel': 'spider_data/database/pets/pets.sqlite'}
    assert row['extra_info'] == {'index': 0, 'id': 'train:7', 'n_criteria': 3, 'replay_mode': 'privileged',
                                 'p0_text': 'P0', 'gold_sql': 'SELECT 1'}


def test_evaluation_scoring(tmp_path):
    make_db(tmp_path)
    record = {'set': 'spider_dev', 'idx': 0, 'db_path': 'raw/spider_data/database/pets/pets.sqlite',
              'gold_sql': "SELECT id FROM pet WHERE kind = 'dog'"}
    evaluate.init(tmp_path / 'raw')
    key, rows = evaluate.gold(record)
    evaluate.init(tmp_path / 'raw', {key: rows})
    answer = '<think>x</think>\n<answer>\n```sql\n{}\n```\n</answer>'
    assert evaluate.score((answer.format("SELECT id FROM pet WHERE kind='dog' ORDER BY id DESC"), record))[1] is True
    assert evaluate.score((answer.format('SELECT id FROM pet'), record))[1] is False
    assert evaluate.score((answer.format('SELECT nope FROM pet'), record)) == ('ERR', None)
    assert evaluate.score(('no sql here', record)) == (None, None)
