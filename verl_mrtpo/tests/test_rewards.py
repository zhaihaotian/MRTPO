import json
import sqlite3

import pytest

from mrtpo.reward.hir_reward import compute_score as hir_score
from mrtpo.reward.sql_reward import compute_score as sql_score
from mrtpo.reward.sql_reward import format_reward
from mrtpo.sql.constructions import FORMAT_BLOCK, execution_sql_block
from mrtpo.sql.runtime import extract_sql


def test_hir_reward_one_score_per_criterion_in_order():
    truth = {"source": "type1", "instruction_id_list": ["startend:end_checker", "startend:quotation"],
             "kwargs": [{"end_phrase": "Done."}, {}]}
    result = hir_score("hir_if", '"Done."', json.dumps(truth), {"n_criteria": 2})
    assert result["criteria_scores"] == "[1.0, 1.0]" and result["score"] == result["all_pass"] == 1.0
    result = hir_score("hir_if", "Done.", json.dumps(truth), {"n_criteria": 2})
    assert json.loads(result["criteria_scores"]) == [1.0, 0.0] and result["score"] == 0.5
    assert result["checker_errors"] == 0


def test_hir_reward_ifbench_checker():
    truth = {"source": "type2", "instruction_id_list": ["count:word_count_range", "format:no_bullets_bullets"],
             "kwargs": [{"min_words": 3, "max_words": 6}, {}]}
    result = hir_score("hir_if", "one two three four", json.dumps(truth), {"n_criteria": 2})
    assert result["n_criteria"] == 2 and result["checker_errors"] == 0
    assert json.loads(result["criteria_scores"])[0] == 1.0


def answer(sql, think="draft"):
    return f"<think>{think}</think>\n<answer>```sql\n{sql}\n```</answer>"


@pytest.fixture
def sql_case(tmp_path, monkeypatch):
    database = tmp_path / "db/tiny/tiny.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b"), (2, "b")])
    monkeypatch.setenv("SQL_DB_ROOT", str(tmp_path))
    return json.dumps({"gold_sql": "SELECT id FROM items", "db_rel": "db/tiny/tiny.sqlite"})


@pytest.mark.parametrize("text,expected", [
    (answer("SELECT 1"), 1),
    ("<think>x</think><answer>```sql\nSELECT 1\n```", 0),
    ("<think>x</think><think>y</think><answer>```sql\nSELECT 1\n```</answer>", 0),
    (answer(" "), 0),
])
def test_sql_format_rule(text, expected):
    assert format_reward(text) == expected


def test_sql_extraction_uses_last_block_after_answer():
    text = "<think>```sql\nSELECT 1\n```</think><answer>```sql\nSELECT 3\n```\n```sql\nSELECT 2\n```</answer>"
    assert extract_sql(text) == "SELECT 2"


@pytest.mark.parametrize("text,scores,error", [
    (answer("SELECT id FROM items ORDER BY id DESC"), [1, 1, 1], False),
    (answer("SELECT name FROM items"), [1, 1, 0], False),
    (answer("SELECT missing FROM items"), [1, 0, 0], True),
    ("<think>x</think><answer>no query</answer>", [0, 0, 0], True),
])
def test_sql_reward_scores_and_replay_blocks(sql_case, text, scores, error):
    result = sql_score("spider_sql", text, sql_case, {"n_criteria": 3})
    assert json.loads(result["criteria_scores"]) == scores
    assert result["score"] == pytest.approx(sum(scores) / 3) and result["checker_errors"] == 0
    blocks = json.loads(result["replay_blocks_json"])
    assert blocks[0] == FORMAT_BLOCK and blocks[2] == execution_sql_block("SELECT id FROM items")
    assert bool(blocks[1]) == error
