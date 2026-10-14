"""Three ordered Spider criteria for verl: format, executable, execution."""

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from mrtpo.sql.constructions import FORMAT_BLOCK, execution_sql_block
from mrtpo.sql.runtime import execute, extract_sql

LOGGER = logging.getLogger(__name__)
_logged_checker_error = False
_FORMAT_SQL_RE = re.compile(r"```sql(.*?)```", re.DOTALL)


def format_reward(response):
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    if any(response.count(tag) != 1 for tag in tags):
        return 0
    positions = [response.index(tag) for tag in tags]
    if positions != sorted(positions):
        return 0
    answer = response[positions[2] + len(tags[2]):positions[3]]
    return int(any(sql.strip() for sql in _FORMAT_SQL_RE.findall(answer)))


@lru_cache(maxsize=None)
def _gold_rows(db_path, gold_sql):
    rows, error = execute(gold_sql, Path(db_path), timeout=30.0)
    if error is not None:
        raise ValueError(f"gold SQL failed: {error}")
    return set(rows)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Score a response and return its three privileged replay blocks."""
    global _logged_checker_error
    # Block k is the privileged context for criterion k of scores = [format, executable, execution]:
    # the format rule, this response's own SQLite error (empty when it executes), the gold SQL.
    blocks = [FORMAT_BLOCK, "", ""]
    try:
        truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        gold_sql = truth["gold_sql"]
        blocks[2] = execution_sql_block(gold_sql)
        db_path = Path(os.environ["SQL_DB_ROOT"]) / truth["db_rel"]
        sql = extract_sql(solution_str)
        format_score = format_reward(solution_str)
        executable = execution = 0
        sql_error = ""
        if sql is None:
            sql_error = "This response contains no ```sql block inside <answer>."
            blocks[1] = sql_error
        else:
            rows, error = execute(sql, db_path, timeout=30.0)
            if error is not None:
                sql_error = error
                blocks[1] = f"When this query was executed on this database, SQLite raised: `{error}`."
            else:
                executable = 1
                execution = int(set(rows) == _gold_rows(str(db_path), gold_sql))
        scores = [format_score, executable, execution]
        return {"score": sum(scores) / 3, "n_criteria": 3,
                "criteria_scores": json.dumps(scores), "all_pass": float(all(scores)),
                "sql_error": sql_error, "replay_blocks_json": json.dumps(blocks),
                "checker_errors": 0}
    except Exception:
        if not _logged_checker_error:
            LOGGER.exception("SQL reward checker failed; subsequent failures are counted")
            _logged_checker_error = True
        return {"score": 0.0, "n_criteria": 3, "criteria_scores": json.dumps([0, 0, 0]),
                "all_pass": 0.0, "sql_error": "", "replay_blocks_json": json.dumps(blocks),
                "checker_errors": 1}
