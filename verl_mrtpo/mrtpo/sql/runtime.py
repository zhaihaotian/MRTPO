"""SQLite execution and SQL extraction for the Text-to-SQL reward."""

import re
import time
from pathlib import Path

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3


SQL_RE = re.compile(r"```sql(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(response: str) -> str | None:
    """Use the last SQL block after the final answer tag."""
    start = response.rfind("<answer>")
    matches = SQL_RE.findall(response[max(start, 0):])
    return matches[-1].strip() if matches and matches[-1].strip() else None


def execute(sql: str, db_path: Path, timeout: float = 5.0) -> tuple[list[tuple] | None, str | None]:
    """Return result rows or the SQLite error from a read-only database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        return conn.execute(sql).fetchall(), None
    except (sqlite3.Error, sqlite3.Warning) as error:
        return None, str(error)
    finally:
        conn.close()
