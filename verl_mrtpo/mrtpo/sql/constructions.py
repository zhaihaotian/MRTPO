"""SQL-R1 Spider prompt (P0), the format rule and the reference-query block used by the SQL reward and MRTPO replay."""

from __future__ import annotations

import json
from pathlib import Path


RESERVED = set(json.loads((Path(__file__).with_name("sql_r1_reserved_words.json")).read_text()))

# SQL-R1 src/utils/prepare_input_seq.py, prepare_input_output_pairs (Spider/dev).
P0_TEMPLATE = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Note that while the reasoning process and SQL query need to be enclosed within <think> </think> and <answer> </answer> tags respectively, this should not affect the quality of the SQL generation.
- The answer must contain the SQL query within ```sql ``` tags.

Output Format:
Put your reasoning inside <think> </think> tags and your answer inside <answer> </answer> tags, with the final SQL query in a ```sql code block inside <answer>:
<think>
...
</think>
<answer>
```sql
-- Your SQL query
```
</answer>

Take a deep breath and think step by step to find the correct SQL query.
'''

# The SQL-R1 inference template (src/utils/prepare_input_seq.py) asks for <think>/<answer> tags in one Instructions
# bullet but its Output Format shows only a ```sql block. Its training data (src/data_preprocess/nl2sql_synsql_*.py)
# instead shows a full tagged example and pre-fills <think>. We keep the inference template and make its Output Format
# show the tagged form the format reward checks, so the visible instruction and the reward agree.
OLD_OUTPUT_FORMAT = """Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```
"""
OUTPUT_FORMAT = P0_TEMPLATE[P0_TEMPLATE.index("Output Format:"):P0_TEMPLATE.index("\nTake a deep breath")]


def upgrade_output_format(text: str) -> str:
    """Rewrite a prompt built with the old template so its Output Format shows the tagged answer form."""
    if text.count(OLD_OUTPUT_FORMAT) != 1:
        raise ValueError("prompt does not contain exactly one old Output Format section")
    return text.replace(OLD_OUTPUT_FORMAT, OUTPUT_FORMAT)


FORMAT_BLOCK = '''For the final user request, respond in this form:

<think>...</think>
<answer>
```sql
...
```
</answer>

Use exactly one <think> block followed by exactly one <answer> block, and put the final SQL query in a ```sql code block inside <answer>.'''


def _identifier(name: str) -> str:
    return ('`' + name.replace('`', '``') + '`'
            if name.upper() in RESERVED or any(not (c.isascii() and (c.isalnum() or c == '_')) for c in name)
            else name)


def schema_ddl(db_info: dict, sampled_values: dict[str, list]) -> str:
    """SQL-R1 Spider/dev schema from its released DB-info and value caches."""
    names = db_info["table_names_original"]
    cols = db_info["column_names_original"]
    types = db_info["column_types"]
    primary = {i for entry in db_info["primary_keys"] for i in (entry if isinstance(entry, list) else [entry])}
    statements = []
    for ti, table in enumerate(names):
            lines = []
            pk = []
            for ci, (owner, col) in enumerate(cols):
                if owner != ti:
                    continue
                quoted_c = _identifier(col)
                values = sampled_values.get(f"{table}.{col}".lower(), [])[:6]
                comment = db_info["column_names"][ci][1]
                if col.lower() in (comment.lower(), comment.lower().replace(' ', '_'), comment.lower().replace(' ', '')) or not comment.strip():
                    line = f"    {quoted_c} {types[ci]},"
                    if values:
                        line += f" -- example: {values}"
                else:
                    line = f"    {quoted_c} {types[ci]}, -- {comment}"
                    if values:
                        line += f", example: {values}"
                lines.append(line)
                if ci in primary:
                    pk.append(quoted_c)
            if pk:
                lines.append(f"    PRIMARY KEY ({', '.join(pk)}),")
            fk_lines = []
            for ci, (owner, _) in enumerate(cols):
                if owner != ti:
                    continue
                for src, dst in db_info["foreign_keys"]:
                    if src == ci:
                        target_table = names[cols[dst][0]]
                        fk_lines.append(
                            f"    CONSTRAINT fk_{table.lower().replace(' ', '_')}_{cols[src][1].lower().replace(' ', '_')} "
                            f"FOREIGN KEY ({_identifier(cols[src][1])}) REFERENCES {_identifier(target_table)} "
                            f"({_identifier(cols[dst][1])}),"
                        )
            lines.extend(dict.fromkeys(fk_lines))
            if lines:
                lines[-1] = lines[-1].removesuffix(',')
                statements.append(f"CREATE TABLE {_identifier(table)} (\n" + "\n".join(lines) + "\n);")
    return "\n\n".join(statements)


def p0_messages(question: str, schema: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": P0_TEMPLATE.format(question=question, schema=schema)}]


def execution_sql_block(gold_sql: str) -> str:
    """The reference SQL consumed by the execution scorer."""
    return f"The reference query for this question is: `{gold_sql}`."


