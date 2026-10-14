"""Re-judge AdvancedIF rows whose judge call failed (timeout, server error or unparseable output).
usage: retry_failed.py split OUT_DIR  -> OUT_DIR/retry_in.jsonl (the original rows), prints their number
       retry_failed.py merge OUT_DIR  -> puts the retried rows back in place in advancedif_judged.jsonl (first pass kept
                                         as advancedif_judged.first_pass.jsonl), prints replaced / still failed"""
import json
import sys
from pathlib import Path

mode, out = sys.argv[1], Path(sys.argv[2])
judged = out/'advancedif_judged.jsonl'
rows = [json.loads(l) for l in judged.open()]
key = lambda r: (r['benchmark_name'], r['prompt_index'], r['sample_index'])
idx = [i for i, r in enumerate(rows) if not r['judge_result']['success']]
if mode == 'split':
    with (out/'retry_in.jsonl').open('w') as f:
        for i in idx:
            f.write(json.dumps({k: v for k, v in rows[i].items() if k != 'judge_result'}) + '\n')
    print(len(idx))
else:
    new = {key(r): r for r in map(json.loads, (out/'retry_out.jsonl').open())}
    first_pass = out/'advancedif_judged.first_pass.jsonl'
    if not first_pass.exists():
        first_pass.write_text(judged.read_text())
    for i in idx:
        rows[i] = new.get(key(rows[i]), rows[i])
    temporary = judged.with_name(judged.name + '.partial')
    with temporary.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    temporary.replace(judged)
    print(f'retried {len(idx)}, still failed {sum(not r["judge_result"]["success"] for r in rows)}')
