"""AdvancedIF metrics from OUT_DIR/advancedif_judged.jsonl, per subset and overall, written to OUT_DIR/summary.json.

Overall (all_rubrics_pass): share of the 8,225 responses whose verdict is that every rubric is satisfied.
Rubric (micro_rubric_pass): share of all checked rubrics that are satisfied.
As in the official code, a response whose judge call failed counts as not passed."""
import json
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
per = defaultdict(lambda: dict(n=0, ok=0, all_pass=0, rub=0, rub_pass=0))
for line in open(out / 'advancedif_judged.jsonl'):
    r = json.loads(line)
    t = per[r['benchmark_name']]
    t['n'] += 1
    j = r.get('judge_result') or {}   # official writer: original row + nested judge_result
    if not j.get('success'):
        continue
    t['ok'] += 1
    t['all_pass'] += str(j.get('satisfied_all_requirements', 'NO')).lower() == 'yes'
    for v in (j.get('rubrics_check') or {}).values():
        t['rub'] += 1
        t['rub_pass'] += 'yes' in str(v).lower()
summary = {k: dict(samples=v['n'], judged=v['ok'], all_rubrics_pass=100 * v['all_pass'] / max(v['n'], 1),
                   micro_rubric_pass=100 * v['rub_pass'] / max(v['rub'], 1)) for k, v in per.items()}
tot = {k: sum(v[k] for v in per.values()) for k in ('n', 'ok', 'all_pass', 'rub', 'rub_pass')}
summary['overall'] = dict(samples=tot['n'], judged=tot['ok'], all_rubrics_pass=100 * tot['all_pass'] / max(tot['n'], 1),
                          micro_rubric_pass=100 * tot['rub_pass'] / max(tot['rub'], 1))
(out / 'summary.json').write_text(json.dumps({'advancedif': summary}, indent=1) + '\n')
print(json.dumps({'advancedif': summary}, indent=1))
