"""Score saved BFCL V4 single-turn outputs with the official checkers (no type coercion); CPU only."""
import argparse
import json
import os
from pathlib import Path
import sys

from protocol import categories, expected_counts, aggregate, V4_COMMIT
from format_rate import category_format


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    os.environ['BFCL_PROJECT_ROOT'] = str(out)
    sys.path.insert(0, str(Path(__file__).parent / 'v4'))
    from adapter import register, RLLAHandler
    register()
    from bfcl_eval.eval_checker import eval_runner
    from bfcl_eval.utils import sort_key, get_directory_structure_by_category
    counts = expected_counts()
    by_category = {}
    for cat in categories():
        rows = [json.loads(line) for line in (out / 'raw' / f'{cat}.jsonl').read_text().splitlines()]
        if len(rows) != counts[cat] or len({r['id'] for r in rows}) != counts[cat]:
            raise ValueError(f'{cat}: expected {counts[cat]} unique cases; inference is incomplete')
        by_category[cat] = sorted(rows, key=sort_key)
    handler = RLLAHandler('rlla-eval', 0.6)
    for rows in by_category.values():
        handler.write(rows, out / 'result', update_mode=True)
    eval_runner.main(['rlla-eval'], list(by_category), out / 'result', out / 'score')
    scores, formats = {}, {}
    for cat, rows in by_category.items():
        formats[cat] = category_format(rows)
        path = out / 'score/rlla-eval' / get_directory_structure_by_category(cat) / f'BFCL_v4_{cat}_score.json'
        score = json.loads(path.read_text().splitlines()[0])
        if score['total_count'] != counts[cat]:
            raise ValueError(f'Official checker count mismatch: {cat}')
        scores[cat] = score
    accuracy = aggregate({k: v['correct_count'] / v['total_count'] for k, v in scores.items()}, counts)
    format_rate = aggregate(formats, counts)
    result = dict(version='v4', source_commit=V4_COMMIT, n_cases=sum(counts.values()),
                  categories={k: dict(correct_count=v['correct_count'], total_count=v['total_count'],
                                      accuracy=v['accuracy'], format=formats[k]) for k, v in scores.items()},
                  accuracy=accuracy, format=format_rate)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({metric: {k: round(100 * v, 2) for k, v in values.items()}
                      for metric, values in (('accuracy', accuracy), ('format', format_rate))}, indent=2))


if __name__ == '__main__':
    main()
