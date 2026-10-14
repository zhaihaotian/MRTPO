"""Average evaluation results over seeds: execution accuracy per set and Avg (mean of the six sets).

    python evaluation/summarize.py eval/coder7b-mrtpo-s42 eval/coder7b-mrtpo-s1 eval/coder7b-mrtpo-s2
"""
import argparse
import json
from pathlib import Path

SETS = ('spider_dev', 'spider_test', 'spider_syn', 'spider_realistic', 'spider_dk', 'bird_dev')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path, help='evaluation output directories, one per seed')
    args = parser.parse_args()
    table = {}
    for name in SETS:
        values = [json.loads((run / name / 'result.json').read_text())['accuracy'] for run in args.runs]
        table[name] = sum(values) / len(values)
    table['avg'] = sum(table[name] for name in SETS) / len(SETS)
    print(json.dumps({'runs': [str(run) for run in args.runs],
                      'accuracy': {k: round(v, 2) for k, v in table.items()}}, indent=2))


if __name__ == '__main__':
    main()
