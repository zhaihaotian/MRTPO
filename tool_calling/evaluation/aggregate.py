"""Paper table rows: mean and sample SD over seeds of the BFCL V4 single-turn accuracies (%).

usage: python evaluation/aggregate.py [--metric format] --row 'MRTPO 1.5B' eval/1p5b-mrtpo-s{0,1,2,4,5} [--row NAME DIR ...]
"""
import argparse
import json
from pathlib import Path
import statistics

from protocol import COLUMNS

HEADER = ['NL Simple', 'NL Multiple', 'NL Parallel', 'NL Par. Mult.', 'NL Overall',
          'L Simple', 'L Multiple', 'L Parallel', 'L Par. Mult.', 'L Overall']


def row(name, dirs, metric='accuracy'):
    values = [json.loads((Path(d) / 'summary.json').read_text())[metric] for d in dirs]
    cells = []
    for column in COLUMNS:
        xs = [100 * v[column] for v in values]
        cells.append(f'{statistics.mean(xs):.2f}' + (f' ± {statistics.stdev(xs):.2f}' if len(xs) > 1 else ''))
    return f'| {name} | {len(dirs)} | ' + ' | '.join(cells) + ' |'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', nargs='+', action='append', required=True, metavar=('NAME', 'DIR'),
                   help='Row name followed by the evaluation directories of its seeds')
    p.add_argument('--metric', choices=['accuracy', 'format'], default='accuracy')
    args = p.parse_args()
    print('| Method | seeds | ' + ' | '.join(HEADER) + ' |')
    print('|' + '---|' * (len(HEADER) + 2))
    for name, *dirs in args.row:
        print(row(name, dirs, args.metric))


if __name__ == '__main__':
    main()
