"""BFCL V4 single-turn categories and the paper's aggregation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import protocol  # noqa: E402


class Protocol(unittest.TestCase):
    def test_counts(self):
        counts = protocol.expected_counts()
        self.assertEqual(sum(counts.values()), 2501)
        self.assertEqual(sum(counts[k] for k in protocol.LIVE), 1351)
        self.assertEqual(sum(counts[k] for k in protocol.SIMPLE + protocol.CALLS), 1150)

    def test_aggregation(self):
        counts = protocol.expected_counts()
        values = dict(simple_python=0.9, simple_java=0.6, simple_javascript=0.3, multiple=0.8, parallel=0.7,
                      parallel_multiple=0.5, live_simple=0.5, live_multiple=0.4, live_parallel=1.0,
                      live_parallel_multiple=0.0)
        out = protocol.aggregate(values, counts)
        self.assertAlmostEqual(out['simple'], 0.6)
        self.assertAlmostEqual(out['non_live'], (0.6 + 0.8 + 0.7 + 0.5) / 4)
        self.assertAlmostEqual(out['live'], (0.5 * 258 + 0.4 * 1053 + 16) / 1351)
        self.assertAlmostEqual(out['single_turn'], (1150 * out['non_live'] + 1351 * out['live']) / 2501)
        self.assertEqual(list(out), protocol.SUMMARY)


if __name__ == '__main__':
    unittest.main()
