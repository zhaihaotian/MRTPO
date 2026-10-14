"""MRTPO token weights, the beta = 0 limit, the reallocation bound and the replay of varying groups only."""
import math
import unittest

import torch

from _load import mrtpo


def random_batch(rows=24, length=12, seed=0, tool_fraction=0.75):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, length + 1, (rows,), generator=g)
    mask = (torch.arange(length).unsqueeze(0) < lengths.unsqueeze(-1)).long()
    js = torch.rand(rows, length, generator=g) ** 6 * math.log(2) * mask  # concentrated, within [0, ln 2]
    tool = torch.rand(rows, generator=g) < tool_fraction
    return mask, js, tool, g


def group_advantage(scores, index, mask):
    """GDPO group z-score (sample std, eps 1e-6) broadcast over the response tokens."""
    out = torch.zeros(len(index))
    for uid in set(index):
        rows = [i for i, u in enumerate(index) if u == uid]
        s = scores[rows]
        out[rows] = (s - s.mean()) / (s.std() + 1e-6)
    return out.unsqueeze(-1) * mask


class TokenWeights(unittest.TestCase):
    def test_mean_one_per_response_and_positive(self):
        for seed in range(5):
            mask, js, tool, _ = random_batch(seed=seed)
            for b in (0.05, 0.25, 0.5, 0.9):
                w = mrtpo.token_weights(js, mask, tool, b)
                lengths = mask.sum(-1)
                means = (w * mask).sum(-1) / lengths
                self.assertTrue(torch.allclose(means, torch.ones_like(means), atol=1e-6))
                self.assertTrue(torch.all(w[mask.bool()] > 0))
                self.assertTrue(torch.all(w[~mask.bool()] == 0))
                self.assertTrue(torch.all(w[~tool][mask[~tool].bool()] == 1.0))  # not applicable -> uniform

    def test_mixture_formula(self):
        mask = torch.ones(1, 4, dtype=torch.long)
        js = torch.tensor([[0.0, math.log(2) / 4, math.log(2), 0.0]])
        q = torch.sqrt(js / math.log(2))
        b = 0.5
        expected = (1 - b) + b * q / q.mean()
        w = mrtpo.token_weights(js, mask, torch.tensor([True]), b)
        self.assertTrue(torch.allclose(w, expected, atol=1e-6))

    def test_zero_js_row_keeps_uniform_weights(self):
        mask = torch.ones(2, 5, dtype=torch.long)
        js = torch.zeros(2, 5)
        js[1, 2] = 0.3
        w = mrtpo.token_weights(js, mask, torch.tensor([True, True]), 0.5)
        self.assertTrue(torch.equal(w[0], torch.ones(5)))
        self.assertGreater(float(w[1, 2]), 1.0)

    def test_smoothstep(self):
        self.assertEqual(mrtpo.smoothstep(0, 10), 0.0)
        self.assertAlmostEqual(mrtpo.smoothstep(5, 10), 0.5)
        self.assertEqual(mrtpo.smoothstep(10, 10), 1.0)
        self.assertEqual(mrtpo.smoothstep(60, 10), 1.0)


class BetaZeroIsGDPO(unittest.TestCase):
    def test_beta_zero_equals_gdpo_exactly(self):
        for seed in range(5):
            mask, js_c, tool, g = random_batch(seed=seed)
            js_f = torch.rand(js_c.shape, generator=g) * math.log(2) * mask
            index = [i // 4 for i in range(mask.shape[0])]
            a_c = group_advantage(torch.randn(mask.shape[0], generator=g), index, mask)
            a_f = group_advantage((torch.rand(mask.shape[0], generator=g) > 0.5).float(), index, mask)
            for bound in (False, True):
                out, metrics = mrtpo.combine(a_c, a_f, js_c, js_f, mask, tool, b=0.0, reallocation_bound_on=bound)
                self.assertTrue(torch.equal(out, a_c + a_f))
                self.assertEqual(metrics['mrtpo/reallocation_scale'], 1.0)


class ReallocationBound(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.mask = torch.ones(8, 12)
        self.mask[0, 9:] = 0
        self.mask[3, 5:] = 0
        per_response = torch.randn(8, generator=g)
        self.base = per_response[:, None] * self.mask
        w = torch.rand(8, 12, generator=g) ** 4 * 30 + 0.5
        w = w * self.mask
        w = w / (w.sum(-1, keepdim=True) / self.mask.sum(-1, keepdim=True)) * self.mask
        self.token = self.base * w
        self.per_response = per_response

    def second_moment(self, x):
        return float((x * self.mask).double().square().sum() / self.mask.sum())

    def test_above_budget_scaled_to_budget(self):
        out, scale = mrtpo.reallocation_bound(self.token, self.base, self.mask, 1.0)
        self.assertLess(scale, 1.0)
        self.assertAlmostEqual(self.second_moment(out - self.base), self.second_moment(self.base), places=5)
        means_in = (self.token * self.mask).sum(-1) / self.mask.sum(-1)
        means_out = (out * self.mask).sum(-1) / self.mask.sum(-1)
        self.assertTrue(torch.allclose(means_in, means_out, atol=1e-5))  # response means kept
        for i in range(8):
            valid = self.mask[i].bool()
            if self.per_response[i] != 0:
                self.assertTrue(torch.equal(torch.argsort(self.token[i, valid]), torch.argsort(out[i, valid])))
        self.assertEqual(float((out * (1 - self.mask)).abs().sum()), 0.0)

    def test_ratio_scales_the_factor(self):
        _, s1 = mrtpo.reallocation_bound(self.token, self.base, self.mask, 1.0)
        _, s4 = mrtpo.reallocation_bound(self.token, self.base, self.mask, 0.25)
        self.assertAlmostEqual(s4, 0.5 * s1, places=9)
        out0, _ = mrtpo.reallocation_bound(self.token, self.base, self.mask, 0.0)
        self.assertTrue(torch.allclose(out0, self.base))

    def test_within_budget_unchanged(self):
        small = self.base + 0.01 * (self.token - self.base)
        out, scale = mrtpo.reallocation_bound(small, self.base, self.mask, 1.0)
        self.assertEqual(scale, 1.0)
        self.assertTrue(torch.equal(out, small))


class ReplayOnlyVaryingGroups(unittest.TestCase):
    def test_replay_rows(self):
        mask = torch.ones(8, 3, dtype=torch.long)
        index = ['a'] * 4 + ['b'] * 4
        constant = group_advantage(torch.tensor([1., 1., 1., 1., 0., 1., 0., 0.]), index, mask)
        self.assertTrue(torch.equal(constant[:4], torch.zeros(4, 3)))  # constant group: advantage exactly 0
        rows = mrtpo.replay_rows(constant, mask, index)
        self.assertEqual(rows.tolist(), [False] * 4 + [True] * 4)

    def test_skipping_constant_groups_changes_nothing(self):
        for seed in range(5):
            mask, js_c, tool, g = random_batch(rows=32, seed=seed, tool_fraction=1.0)
            js_f = torch.rand(js_c.shape, generator=g) * math.log(2) * mask
            index = [i // 4 for i in range(32)]
            sc = torch.randn(32, generator=g)
            sf = (torch.rand(32, generator=g) > 0.5).float()
            sc[:8] = 1.5   # groups 0, 1 constant on correctness
            sf[8:16] = 1.0  # groups 2, 3 constant on format
            a_c, a_f = group_advantage(sc, index, mask), group_advantage(sf, index, mask)
            rc = mrtpo.replay_rows(a_c, mask, index) & tool
            rf = mrtpo.replay_rows(a_f, mask, index)
            self.assertFalse(bool(rc[:8].any()) or bool(rf[8:16].any()))
            for bound in (False, True):
                full, _ = mrtpo.combine(a_c, a_f, js_c, js_f, mask, tool, b=0.5, reallocation_bound_on=bound)
                skipped, _ = mrtpo.combine(a_c, a_f, js_c * rc.unsqueeze(-1), js_f * rf.unsqueeze(-1), mask, tool,
                                           b=0.5, reallocation_bound_on=bound)
                self.assertTrue(torch.equal(full, skipped))


class ReplayJS(unittest.TestCase):
    """replay_js on a toy per-row model: right pairs, one forward call per context and micro-batch, counts."""

    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.rows, self.length, self.vocab = 20, 6, 11
        lengths = torch.randint(1, self.length + 1, (self.rows,), generator=g)
        self.mask = (torch.arange(self.length).unsqueeze(0) < lengths.unsqueeze(-1)).long()
        self.batch = {'responses': torch.randint(0, self.vocab, (self.rows, self.length), generator=g),
                      'tool_call': torch.rand(self.rows, generator=g) < 0.7,
                      'correctness_replay': torch.rand(self.rows, generator=g) < 0.5,
                      'format_replay': torch.rand(self.rows, generator=g) < 0.5}
        self.offset = {}
        for k, context in enumerate(mrtpo.CONTEXTS):
            prompt = torch.full((self.rows, 3), k + 1)
            self.batch[f'{context}_input_ids'] = torch.cat([prompt, self.batch['responses']], -1)
            self.batch[f'{context}_attention_mask'] = torch.cat([torch.ones(self.rows, 3, dtype=torch.long), self.mask], -1)
            self.batch[f'{context}_position_ids'] = torch.zeros(self.rows, 3 + self.length, dtype=torch.long)
        self.calls = []

    def logits(self, row, context):
        g = torch.Generator().manual_seed(1000 * mrtpo.CONTEXTS.index(context) + row)
        return torch.randn(int(self.mask[row].sum()), self.vocab, generator=g)

    def forward(self, rows, context):
        ids = rows['responses']
        self.calls.append((context, ids.shape[0]))
        # identify rows by their responses
        found = [next(i for i in range(self.rows) if torch.equal(self.batch['responses'][i], r)) for r in ids]
        return torch.cat([self.logits(i, context) for i in found]), rows[f'{context}_attention_mask'][:, -self.length:]

    def test_pairs_calls_and_counts(self):
        c_js, f_js, counts = mrtpo.replay_js(self.batch, self.forward, micro_batch_size=8, chunk_size=4)
        tool, c, f = self.batch['tool_call'], self.batch['correctness_replay'], self.batch['format_replay']
        for i in range(self.rows):
            valid = self.mask[i].bool()
            if c[i] and tool[i]:
                ref = mrtpo.js_divergence_from_logits(self.logits(i, 'full'), self.logits(i, 'format'))
                self.assertTrue(torch.allclose(c_js[i][valid], ref, atol=1e-6))
            else:
                self.assertTrue(torch.all(c_js[i] == 0))
            if f[i]:
                pair = ('full', 'correctness') if tool[i] else ('format', 'base')
                ref = mrtpo.js_divergence_from_logits(self.logits(i, pair[0]), self.logits(i, pair[1]))
                self.assertTrue(torch.allclose(f_js[i][valid], ref, atol=1e-6))
            else:
                self.assertTrue(torch.all(f_js[i] == 0))
        micro_batches = math.ceil(self.rows / 8)
        self.assertEqual(len(self.calls), 4 * micro_batches)
        self.assertEqual([context for context, _ in self.calls], list(mrtpo.CONTEXTS) * micro_batches)
        expected = mrtpo.replay_sequence_counts(tool, c, f, world_size=1, micro_batch_size=8)
        self.assertEqual(counts, expected)


if __name__ == '__main__':
    unittest.main()
