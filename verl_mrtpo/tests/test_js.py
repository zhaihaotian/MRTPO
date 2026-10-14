import math

import pytest
import torch

from mrtpo.js import chunked_distribution_js


@pytest.mark.parametrize("chunk", [1, 7, 128, 512])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_js_matches_float64_reference(chunk, temperature):
    generator = torch.Generator().manual_seed(17)
    left = torch.randn(29, 101, generator=generator)
    right = torch.randn(29, 101, generator=generator)
    p = torch.softmax(left.double() / temperature, -1)
    q = torch.softmax(right.double() / temperature, -1)
    m = (p + q) / 2
    reference = 0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1))
    js = chunked_distribution_js(left, right, chunk, temperature)
    torch.testing.assert_close(js.double(), reference, rtol=1e-5, atol=1e-5)
    assert (js >= 0).all() and (js <= math.log(2)).all()


def test_identical_and_disjoint_distributions():
    logits = torch.randn(40, 77)
    torch.testing.assert_close(chunked_distribution_js(logits, logits, 3), torch.zeros(40), rtol=0, atol=1e-7)
    left = torch.tensor([[1000.0, -1000.0]])
    torch.testing.assert_close(chunked_distribution_js(left, -left), torch.tensor([math.log(2)]))
