import math

import pytest
import torch

from mrtpo.advantage import (
    allocate_criterion_advantage,
    bound_reallocation,
    compute_criterion_advantage,
    prepare_criteria,
    saw_criterion_weights,
    smoothstep,
)
from mrtpo.criteria import criterion_group_advantages, parse_criteria_scores

from helpers import criterion_batch


def config(**mrtpo):
    return {"batch_norm": "response", "saw": False, "mrtpo": {"beta": 0.5, "ramp_steps": 10, **mrtpo}}


def test_group_advantages_hand_computed_and_ragged():
    scores, mask = parse_criteria_scores(['[1, 0]', '[0, 0]', '[1, 1, 1]', '[0, 1, 1]'])
    adv, active, fraction = criterion_group_advantages(scores, mask, ["a", "a", "b", "b"])
    std = torch.tensor([1.0, 0.0]).std(unbiased=True)
    torch.testing.assert_close(adv[0, 0], 0.5 / (std + 1e-6))
    assert adv[0, 1] == 0 and not active[0, 1]           # criterion that does not vary in its group
    assert not mask[0, 2] and adv[0, 2] == 0             # padding of the shorter criterion list
    assert active.tolist() == [[True, False, False], [True, False, False], [True, False, False], [True, False, False]]
    assert fraction == 2 / 5


def test_gdpo_is_sum_of_criterion_advantages_over_batch_std():
    data = criterion_batch()
    adv, metrics = compute_criterion_advantage(data, "gdpo", config())
    per_response = data.batch["criterion_advantages"].sum(-1)
    expected = per_response[:, None] * data.batch["response_mask"] / (per_response.std(unbiased=True) + 1e-8)
    assert torch.equal(adv, expected * data.batch["response_mask"])
    assert metrics["mrtpo/weight_max"] == 1.0


@pytest.mark.parametrize("bound", [False, True])
def test_beta_zero_equals_gdpo_exactly(bound):
    gdpo, _ = compute_criterion_advantage(criterion_batch(), "gdpo", config())
    mrtpo, _ = compute_criterion_advantage(criterion_batch(), "mrtpo", config(beta=0.0, reallocation_bound=bound))
    assert torch.equal(gdpo, mrtpo)


def test_schedule_zero_equals_gdpo_and_ramp_values():
    gdpo, _ = compute_criterion_advantage(criterion_batch(step=0), "gdpo", config())
    mrtpo, metrics = compute_criterion_advantage(criterion_batch(step=0), "mrtpo", config())
    assert torch.equal(gdpo, mrtpo) and metrics["mrtpo/schedule"] == 0.0
    assert smoothstep(5, 10) == 0.5 and smoothstep(10, 10) == 1.0 and smoothstep(30, 10) == 1.0
    assert smoothstep(1, 10) == pytest.approx(3 * 0.01 - 2 * 0.001)


def test_weights_mean_one_per_response_and_formula():
    data = criterion_batch(seed=3)
    prepare_criteria(data)
    mask = data.batch["response_mask"]
    for j in range(3):
        active = data.batch["criterion_active"][:, j]
        signed = data.batch["criterion_advantages"][:, j, None].expand_as(mask)
        allocated, w = allocate_criterion_advantage(signed, data.batch["criterion_js"][:, j], mask, active,
                                                    schedule=1.0, beta=0.5)
        means = (w * mask).sum(-1) / mask.sum(-1)
        torch.testing.assert_close(means, torch.ones_like(means))
        assert (w[mask.bool()] >= 0.5 - 1e-6).all()
        torch.testing.assert_close(allocated, signed * w)
        q = torch.sqrt(data.batch["criterion_js"][:, j] / math.log(2)) * mask
        for row in range(len(mask)):
            q_mean = q[row].sum() / mask[row].sum()
            if active[row] and q_mean > 0:
                expected = (0.5 + 0.5 * q[row] / q_mean) * mask[row]
                torch.testing.assert_close(w[row], expected)
            else:
                assert torch.equal(w[row], mask[row])


def test_token_reallocation_keeps_each_response_advantage():
    data = criterion_batch(seed=5)
    adv, metrics = compute_criterion_advantage(data, "mrtpo", config(reallocation_bound=False))
    gdpo, _ = compute_criterion_advantage(criterion_batch(seed=5), "gdpo", config())
    mask = data.batch["response_mask"]
    torch.testing.assert_close((adv * mask).sum(-1) / mask.sum(-1), (gdpo * mask).sum(-1) / mask.sum(-1))
    assert not torch.equal(adv, gdpo) and metrics["mrtpo/weight_max"] > 1


@pytest.mark.parametrize("ratio", [1.0, 0.25, 0.0])
def test_reallocation_bound_caps_residual_second_moment(ratio):
    data = criterion_batch(seed=7)
    unbounded, _ = compute_criterion_advantage(data, "mrtpo", config(reallocation_bound=False))
    data = criterion_batch(seed=7)
    bounded, metrics = compute_criterion_advantage(
        data, "mrtpo", config(reallocation_bound=True, reallocation_ratio=ratio))
    mask = data.batch["response_mask"]
    base = ((bounded * mask).sum(-1) / mask.sum(-1))[:, None] * mask
    residual = (bounded - base) * mask
    tokens = mask.sum()
    assert residual.double().square().sum() / tokens <= ratio * base.double().square().sum() / tokens + 1e-9
    torch.testing.assert_close((bounded * mask).sum(-1), (unbounded * mask).sum(-1), rtol=1e-5, atol=1e-5)
    assert metrics["mrtpo/residual_scale"] <= 1.0
    if ratio == 0.0:
        gdpo, _ = compute_criterion_advantage(criterion_batch(seed=7), "gdpo", config())
        torch.testing.assert_close(bounded, gdpo)


def test_bound_leaves_small_residual_unchanged():
    mask = torch.ones(2, 4)
    per_response = torch.tensor([1.0, -1.0])
    tokens = per_response[:, None] * torch.tensor([[1.1, 0.9, 1.0, 1.0], [1.0, 1.0, 1.2, 0.8]])
    out, metrics = bound_reallocation(tokens, per_response, mask)
    assert torch.equal(out, tokens) and metrics["mrtpo/residual_scale"] == 1.0


def test_bound_is_the_default():
    data = criterion_batch(seed=7)
    _, metrics = compute_criterion_advantage(data, "mrtpo", {"mrtpo": {}})
    assert "mrtpo/residual_scale" in metrics


def test_saw_weights_formula_and_equal_weights_without_variation():
    scores = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    weights, metrics = saw_criterion_weights(scores, torch.ones_like(scores, dtype=torch.bool))
    cv = scores.double().std(0, unbiased=True) / (scores.double().mean(0) + 1e-8)
    torch.testing.assert_close(weights.double(), 3 * cv / (cv.sum() + 1e-8))
    assert weights[0] == 0 and metrics["saw/weight_1"] == pytest.approx(weights[1].item())
    same, _ = saw_criterion_weights(torch.ones(4, 3), torch.ones(4, 3, dtype=torch.bool))
    assert torch.equal(same, torch.ones(3))


@pytest.mark.parametrize("method", ["gdpo", "mrtpo"])
def test_saw_scales_criterion_advantages(method):
    plain = criterion_batch(seed=9)
    prepare_criteria(plain)
    scores, mask = parse_criteria_scores(plain.non_tensor_batch["criteria_scores"])
    weights, _ = saw_criterion_weights(scores, mask)
    expected_batch = criterion_batch(seed=9)
    prepare_criteria(expected_batch)
    expected_batch.batch["criterion_advantages"] = expected_batch.batch["criterion_advantages"] * weights
    expected, _ = compute_criterion_advantage(expected_batch, method, config(reallocation_bound=False))
    actual, metrics = compute_criterion_advantage(criterion_batch(seed=9), method,
                                                  {**config(reallocation_bound=False), "saw": True})
    assert torch.equal(actual, expected) and "saw/cv_0" in metrics


def test_rejects_uncentred_batch_and_bad_options():
    data = criterion_batch()
    prepare_criteria(data)
    data.batch["criterion_advantages"] = data.batch["criterion_advantages"] + 1
    with pytest.raises(ValueError, match="group-centred"):
        compute_criterion_advantage(data, "gdpo", config())
    for options in ({"beta": 1.0}, {"reallocation_ratio": 1.5}):
        with pytest.raises(ValueError):
            compute_criterion_advantage(criterion_batch(), "mrtpo", config(**options))
    with pytest.raises(ValueError):
        compute_criterion_advantage(criterion_batch(), "gdpo", {"batch_norm": "token"})
