"""Per-criterion GDPO advantages and MRTPO's token-level reallocation.

GDPO: every criterion k is normalized within its prompt's response group, A_k = (r_k - mean) / (std + 1e-6), the
criterion advantages are summed per response, and the sum is divided by its batch standard deviation (GDPO Eq. 6).

MRTPO: the same criterion advantage A_k is spread over the response's tokens with weights

    w_t = (1 - b) + b * q_t / mean(q),   q_t = sqrt(JS_t / ln 2),   b = beta * smoothstep(step / ramp_steps),

where JS_t compares the policy's next-token distributions on the same response with and without criterion k's
context (leave-one-out replay, computed by the actor worker). Each response's weights average 1, so the response's
total credit equals GDPO's. The reallocation bound (paper appendix "Bounding the reallocation") shrinks the
token-level residual (token advantage minus the response's GDPO advantage) by one common factor so that its second
moment over the batch's tokens is at most reallocation_ratio times the GDPO advantage's second moment.

SAW (optional, GDPO and MRTPO): criterion k's group advantage is multiplied by n * CV_k / sum_l CV_l before the sum.
"""

import math

import torch

from mrtpo.criteria import criterion_group_advantages, criterion_metrics, parse_criteria_scores

LOG_TWO = 0.6931471805599453


def prepare_criteria(data):
    """Store per-criterion group advantages and their activity mask on the batch; return reward metrics."""
    scores, mask = parse_criteria_scores(
        data.non_tensor_batch["criteria_scores"], device=data.batch["response_mask"].device
    )
    advantage, active, fraction = criterion_group_advantages(scores, mask, data.non_tensor_batch["uid"])
    data.batch["criterion_advantages"] = advantage
    data.batch["criterion_active"] = active
    data.batch["criterion_mask"] = mask
    return criterion_metrics(scores, mask, fraction)


def smoothstep(trainer_step: int, ramp_steps: int) -> float:
    """Cubic ramp 3p^2 - 2p^3 with p = clamp(step / ramp_steps, 0, 1)."""
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    progress = min(max(trainer_step / ramp_steps, 0.0), 1.0)
    return float(3.0 * progress**2 - 2.0 * progress**3)


def mrtpo_options(config):
    """Read and validate algorithm.mrtpo.* (defaults are the paper values)."""
    options = (config or {}).get("mrtpo", {})
    beta = float(options.get("beta", 0.5))
    ramp_steps = int(options.get("ramp_steps", 10))
    bound = bool(options.get("reallocation_bound", True))
    ratio = float(options.get("reallocation_ratio", 1.0))
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("reallocation_ratio must be finite and lie in [0, 1]")
    return beta, ramp_steps, bound, ratio


@torch.no_grad()
def compute_criterion_advantage(data, method, config=None):
    """Token advantages for method "gdpo" or "mrtpo"; MRTPO reads data.batch["criterion_js"]."""
    if method not in ("gdpo", "mrtpo"):
        raise ValueError(f"unknown criterion method: {method}")
    if str((config or {}).get("batch_norm", "response")) != "response":
        raise ValueError("only batch_norm=response is supported")
    if "criterion_advantages" not in data.batch:
        prepare_criteria(data)
    channel_adv = data.batch["criterion_advantages"]
    active = data.batch["criterion_active"]
    response_mask = data.batch["response_mask"]
    beta, ramp_steps, bound, ratio = mrtpo_options(config)
    if ratio != 1.0 and (method != "mrtpo" or not bound):
        raise ValueError("reallocation_ratio requires MRTPO with reallocation_bound")
    saw_metrics = {}
    if (config or {}).get("saw", False):
        scores, criterion_mask = parse_criteria_scores(
            data.non_tensor_batch["criteria_scores"], device=channel_adv.device
        )
        weights, saw_metrics = saw_criterion_weights(scores, criterion_mask)
        channel_adv = channel_adv * weights
    schedule = smoothstep(int(data.meta_info.get("global_steps", 1)), ramp_steps)
    metrics = {"mrtpo/schedule": schedule, "mrtpo/weight_min": 1.0, "mrtpo/weight_max": 1.0, "mrtpo/weight_std": 0.0}
    if method == "gdpo" or beta == 0.0 or schedule == 0.0 or not active.any():
        token_adv = channel_adv.sum(dim=-1, keepdim=True) * response_mask
    else:
        js = data.batch["criterion_js"]
        token_adv = torch.zeros_like(response_mask, dtype=channel_adv.dtype)
        weight_count = 0
        weight_sum = weight_square_sum = 0.0
        weight_min, weight_max = float("inf"), float("-inf")
        for j in range(channel_adv.size(1)):
            signed = channel_adv[:, j, None].expand_as(response_mask)
            allocated, weights = allocate_criterion_advantage(
                signed, js[:, j], response_mask, active[:, j], schedule=schedule, beta=beta,
            )
            token_adv += allocated
            valid_weights = weights[active[:, j, None] & response_mask.bool()]
            if valid_weights.numel():
                work = valid_weights.double()
                weight_count += work.numel()
                weight_sum += work.sum().item()
                weight_square_sum += work.square().sum().item()
                weight_min = min(weight_min, work.min().item())
                weight_max = max(weight_max, work.max().item())
        if weight_count:
            metrics.update({
                "mrtpo/weight_min": weight_min,
                "mrtpo/weight_max": weight_max,
                "mrtpo/weight_std": max(0.0, weight_square_sum / weight_count - (weight_sum / weight_count) ** 2) ** 0.5,
            })
    per_response = channel_adv.sum(dim=-1)
    if method == "mrtpo" and bound:
        token_adv, bound_metrics = bound_reallocation(token_adv, per_response, response_mask, ratio)
        scale = bound_metrics["mrtpo/residual_scale"]
        for name in ("min", "max", "std"):
            key = f"mrtpo/weight_{name}"
            original = metrics[key]
            bound_metrics[f"mrtpo/pre_residual_weight_{name}"] = original
            metrics[key] = scale * original if name == "std" else 1.0 + scale * (original - 1.0)
        metrics.update(bound_metrics)
    # Each criterion advantage is group-centred and the reallocation keeps every response's mean, so the batch
    # normalization only divides by the standard deviation of the per-response advantages.
    mean = per_response.mean()
    std = per_response.std(unbiased=True) if per_response.numel() > 1 else torch.zeros_like(mean)
    if mean.abs() > 1e-4 * (std + 1e-8) + 1e-6:
        raise ValueError(f"per-response advantages should be group-centred; batch mean {mean.item():.3e}")
    advantage = token_adv / (std + 1e-8) * response_mask
    lengths = response_mask.sum(dim=-1).clamp(min=1)
    token_mean = (token_adv * response_mask).sum(dim=-1) / lengths
    metrics.update({
        "advantage/response_std_before_norm": std.item(),
        "advantage/token_mean_drift": (token_mean - per_response).abs().max().item(),
    })
    metrics.update(saw_metrics)
    return advantage, metrics


def bound_reallocation(token_adv, per_response, response_mask, ratio=1.0):
    """Reallocation bound: scale the token residual so that its token second moment is <= ratio x the GDPO one."""
    lengths = response_mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("Every response must contain at least one valid token")
    base = per_response[:, None] * response_mask
    residual = (token_adv - base) * response_mask
    tokens = lengths.sum().item()
    base_second = base.double().square().sum().item() / tokens
    residual_second = residual.double().square().sum().item() / tokens
    budget_second = ratio * base_second
    scale = math.sqrt(budget_second / residual_second) if residual_second > budget_second else 1.0
    if scale < 1.0:
        token_adv = (base + scale * residual) * response_mask
    metrics = {
        "mrtpo/residual_scale": scale,
        "mrtpo/base_second_moment": base_second,
        "mrtpo/residual_second_moment_raw": residual_second,
        "mrtpo/residual_second_moment_effective": scale**2 * residual_second,
    }
    if ratio != 1.0:
        metrics["mrtpo/reallocation_ratio"] = ratio
    return token_adv, metrics


def saw_criterion_weights(scores, criterion_mask, epsilon=1e-8):
    """SAW (arXiv:2606.07705), advantage-level mode: weight criterion k by n * CV_k / sum_l CV_l, where CV_k is
    the batch std / mean of its raw score (scores lie in [0, 1], so the offset to the theoretical minimum is 0).
    Equal weights when every CV is zero, as in the released code."""
    if not criterion_mask.all():
        raise ValueError("saw requires the same criteria for every response")
    if scores.min() < 0:
        raise ValueError("saw expects criterion scores with minimum 0")
    work = scores.double()
    cv = work.std(dim=0, unbiased=True) / (work.mean(dim=0).abs() + epsilon)
    total = cv.sum()
    n = scores.size(1)
    weights = torch.ones_like(cv) if total == 0 else n * cv / (total + epsilon)
    metrics = {f"saw/weight_{k}": weights[k].item() for k in range(n)}
    metrics.update({f"saw/cv_{k}": cv[k].item() for k in range(n)})
    return weights.to(scores.dtype), metrics


@torch.no_grad()
def allocate_criterion_advantage(
    channel_advantage: torch.Tensor,
    js_divergence: torch.Tensor,
    response_mask: torch.Tensor,
    applicable_rows: torch.Tensor,
    *,
    schedule: float,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread one criterion's advantage over tokens: w = (1 - b) + b * q / mean(q), b = beta * schedule,
    q = sqrt(JS / ln 2).

    Each response's weights average exactly 1 (the criterion's credit is only moved between tokens) and stay
    >= 1 - b > 0. Rows that are not applicable (the criterion does not vary in the group), or whose JS is all zero,
    get weight 1.
    """
    shape = channel_advantage.shape
    for name, tensor in (("js_divergence", js_divergence), ("response_mask", response_mask)):
        if tensor.shape != shape:
            raise ValueError(f"{name} must match channel_advantage shape")
    if applicable_rows.shape != (shape[0],):
        raise ValueError("applicable_rows must contain one boolean per response")
    if not 0.0 <= schedule <= 1.0:
        raise ValueError("schedule must lie in [0, 1]")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    mask = response_mask.bool()
    lengths = mask.sum(dim=-1, keepdim=True)
    if torch.any(lengths == 0):
        raise ValueError("Every response must contain at least one valid token")
    applicable = applicable_rows.bool().unsqueeze(-1) & mask
    js = js_divergence.detach().float()
    if not torch.isfinite(js[applicable]).all():
        raise ValueError("Applicable JS divergence must be finite")
    if torch.any(js[applicable] < -1e-6) or torch.any(js[applicable] > LOG_TWO + 1e-4):
        raise ValueError("JS divergence must lie in [0, log(2)]")
    work_mask = mask.float()
    q = torch.sqrt(js.clamp(min=0.0, max=LOG_TWO) / LOG_TWO) * applicable.float()
    q_mean = q.sum(dim=-1, keepdim=True) / lengths.float()
    usable = applicable_rows.bool().unsqueeze(-1) & (q_mean > 0)
    ratio = torch.where(usable, q / q_mean.clamp(min=1e-12), torch.ones_like(q))
    b = beta * schedule
    weights = ((1.0 - b) + b * ratio) * work_mask
    row_mean = weights.sum(dim=-1, keepdim=True) / lengths.float()
    weights = (weights / row_mean) * work_mask
    weights = weights.to(channel_advantage.dtype)
    if torch.any(weights[mask] <= 0.0):
        raise ValueError("Every valid token weight must be strictly positive")
    return channel_advantage * weights, weights
