"""Full-vocabulary Jensen-Shannon divergence with bounded fp32 workspace."""

import math

import torch


@torch.no_grad()
def js_divergence_from_logits(left_logits, right_logits, temperature=1.0):
    """Compute one token chunk in fp32, with no vocabulary approximation."""
    if left_logits.shape != right_logits.shape:
        raise ValueError("Paired logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left = torch.log_softmax(left_logits.float() / temperature, dim=-1)
    right = torch.log_softmax(right_logits.float() / temperature, dim=-1)
    log_mixture = torch.logaddexp(left, right) - math.log(2.0)
    divergence = 0.5 * (
        (left.exp() * (left - log_mixture)).sum(dim=-1)
        + (right.exp() * (right - log_mixture)).sum(dim=-1)
    )
    return divergence.clamp(min=0.0, max=math.log(2.0))


@torch.no_grad()
def chunked_distribution_js(left_logits, right_logits, chunk_size=128, temperature=1.0):
    """Keep long-sequence logits in their native dtype; cast only a token chunk."""
    if left_logits.shape != right_logits.shape or left_logits.ndim != 2:
        raise ValueError("Paired logits must have shape [tokens, vocabulary]")
    if not 1 <= chunk_size <= 512:
        raise ValueError("JS chunk_size must lie in [1, 512]")
    result = torch.empty(left_logits.size(0), dtype=torch.float32, device=left_logits.device)
    for start in range(0, left_logits.size(0), chunk_size):
        stop = start + chunk_size
        result[start:stop] = js_divergence_from_logits(
            left_logits[start:stop], right_logits[start:stop], temperature=temperature
        )
    return result
