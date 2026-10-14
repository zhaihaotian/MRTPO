"""MRTPO for tool calling: leave-one-out replay, token weights and the reallocation bound.

Each reward k (format, correctness) keeps its GDPO group advantage A_k per response. MRTPO spreads A_k over the
response tokens with weights w_t = (1 - b) + b * q_t / mean(q), q_t = sqrt(JS_t / ln 2), b = beta * s(step), where
JS_t compares the policy's next-token distributions on the same response with and without the context that reward
k depends on (leave-one-out replay). The weights have mean 1 over every response, so each response keeps its GDPO
advantage on average. Optionally the reallocation bound limits how much the token advantages may move away from the
uniform GDPO advantages. This module only needs torch, so it can be tested on CPU without the training stack.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch

LOG_TWO = 0.6931471805599453


def smoothstep(step: int, ramp_steps: int) -> float:
    """Warm-up of the mixture strength: 3p^2 - 2p^3 with p = min(step / K, 1)."""
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    progress = min(max(step / ramp_steps, 0.0), 1.0)
    return float(3.0 * progress**2 - 2.0 * progress**3)


@torch.no_grad()
def token_weights(js: torch.Tensor, response_mask: torch.Tensor, applicable_rows: torch.Tensor,
                  b: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Per-token weights w = (1 - b) + b * q / mean(q), q = sqrt(JS / ln 2), renormalised to mean 1 per response.

    Rows that are not applicable, or whose JS is zero everywhere, keep weight 1 on every token. Padding gets 0.
    """
    if js.shape != response_mask.shape:
        raise ValueError("js must match response_mask")
    if applicable_rows.shape != (js.shape[0],):
        raise ValueError("applicable_rows must contain one boolean per response")
    if not 0.0 <= b < 1.0:
        raise ValueError("b = beta * schedule must lie in [0, 1)")
    mask = response_mask.bool()
    lengths = mask.sum(dim=-1, keepdim=True)
    if torch.any(lengths == 0):
        raise ValueError("Every response must contain at least one valid token")
    if b == 0.0:
        return mask.to(dtype)
    applicable = applicable_rows.bool().unsqueeze(-1) & mask
    work_mask = mask.float()
    work_applicable = applicable.float()
    work_lengths = lengths.float()
    js = js.detach().float()
    if not torch.isfinite(js[applicable]).all():
        raise ValueError("Applicable JS divergence must be finite")
    if torch.any(js[applicable] < -1e-6) or torch.any(js[applicable] > LOG_TWO + 1e-4):
        raise ValueError("JS divergence must lie in [0, log(2)]")
    js = js.clamp(min=0.0, max=LOG_TWO)
    js = js * work_applicable
    q = torch.sqrt(js / LOG_TWO) * work_applicable
    q_mean = q.sum(dim=-1, keepdim=True) / work_lengths
    usable = applicable_rows.bool().unsqueeze(-1) & (q_mean > 0)
    ratio = torch.where(usable, q / q_mean.clamp(min=1e-12), torch.ones_like(q))
    weights = ((1.0 - b) + b * ratio) * work_mask
    row_mean = weights.sum(dim=-1, keepdim=True) / work_lengths
    weights = (weights / row_mean) * work_mask
    weights = weights.to(dtype)
    if torch.any(weights[mask] <= 0.0):
        raise ValueError("Every valid token weight must be strictly positive")
    return weights


@torch.no_grad()
def reallocation_bound(token_advantage: torch.Tensor, base_advantage: torch.Tensor, response_mask: torch.Tensor,
                       ratio: float = 1.0) -> Tuple[torch.Tensor, float]:
    """Bound the reallocation R = token_advantage - base_advantage (base = uniform GDPO advantage on every token).

    Over all valid tokens of the batch, the second moment of R may be at most `ratio` times that of the base; above
    the bound R is shrunk by one common factor, which keeps every response's mean and its token ordering.
    Returns the (possibly scaled) token advantages and the factor (1.0 when within the bound).
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("reallocation_ratio must lie in [0, 1]")
    mask = response_mask.to(token_advantage.dtype)
    tokens = mask.sum()
    base = base_advantage * mask
    residual = (token_advantage - base_advantage) * mask
    base_second = (base.double().square().sum() / tokens).item()
    residual_second = (residual.double().square().sum() / tokens).item()
    budget = ratio * base_second
    scale = (budget / residual_second) ** 0.5 if residual_second > budget else 1.0
    if scale < 1.0:
        token_advantage = (base + scale * residual) * mask
    return token_advantage, scale


@torch.no_grad()
def combine(correctness_advantage: torch.Tensor, format_advantage: torch.Tensor,
            correctness_js: torch.Tensor, format_js: torch.Tensor, response_mask: torch.Tensor,
            tool_call_rows: torch.Tensor, b: float, reallocation_bound_on: bool = False,
            reallocation_ratio: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level MRTPO advantage before the batch whitening shared with GDPO.

    correctness_advantage / format_advantage: GDPO group z-scores broadcast over the response tokens.
    correctness_js: JS(P_FC, P_F) per token (tool-call rows); format_js: JS(P_FC, P_C) on tool-call rows and
    JS(P_F, P_0) on direct-answer rows. With b = 0 the result equals the GDPO sum A_correctness + A_format.
    """
    tool_call_rows = tool_call_rows.bool()
    every_row = torch.ones_like(tool_call_rows)
    weights_c = token_weights(correctness_js, response_mask, tool_call_rows, b, correctness_advantage.dtype)
    weights_f = token_weights(format_js, response_mask, every_row, b, format_advantage.dtype)
    gdpo = correctness_advantage + format_advantage
    advantage = correctness_advantage * weights_c + format_advantage * weights_f
    scale = 1.0
    if reallocation_bound_on:
        advantage, scale = reallocation_bound(advantage, gdpo, response_mask, reallocation_ratio)
    valid = response_mask.bool()
    metrics = {
        'mrtpo/b': float(b),
        'mrtpo/reallocation_scale': float(scale),
        'mrtpo/correctness_weight_std': float(weights_c[valid & tool_call_rows.unsqueeze(-1)].std(unbiased=False))
        if torch.any(tool_call_rows) else 0.0,
        'mrtpo/format_weight_std': float(weights_f[valid].std(unbiased=False)),
    }
    return advantage, metrics


def replay_rows(channel_advantage: torch.Tensor, response_mask: torch.Tensor, index: Sequence) -> torch.Tensor:
    """True for every response whose prompt group has a nonzero advantage on this reward anywhere.

    When all responses of a prompt score the same on a reward, its group advantage is exactly 0, so the token weights
    of that reward cannot change anything and its replay is skipped.
    """
    row_nonzero = ((channel_advantage != 0) & response_mask.bool()).any(dim=-1).tolist()
    group = {}
    for uid, nonzero in zip(index, row_nonzero):
        group[uid] = group.get(uid, False) or nonzero
    return torch.tensor([group[uid] for uid in index], dtype=torch.bool)


def js_divergence_from_logits(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Exact Jensen-Shannon divergence over the final vocabulary dimension."""
    if left_logits.shape != right_logits.shape:
        raise ValueError("Paired logits must have identical shapes")
    left = torch.log_softmax(left_logits.float(), dim=-1)
    right = torch.log_softmax(right_logits.float(), dim=-1)
    log_mixture = torch.logaddexp(left, right) - LOG_TWO
    divergence = 0.5 * ((left.exp() * (left - log_mixture)).sum(dim=-1)
                        + (right.exp() * (right - log_mixture)).sum(dim=-1))
    return divergence.clamp(min=0.0, max=LOG_TWO)


def chunked_js(left_logits: torch.Tensor, right_logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if left_logits.shape != right_logits.shape:
        raise ValueError("Paired response distributions must have identical shapes")
    pieces = [js_divergence_from_logits(left_logits[start:start + chunk_size], right_logits[start:start + chunk_size])
              for start in range(0, left_logits.size(0), chunk_size)]
    return torch.cat(pieces, dim=0)


# The four replay contexts of a row (input_ids / attention_mask / position_ids prefixes):
#   full        P_FC: prompt + format block + correctness block (tool-call rows)
#   format      P_F : prompt + format block
#   correctness P_C : prompt + correctness block (tool-call rows)
#   base        P_0 : the rollout prompt
CONTEXTS = ('full', 'format', 'correctness', 'base')

Forward = Callable[[Dict[str, torch.Tensor], str], Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def replay_js(batch: Dict[str, torch.Tensor], forward: Forward, micro_batch_size: int,
              chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Leave-one-out replay JS for the rows of one data-parallel rank.

    `batch` holds 'responses', 'tool_call', 'correctness_replay', 'format_replay' (one bool per row) and the
    <context>_input_ids / _attention_mask / _position_ids of the four contexts. `forward(rows, context)` returns the
    logits at every valid response position of `rows` (flattened row by row) and the response mask.
    Per micro-batch exactly one forward call is made per context (a one-row call when no row needs it), so every
    rank issues the same sequence of collective forward passes.
    Returns correctness JS, format JS (zeros where not replayed) and the number of sequences forwarded.
    """
    rows = batch['responses'].shape[0]
    correctness_out, format_out = [], []
    counts = {'sequences': 0, 'padding_sequences': 0}
    for start in range(0, rows, micro_batch_size):
        micro = {key: value[start:start + micro_batch_size] for key, value in batch.items()}
        response_mask = micro['full_attention_mask'][:, -micro['responses'].shape[-1]:].bool()
        tool = micro['tool_call'].bool()
        c = micro['correctness_replay'].bool() & tool
        ft = micro['format_replay'].bool() & tool
        fd = micro['format_replay'].bool() & ~tool
        needs = {'full': c | ft, 'format': c | fd, 'correctness': ft, 'base': fd}
        logits, masks = {}, {}
        for context in CONTEXTS:
            selected = needs[context].nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                forward({key: value[:1] for key, value in micro.items()}, context)  # keeps ranks in lockstep
                counts['padding_sequences'] += 1
                continue
            logits[context], masks[context] = forward({key: value[selected] for key, value in micro.items()}, context)
            if not torch.equal(masks[context].bool(), response_mask[selected]):
                raise ValueError(f"Response mask of context {context} differs from the rollout response mask")
            counts['sequences'] += int(selected.numel())
        correctness_js = torch.zeros(response_mask.shape, dtype=torch.float32, device=response_mask.device)
        format_js = torch.zeros_like(correctness_js)

        def tokens(context, wanted):
            """Logits of context `context` at the tokens of the rows in `wanted` (a subset of needs[context])."""
            have = needs[context].nonzero(as_tuple=True)[0]
            keep = torch.repeat_interleave(wanted[have], masks[context].sum(dim=-1))
            return logits[context][keep]

        if torch.any(c):
            correctness_js[c.unsqueeze(-1) & response_mask] = chunked_js(tokens('full', c), tokens('format', c),
                                                                         chunk_size)
        if torch.any(ft):
            format_js[ft.unsqueeze(-1) & response_mask] = chunked_js(tokens('full', ft), tokens('correctness', ft),
                                                                     chunk_size)
        if torch.any(fd):
            format_js[fd.unsqueeze(-1) & response_mask] = chunked_js(tokens('format', fd), tokens('base', fd),
                                                                     chunk_size)
        correctness_out.append(correctness_js)
        format_out.append(format_js)
    return torch.cat(correctness_out, dim=0), torch.cat(format_out, dim=0), counts


def replay_sequence_counts(tool_call: torch.Tensor, correctness_replay: torch.Tensor, format_replay: torch.Tensor,
                           world_size: int, micro_batch_size: int) -> Dict[str, int]:
    """Number of sequences `replay_js` forwards for the selected rows of one step, over all ranks.

    The rows are padded to a multiple of `world_size` by repeating the first rows (as the trainer does) and split
    evenly over the ranks; a context no row of a micro-batch needs costs one padding sequence.
    """
    rows = tool_call.numel()
    pad = (-rows) % world_size
    order = list(range(rows)) + list(range(pad))
    per_rank = len(order) // world_size
    counts = {'sequences': 0, 'padding_sequences': 0}
    for rank in range(world_size):
        chunk = order[rank * per_rank:(rank + 1) * per_rank]
        for start in range(0, len(chunk), micro_batch_size):
            rows_ = torch.tensor(chunk[start:start + micro_batch_size], dtype=torch.long)
            tool = tool_call[rows_].bool()
            c = correctness_replay[rows_].bool() & tool
            ft = format_replay[rows_].bool() & tool
            fd = format_replay[rows_].bool() & ~tool
            for need in (c | ft, c | fd, ft, fd):
                n = int(need.sum())
                counts['sequences'] += n
                counts['padding_sequences'] += int(n == 0)
    return counts
