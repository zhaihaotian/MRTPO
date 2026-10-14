"""CoRT: Counterfactual Replay for Token-level rubric-guided policy optimization (arXiv:2607.25659).

CoRT keeps the GRPO response-level advantage A_i and redistributes it over tokens with weights derived from a
counterfactual replay: the same sampled response is re-scored under the criteria-free prompt x^- and the
log-probability contrast Delta_{i,t} = l^+_{i,t} - l^-_{i,t} is mapped to bounded, response-normalised weights

    s_{i,t} = sigmoid(tau * (Delta_{i,t} - b)) - 1/2
    w~_{i,t} = 1 + eta * lambda_k * s_{i,t}
    w_{i,t} = w~_{i,t} / mean_t(w~_{i,t})
    A^_{i,t} = sg(w_{i,t}) * A_i

with the cubic SmoothStep schedule lambda_k over trainer steps (defaults: tau = 1, b = 0, eta = 0.5, 100-step ramp).
For HIR the criteria-free prompt is the original question without the instruction list.
"""

import json

import torch

from mrtpo.advantage import smoothstep
from mrtpo.criteria import encode_replay_prompt
from mrtpo.hir import HIRQuestionBuilder

_SENTINEL = "\x00HIR_RAW_QUESTION\x00"


def _affixes(criteria, q_type):
    rendered = HIRQuestionBuilder()._concat_question(_SENTINEL, list(criteria), q_type)
    if rendered.count(_SENTINEL) != 1:
        raise ValueError("unexpected HIR question layout")
    head, tail = rendered.split(_SENTINEL)
    return head, tail


def raw_question(prompt_text, criteria, q_type):
    """Invert HIRQuestionBuilder._concat_question: recover the question without the instruction list."""
    head, tail = _affixes(criteria, q_type)
    if not prompt_text.startswith(head) or not prompt_text.endswith(tail):
        raise ValueError("prompt does not match the HIR layout for its criteria")
    raw = prompt_text[len(head): len(prompt_text) - len(tail)] if tail else prompt_text[len(head):]
    if HIRQuestionBuilder()._concat_question(raw, list(criteria), q_type) != prompt_text:
        raise ValueError("HIR question inversion did not round-trip")
    return raw


def counterfactual_prompt(extra):
    """Criteria-free prompt x^- for one HIR row, recovered from its leave-one-out prompts."""
    criteria = json.loads(extra["criteria_json"])
    minus = json.loads(extra["minus_prompts_json"])
    if len(criteria) != len(minus) or len(criteria) != int(extra["n_criteria"]):
        raise ValueError("criteria / minus prompt counts disagree")
    return raw_question(minus[0], criteria[1:], extra["q_type"])


def build_cort_replay(data, tokenizer, max_prompt_length=2048, truncation="error"):
    """Same response tokens appended to the criteria-free prompt, encoded like RLHFDataset."""
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    encoded = {}
    prompt_ids, prompt_masks = [], []
    for extra, uid in zip(data.non_tensor_batch["extra_info"], data.non_tensor_batch["uid"]):
        if uid not in encoded:
            encoded[uid] = encode_replay_prompt(counterfactual_prompt(extra), tokenizer, max_prompt_length, truncation)
        ids, mask = encoded[uid]
        prompt_ids.append(ids[0])
        prompt_masks.append(mask[0])
    input_ids = torch.cat((torch.stack(prompt_ids).to(responses.device), responses), dim=-1)
    attention_mask = torch.cat((torch.stack(prompt_masks).to(response_mask.device), response_mask), dim=-1)
    return DataProto.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "responses": responses,
        "response_mask": response_mask,
    })


@torch.no_grad()
def cort_token_weights(factual_log_probs, counterfactual_log_probs, response_mask, trainer_step, options=None):
    """Response-normalised CoRT weights (mean 1 over each response) and logging metrics."""
    options = options or {}
    eta = float(options.get("eta", 0.5))
    tau = float(options.get("tau", 1.0))
    center = float(options.get("center", 0.0))
    schedule = smoothstep(int(trainer_step), int(options.get("ramp_steps", 100)))
    mask = response_mask.float()
    delta = (factual_log_probs.float() - counterfactual_log_probs.float()) * mask
    score = torch.sigmoid(tau * (delta - center)) - 0.5
    provisional = 1.0 + eta * schedule * score
    lengths = mask.sum(-1, keepdim=True).clamp_min(1.0)
    mean = (provisional * mask).sum(-1, keepdim=True) / lengths
    weights = torch.where(mask.bool(), provisional / mean, torch.ones_like(provisional))
    valid = mask.bool()
    metrics = {
        "cort/schedule": schedule,
        "cort/delta_mean": delta[valid].mean().item() if valid.any() else 0.0,
        "cort/delta_abs_mean": delta[valid].abs().mean().item() if valid.any() else 0.0,
        "cort/weight_min": weights[valid].min().item() if valid.any() else 1.0,
        "cort/weight_max": weights[valid].max().item() if valid.any() else 1.0,
        "cort/weight_std": weights[valid].std().item() if valid.sum() > 1 else 0.0,
    }
    return weights, metrics
