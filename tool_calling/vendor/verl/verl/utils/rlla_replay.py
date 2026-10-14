"""Replay contexts for MRTPO on RLLA tool calling.

The rollout context P_0 is the original system/user pair. A replay context keeps the system message and appends to
the user message only the information one reward depends on:
  P_F  : the response form the format reward checks (tool-call block or direct answer),
  P_C  : the reference calls the correctness reward compares against (operation names, parameter names, values),
  P_FC : both blocks.
Leave-one-out replay compares P_FC with P_F (correctness) and P_FC with P_C (format) on the same sampled response.
Direct-answer rows have no correctness reward; their format replay compares P_F with P_0.
The correctness block is written from `extra_info.teacher_output` (the label with numbers and booleans in the types the
tool schema declares) when present; the reward itself always reads the original label.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_TOOL_CALL_FORMAT = """For the final user request, respond in this form:

<think>...</think>
<tool_call>
...
</tool_call>

Use one <tool_call> block, place its content between the opening and closing tags on separate lines, and end the answer with </tool_call>. Do not add a <response> block."""

_DIRECT_RESPONSE_FORMAT = """For the final user request, respond in this form:

<think>...</think>
<response>...</response>

Use one <response> block and end the answer with </response>. Do not add a <tool_call> block."""


def label_kind(ground_truth: str) -> str:
    """'tool_call' or 'response': the one response mode present in an RLLA label."""
    has_tool_call = "<tool_call>" in ground_truth
    has_response = "<response>" in ground_truth
    if has_tool_call == has_response:
        raise ValueError("RLLA labels contain exactly one of <tool_call> or <response>")
    return "tool_call" if has_tool_call else "response"


def parse_tool_calls(ground_truth: str) -> List[Dict[str, Any]]:
    """Parse the label's tool-call payload, one JSON object per line."""
    if label_kind(ground_truth) != "tool_call":
        return []
    payload = ground_truth.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    calls = [json.loads(line) for line in payload.splitlines() if line.strip()]
    for call in calls:
        if not isinstance(call, dict) or "name" not in call or "parameters" not in call:
            raise ValueError("Each RLLA tool call must contain name and parameters")
        if not isinstance(call["parameters"], dict):
            raise ValueError("RLLA tool-call parameters must be a JSON object")
    if not calls:
        raise ValueError("Tool-call labels must contain at least one JSON call")
    return calls


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def format_block(ground_truth: str) -> str:
    return _TOOL_CALL_FORMAT if label_kind(ground_truth) == "tool_call" else _DIRECT_RESPONSE_FORMAT


def correctness_block(reference: str) -> Optional[str]:
    """The reference calls as text; None for direct-answer rows (no correctness reward)."""
    if label_kind(reference) == "response":
        return None
    calls = parse_tool_calls(reference)
    noun = "operation" if len(calls) == 1 else "operations"
    lines = [f"The correctness reference for the final user request contains these ordered {noun}:"]
    for index, call in enumerate(calls, start=1):
        lines.append(f"{index}. Operation name: `{call['name']}`.")
        parameters = call["parameters"]
        if not parameters:
            lines.append("   Parameters: none.")
        else:
            for key, value in parameters.items():
                lines.append(f"   Parameter `{key}`: {_json_literal(value)}.")
    lines.append("The operation order, names, parameter names, and parameter values above "
                 "are the complete correctness reference.")
    return "\n".join(lines)


def _messages(instruction: str, user_input: str, added: Optional[str] = None) -> List[Dict[str, str]]:
    content = user_input if added is None else f"{user_input}\n\n{added}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": content}]


def context_messages(row: Dict[str, Any], context: str) -> Tuple[List[Dict[str, str]], bool]:
    """Messages of one replay context for one dataset row, and whether the context exists for that row.

    Rows without a correctness reward get P_0 for 'full' and 'correctness' (marked not applicable).
    """
    instruction, user_input, label = row["instruction"], row["input"], row["output"]
    reference = row.get("teacher_output") or label
    if context == "format":
        return _messages(instruction, user_input, format_block(label)), True
    if context == "full":
        block = correctness_block(reference)
        if block is None:
            return _messages(instruction, user_input), False
        return _messages(instruction, user_input, f"{block}\n\n{format_block(label)}"), True
    if context == "correctness":
        block = correctness_block(reference)
        if block is None:
            return _messages(instruction, user_input), False
        return _messages(instruction, user_input, block), True
    raise ValueError(f"Unknown replay context: {context}")


def _token_ids(tokenizer: Any, messages: List[Dict[str, str]]) -> List[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.reshape(-1).tolist()
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError("tokenizer.apply_chat_template must return a list of token ids")
    return encoded


def splice_prompt(actual_base_ids: List[int], full_base_ids: List[int], full_context_ids: List[int]) -> List[int]:
    """Insert the replay block into the exact (possibly left-truncated) P_0 tokens the rollout used."""
    if not actual_base_ids or not full_base_ids or not full_context_ids:
        raise ValueError("Base and context prompt token sequences must be non-empty")
    if len(actual_base_ids) > len(full_base_ids):
        raise ValueError("Actual P_0 prompt cannot be longer than its full rendering")
    if actual_base_ids != full_base_ids[-len(actual_base_ids):]:
        raise ValueError("Actual P_0 prompt must equal a suffix of the full P_0 rendering")
    lcp = 0
    for base_token, context_token in zip(full_base_ids, full_context_ids):
        if base_token != context_token:
            break
        lcp += 1
    if lcp == len(full_base_ids):
        raise ValueError("Context rendering does not insert anything before the P_0 tail")
    base_tail = full_base_ids[lcp:]
    if len(base_tail) > len(actual_base_ids) or actual_base_ids[-len(base_tail):] != base_tail:
        raise ValueError("Context insertion boundary was removed by P_0 truncation")
    return actual_base_ids[:-len(base_tail)] + full_context_ids[lcp:]


def context_tensors(*, tokenizer: Any, responses: torch.Tensor, response_mask: torch.Tensor,
                    extra_info: Sequence[Dict[str, Any]], context: str, base_prompts: torch.Tensor,
                    base_prompt_mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """input_ids / attention_mask / position_ids of one replay context (left-padded prompt + the same response)."""
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError("responses and response_mask must be rank-two tensors with identical shape")
    if len(extra_info) != responses.shape[0] or base_prompts.shape[0] != responses.shape[0]:
        raise ValueError("extra_info, prompts and responses must share the batch size")
    prompt_ids, applicable = [], []
    for row_index, row in enumerate(extra_info):
        messages, is_applicable = context_messages(row, context)
        actual_ids = base_prompts[row_index][base_prompt_mask[row_index].bool()].tolist()
        if is_applicable:
            effective_ids = splice_prompt(actual_ids, _token_ids(tokenizer, _messages(row["instruction"], row["input"])),
                                          _token_ids(tokenizer, messages))
        else:
            effective_ids = actual_ids
        prompt_ids.append(effective_ids)
        applicable.append(is_applicable)

    device, dtype = responses.device, responses.dtype
    max_prompt_length = max(len(ids) for ids in prompt_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    prompts = torch.full((len(prompt_ids), max_prompt_length), fill_value=pad_token_id, dtype=dtype, device=device)
    prompt_mask = torch.zeros_like(prompts, dtype=response_mask.dtype)
    for row_index, ids in enumerate(prompt_ids):
        prompts[row_index, -len(ids):] = torch.tensor(ids, dtype=dtype, device=device)
        prompt_mask[row_index, -len(ids):] = 1
    attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    tensors = {
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": attention_mask,
        "position_ids": torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
    }
    return tensors, torch.tensor(applicable, dtype=torch.bool, device=device)
