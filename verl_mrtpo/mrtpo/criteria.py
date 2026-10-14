"""Criterion rewards and the fixed-response replays that MRTPO scores.

Two kinds of replay context:
- Prompt criteria (instruction following): the full context is the training prompt; the replay prompt for criterion
  k is the same prompt with criterion k left out (``extra_info["minus_prompts_json"][k]``).
- Privileged blocks (Text-to-SQL, ``extra_info["replay_mode"] == "privileged"``): the reward returns one text block
  per criterion (``replay_blocks_json``); the full context is ``p0_text`` followed by every non-empty block, the
  replay prompt for criterion k drops block k, and criteria with an empty block are not replayed.

The response tokens are never re-generated: the same sampled response is appended to both contexts.
"""

import json
from collections import defaultdict

import torch


def parse_criteria_scores(values, device=None):
    rows = [json.loads(value) for value in values]
    if not rows or any(not row for row in rows):
        raise ValueError("Each response needs at least one criterion score")
    scores = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.float32, device=device)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for i, row in enumerate(rows):
        if any(value not in (0, 1) for value in row):
            raise ValueError("Criterion scores must be binary")
        scores[i, : len(row)] = torch.tensor(row, device=device)
        mask[i, : len(row)] = True
    return scores, mask


@torch.no_grad()
def criterion_group_advantages(scores, criterion_mask, uids, epsilon=1e-6):
    """Normalize each real criterion within its own prompt's response group."""
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    advantages = torch.zeros_like(scores)
    active = torch.zeros_like(criterion_mask)
    active_pairs = total_pairs = 0
    for rows in groups.values():
        valid = criterion_mask[rows[0]]
        if not torch.equal(criterion_mask[rows], valid.expand(len(rows), -1)):
            raise ValueError("Responses sharing a uid must have identical criterion counts")
        group_scores = scores[rows][:, valid]
        total_pairs += int(valid.sum())
        if len(rows) < 2:
            continue
        std = group_scores.std(dim=0, unbiased=True)
        varying = std > 0
        group_adv = (group_scores - group_scores.mean(dim=0)) / (std + epsilon)
        group_adv[:, ~varying] = 0.0
        cols = valid.nonzero(as_tuple=True)[0]
        row_ids = torch.tensor(rows, device=scores.device)
        advantages[row_ids[:, None], cols] = group_adv
        active[row_ids[:, None], cols] = varying
        active_pairs += int(varying.sum())
    return advantages, active, active_pairs / total_pairs


def criterion_metrics(scores, criterion_mask, active_fraction):
    all_pass = ((scores == 1) | ~criterion_mask).all(dim=-1)
    return {
        "reward/criterion_micro_pass_rate": scores[criterion_mask].mean().item(),
        "reward/prompt_all_pass_rate": all_pass.float().mean().item(),
        "mrtpo/active_pair_fraction": active_fraction,
    }


def encode_replay_prompt(content, tokenizer, max_length, truncation="error", chat_template_kwargs=None,
                         ids_transform=None):
    """Render and left-pad a text-only prompt exactly like RLHFDataset.

    ``ids_transform`` (optional) maps the list of prompt token ids to a new list before padding.
    """
    from verl.utils.torch_functional import postprocess_data

    messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
    raw = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **(chat_template_kwargs or {}),
    )
    inputs = tokenizer(raw, return_tensors="pt", add_special_tokens=False)
    if ids_transform is not None:
        ids = torch.tensor([ids_transform(inputs["input_ids"][0].tolist())], dtype=inputs["input_ids"].dtype)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    return postprocess_data(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_length=max_length, pad_token_id=tokenizer.pad_token_id,
        left_pad=True, truncation=truncation,
    )


def is_privileged(extra):
    return extra.get("replay_mode") == "privileged"


def replay_blocks(data, row, extra):
    blocks = json.loads(data.non_tensor_batch["replay_blocks_json"][row])
    if len(blocks) != int(extra["n_criteria"]):
        raise ValueError("replay block count does not match n_criteria")
    return blocks


def build_criterion_replay(data, active, tokenizer, max_prompt_length=2048, truncation="error"):
    """Leave-one-out replay: one (response, criterion) row per active pair.

    Returns a DataProto with the replay context in ``input_ids`` and the full context in ``full_input_ids`` (both
    followed by the unchanged response), and the replayed (row, criterion) pairs.
    """
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    pairs = active.nonzero(as_tuple=False).cpu()
    if pairs.numel() == 0:
        return None, pairs
    extras = data.non_tensor_batch["extra_info"]
    privileged = {row for row in pairs[:, 0].tolist() if is_privileged(extras[row])}
    blocks_by_row = {}
    if privileged:
        kept = []
        for row, criterion in pairs.tolist():
            if row in privileged:
                if row not in blocks_by_row:
                    blocks_by_row[row] = replay_blocks(data, row, extras[row])
                if not blocks_by_row[row][criterion]:
                    continue
            kept.append((row, criterion))
        pairs = torch.tensor(kept, dtype=torch.long).reshape(-1, 2)
        if pairs.numel() == 0:
            return None, pairs
    row_ids = pairs[:, 0]
    responses = data.batch["responses"][row_ids]
    response_mask = data.batch["response_mask"][row_ids]
    prompt_ids, prompt_masks, full_ids, full_masks = [], [], [], []
    encoded = {}
    # Privileged contexts carry extra blocks, so they get 1,024 more prompt tokens.
    replay_length = max_prompt_length + 1024 if privileged else max_prompt_length

    def encode(content):
        return encode_replay_prompt(content, tokenizer, replay_length, truncation)

    for row, criterion in pairs.tolist():
        if row in privileged:
            blocks = blocks_by_row[row]
            p0 = extras[row]["p0_text"]
            all_blocks = [block for block in blocks if block]
            full_text = p0 + "\n\n" + "\n\n".join(all_blocks)
            minus_blocks = [block for k, block in enumerate(blocks) if k != criterion and block]
            minus_text = p0 + ("\n\n" + "\n\n".join(minus_blocks) if minus_blocks else "")
            if (row, "full") not in encoded:
                encoded[(row, "full")] = encode(full_text)
            full = encoded[(row, "full")]
            key = (row, criterion)
            if key not in encoded:
                encoded[key] = encode(minus_text)
            full_ids.append(full[0][0])
            full_masks.append(full[1][0])
        else:
            uid = data.non_tensor_batch["uid"][row]
            key = (uid, criterion)
            if key not in encoded:
                extra = extras[row]
                minus_prompts = json.loads(extra["minus_prompts_json"])
                if len(minus_prompts) != int(extra["n_criteria"]):
                    raise ValueError("LOO prompt count does not match n_criteria")
                encoded[key] = encode(minus_prompts[criterion])
            if privileged:
                length = data.batch["responses"].size(-1)
                pad = replay_length - data.batch["input_ids"].size(-1) + length
                full_ids.append(torch.nn.functional.pad(data.batch["input_ids"][row, :-length], (pad, 0), value=tokenizer.pad_token_id))
                full_masks.append(torch.nn.functional.pad(data.batch["attention_mask"][row, :-length], (pad, 0)))
        ids, mask = encoded[key]
        prompt_ids.append(ids[0])
        prompt_masks.append(mask[0])
    input_ids = torch.cat((torch.stack(prompt_ids).to(responses.device), responses), dim=-1)
    attention_mask = torch.cat((torch.stack(prompt_masks).to(response_mask.device), response_mask), dim=-1)
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "responses": responses,
        "response_mask": response_mask,
    }
    if privileged:
        full_attention_mask = torch.cat((torch.stack(full_masks).to(response_mask.device), response_mask), dim=-1)
        tensors["full_input_ids"] = torch.cat((torch.stack(full_ids).to(responses.device), responses), dim=-1)
        tensors["full_attention_mask"] = full_attention_mask
        tensors["full_position_ids"] = compute_position_id_with_mask(full_attention_mask)
    else:
        for key in ("input_ids", "attention_mask", "position_ids"):
            tensors[f"full_{key}"] = data.batch[key][row_ids]
    return DataProto.from_dict(tensors), pairs


def criterion_js_tensor(js, pairs, active, response_length):
    """Place the replayed JS rows into a [responses, criteria, tokens] tensor (zeros where nothing was replayed)."""
    dense = torch.zeros((*active.shape, response_length), dtype=js.dtype)
    dense[pairs[:, 0], pairs[:, 1]] = js
    return dense
