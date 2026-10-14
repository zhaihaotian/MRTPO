"""OPSD baseline: frozen initial model as teacher with the reference solution in context, top-K forward KL."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mrtpo.criteria import encode_replay_prompt


# The teacher sees the student's prompt, the reference solution as a previous assistant turn, and the student's prompt
# again as a new user turn; the student's sampled response is then scored after that context.
TEACHER_TEMPLATE = "reference_field_turn_repeat"


def teacher_messages(p0, reference):
    """Chat messages of the teacher context (template "reference_field_turn_repeat")."""
    return [{"role": "user", "content": p0}, {"role": "assistant", "content": reference},
            {"role": "user", "content": p0}]


def emitted_think_transform(tokenizer):
    """Return a token-id transform that rewrites "<think>" in a prompt from its BPE form "<th"+"ink" to the
    form the policy actually emits, "<"+"think" (the trailing ">" / ">\n" token is left untouched).

    Qwen2.5 tokenizes the literal "<think>" as [<th, ink, >] but generates it as [<, think, >], so a teacher
    whose context holds the BPE form learns to predict "<th" where the student writes "<".
    """
    canon = tokenizer("<think>", add_special_tokens=False)["input_ids"][:2]
    emitted = [tokenizer("<", add_special_tokens=False)["input_ids"][0],
               tokenizer("think", add_special_tokens=False)["input_ids"][0]]
    if len(canon) != 2 or canon == emitted:
        return None

    def transform(ids):
        out, i = [], 0
        while i < len(ids):
            if ids[i : i + 2] == canon:
                out.extend(emitted)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    return transform


def build_teacher_batch(data, tokenizer, max_prompt_length=2048, teacher_chat_template_kwargs=None,
                        template=TEACHER_TEMPLATE, emitted_think_tokens=True, teacher_max_prompt_length=None):
    """Append the unchanged student responses to reference-conditioned prompts.

    The reference is ``extra_info["ref_solution"]`` (a worked solution ending in the SQL answer). Teacher prompts are
    left-padded to ``teacher_max_prompt_length`` (default: student max_prompt_length + 1024).
    """
    if template != TEACHER_TEMPLATE:
        raise ValueError(f"unknown OPSD teacher template: {template}")
    if teacher_max_prompt_length is None:
        teacher_max_prompt_length = max_prompt_length + 1024
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    prompt_ids, prompt_masks = [], []
    encoded = {}
    ids_transform = emitted_think_transform(tokenizer) if emitted_think_tokens else None
    for extra in data.non_tensor_batch["extra_info"]:
        key = (extra["p0_text"], extra["ref_solution"])
        if key not in encoded:
            encoded[key] = encode_replay_prompt(
                teacher_messages(key[0], key[1]), tokenizer, teacher_max_prompt_length, truncation="error",
                chat_template_kwargs=teacher_chat_template_kwargs, ids_transform=ids_transform,
            )
        ids, mask = encoded[key]
        prompt_ids.append(ids[0])
        prompt_masks.append(mask[0])
    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    input_ids = torch.cat((torch.stack(prompt_ids).to(responses.device), responses), dim=-1)
    attention_mask = torch.cat((torch.stack(prompt_masks).to(response_mask.device), response_mask), dim=-1)
    return DataProto.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
        "responses": responses,
        "response_mask": response_mask,
    })


def _gather_chunk(logits, token_ids):
    return F.log_softmax(logits, dim=-1, dtype=torch.float32).gather(-1, token_ids.long())


def gather_student_log_probs(logits, token_ids, chunk_size=128):
    """Gather full-vocabulary log-softmax with bounded fp32 activation storage.

    Logits have already been divided by the rollout temperature. Recompute each
    chunk's log-softmax during backward instead of retaining a tokens-by-vocab copy.
    """
    if logits.dim() == 3:
        return torch.stack([
            gather_student_log_probs(row, ids, chunk_size) for row, ids in zip(logits, token_ids)
        ])
    chunks = []
    for start in range(0, logits.size(0), chunk_size):
        values = logits[start : start + chunk_size]
        ids = token_ids[start : start + chunk_size]
        if torch.is_grad_enabled() and values.requires_grad:
            chunks.append(checkpoint(_gather_chunk, values, ids, use_reentrant=False))
        else:
            chunks.append(_gather_chunk(values, ids))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def teacher_topk_log_probs(logits, k, chunk_size=128):
    """Top-K of the full-vocabulary log-softmax; input logits are temperature-scaled."""
    if logits.dim() == 3:
        rows = [teacher_topk_log_probs(row, k, chunk_size) for row in logits]
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    ids, logp = [], []
    for chunk in logits.split(chunk_size, dim=0):
        values, indices = F.log_softmax(chunk, dim=-1, dtype=torch.float32).topk(k, dim=-1)
        ids.append(indices.to(torch.int32))
        logp.append(values)
    logp = torch.cat(logp, dim=0)
    return {"opsd_topk_ids": torch.cat(ids, dim=0), "opsd_topk_logp": logp}


def opsd_token_loss(student_topk_logp, teacher_topk_logp, token_clip=0.05):
    """Return clipped KL sums and the clipped-entry fraction at each position."""
    teacher_logp = teacher_topk_logp.detach().float()
    terms = teacher_logp.exp() * (teacher_logp - student_topk_logp)
    return terms.clamp(max=token_clip).sum(-1), (terms >= token_clip).float().mean(-1)
