import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mrtpo.opsd import (
    build_teacher_batch,
    emitted_think_transform,
    gather_student_log_probs,
    opsd_token_loss,
    teacher_messages,
    teacher_topk_log_probs,
)
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss

from helpers import CharTokenizer


def official_terms(student, teacher, clip):
    # OPSD generalized_jsd_loss with beta=0 (forward KL), temperature 1.1, per-entry clip.
    student_logp = F.log_softmax(student / 1.1, dim=-1)
    teacher_logp = F.log_softmax(teacher.detach() / 1.1, dim=-1)
    return F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).clamp(max=clip)


@pytest.mark.parametrize("k,clip", [(50, 0.05), (7, 0.05), (7, 2.0)])
def test_topk_forward_kl_matches_reference(k, clip):
    generator = torch.Generator().manual_seed(41)
    student = (torch.randn(3, 5, 50, generator=generator) * 3).requires_grad_()
    teacher = torch.randn(3, 5, 50, generator=generator) * 3
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    topk = teacher_topk_log_probs(teacher / 1.1, k)
    gathered = gather_student_log_probs(student / 1.1, topk["opsd_topk_ids"], chunk_size=2)
    losses, clipped = opsd_token_loss(gathered, topk["opsd_topk_logp"], clip)
    terms = official_terms(student, teacher, clip).gather(-1, topk["opsd_topk_ids"].long())
    actual = agg_loss(losses, mask, "token-mean")
    expected = terms[mask.bool()].sum() / mask.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(clipped, (terms >= clip).float().mean(-1))
    actual.backward()
    grad = student.grad.clone()
    student.grad = None
    expected.backward()
    torch.testing.assert_close(student.grad, grad, rtol=1e-5, atol=1e-6)


def test_identical_distributions_have_zero_loss():
    student = torch.randn(2, 3, 50, requires_grad=True)
    topk = teacher_topk_log_probs(student.detach() / 1.1, 50)
    losses, clipped = opsd_token_loss(gather_student_log_probs(student / 1.1, topk["opsd_topk_ids"]),
                                      topk["opsd_topk_logp"], 2.0)
    torch.testing.assert_close(losses, torch.zeros_like(losses), rtol=0, atol=1e-6)
    assert not clipped.any()


def test_teacher_context_is_prompt_reference_prompt():
    tok = CharTokenizer()
    extra = {"p0_text": "Find the count.", "ref_solution": "<think>count</think><answer>SELECT 1;</answer>"}
    responses = torch.tensor([[9, 10, 11], [12, 0, 0]])
    data = DataProto.from_dict(
        tensors={"responses": responses, "response_mask": (responses != 0).long()},
        non_tensors={"extra_info": np.array([extra, extra], dtype=object)})
    teacher = build_teacher_batch(data, tok, max_prompt_length=100, emitted_think_tokens=False)
    width = 100 + 1024
    ids = teacher.batch["input_ids"][0, :width][teacher.batch["attention_mask"][0, :width].bool()]
    messages = teacher_messages(extra["p0_text"], extra["ref_solution"])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == messages[2]["content"] == extra["p0_text"]
    assert ids.tolist() == tok(tok.apply_chat_template(messages))["input_ids"]
    assert torch.equal(teacher.batch["responses"], responses)
    with pytest.raises(ValueError):
        build_teacher_batch(data, tok, template="official")


def test_emitted_think_transform_rewrites_bpe_think_prefix():
    class Tok:
        table = {"<think>": [13708, 766, 29], "<": [27], "think": [26865]}

        def __call__(self, text, add_special_tokens):
            return {"input_ids": self.table[text]}

    f = emitted_think_transform(Tok())
    assert f([1, 13708, 766, 397, 5, 13708, 766, 29]) == [1, 27, 26865, 397, 5, 27, 26865, 29]
    assert f([13708, 5, 766]) == [13708, 5, 766]

    class SameTok(Tok):
        table = {"<think>": [27, 26865, 29], "<": [27], "think": [26865]}

    assert emitted_think_transform(SameTok()) is None
