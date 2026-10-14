import json

import numpy as np
import pytest
import torch

from mrtpo.cort import build_cort_replay, cort_token_weights, counterfactual_prompt, raw_question
from mrtpo.hir import HIRQuestionBuilder
from verl import DataProto

from helpers import CharTokenizer

QUESTION = 'Provide a detailed response to the following query: "Why is the sky blue?"'
CRITERIA = ["Your response should contain at least 5 sentences", "Wrap your entire response with double quotation marks."]
OPTIONS = {"eta": 0.5, "tau": 1.0, "center": 0.0, "ramp_steps": 100}


@pytest.mark.parametrize("q_type", [f"type{i}" for i in range(1, 7)])
def test_raw_question_inverts_every_layout(q_type):
    builder = HIRQuestionBuilder()
    assert raw_question(builder._concat_question(QUESTION, CRITERIA, q_type), CRITERIA, q_type) == QUESTION
    minus = [builder._concat_question(QUESTION, CRITERIA[:j] + CRITERIA[j + 1:], q_type) for j in range(2)]
    extra = {"criteria_json": json.dumps(CRITERIA), "minus_prompts_json": json.dumps(minus), "n_criteria": 2,
             "q_type": q_type}
    assert counterfactual_prompt(extra) == QUESTION
    with pytest.raises(ValueError):
        raw_question("something else entirely", CRITERIA, q_type)


def test_cort_weights_mean_one_bounded_and_scheduled():
    torch.manual_seed(0)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    factual, counterfactual = torch.randn(2, 6), torch.randn(2, 6)
    w0, m0 = cort_token_weights(factual, counterfactual, mask, trainer_step=0, options=OPTIONS)
    assert torch.equal(w0, torch.ones_like(w0)) and m0["cort/schedule"] == 0.0
    w, m = cort_token_weights(factual, counterfactual, mask, trainer_step=100, options=OPTIONS)
    assert m["cort/schedule"] == 1.0
    torch.testing.assert_close((w * mask).sum(-1) / mask.sum(-1), torch.ones(2))
    assert torch.equal(w[0, 4:], torch.ones(2))
    order = torch.argsort((factual - counterfactual)[1])
    assert torch.all(w[1][order][1:] >= w[1][order][:-1])
    s = torch.sigmoid(factual - counterfactual) - 0.5
    provisional = 1 + 0.5 * s
    torch.testing.assert_close(w[1], provisional[1] / provisional[1].mean())
    _, mid = cort_token_weights(factual, counterfactual, mask, trainer_step=50, options=OPTIONS)
    assert mid["cort/schedule"] == pytest.approx(0.5)


def test_build_cort_replay_uses_criteria_free_prompt():
    tok = CharTokenizer()
    builder = HIRQuestionBuilder()
    minus = [builder._concat_question(QUESTION, CRITERIA[:j] + CRITERIA[j + 1:], "type4") for j in range(2)]
    extra = {"criteria_json": json.dumps(CRITERIA), "minus_prompts_json": json.dumps(minus), "n_criteria": 2,
             "q_type": "type4"}
    responses = torch.tensor([[5, 6, 7], [8, 9, 0]])
    data = DataProto.from_dict(
        tensors={"responses": responses, "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])},
        non_tensors={"extra_info": np.array([extra, extra], dtype=object), "uid": np.array(["a", "a"], dtype=object)})
    replay = build_cort_replay(data, tok, max_prompt_length=200)
    prompt_ids = replay.batch["input_ids"][0, :200][replay.batch["attention_mask"][0, :200].bool()]
    expected = tok(tok.apply_chat_template([{"role": "user", "content": QUESTION}]))["input_ids"]
    assert prompt_ids.tolist() == expected
    assert torch.equal(replay.batch["responses"], responses)
