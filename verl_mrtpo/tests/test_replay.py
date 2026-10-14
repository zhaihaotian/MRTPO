import json

import numpy as np
import torch

from mrtpo.criteria import build_criterion_replay, criterion_js_tensor, encode_replay_prompt
from mrtpo.hir import HIRQuestionBuilder
from verl import DataProto

from helpers import CharTokenizer

TOK = CharTokenizer()
QUESTION = "Describe the sea."
CRITERIA = ["Use two paragraphs", "End with a question?", "No commas"]


def decode(ids, mask):
    return "".join(chr((int(i) - 3) % 1000) for i, m in zip(ids, mask) if m)


def batch(kind, length=6, prompt_length=200):
    builder = HIRQuestionBuilder()
    rows, extras, blocks = 4, [], []
    if kind == "hir":
        prompt = builder._concat_question(QUESTION, CRITERIA, "type3")
        minus = [builder._concat_question(QUESTION, CRITERIA[:j] + CRITERIA[j + 1:], "type3") for j in range(3)]
        extra = {"n_criteria": 3, "q_type": "type3", "criteria_json": json.dumps(CRITERIA),
                 "minus_prompts_json": json.dumps(minus)}
        extras = [extra] * rows
        blocks = ["[]"] * rows
    else:
        prompt = "P0"
        extras = [{"n_criteria": 3, "replay_mode": "privileged", "p0_text": prompt}] * rows
        blocks = [json.dumps(["FMT", "" if r % 2 else "ERR", "REF"]) for r in range(rows)]
    ids, mask = encode_replay_prompt(prompt, TOK, prompt_length)
    responses = torch.arange(rows * length).reshape(rows, length) % 50 + 3
    response_mask = torch.ones(rows, length)
    response_mask[1, 4:] = 0
    full = torch.cat([ids.expand(rows, -1), responses], -1)
    attention = torch.cat([mask.expand(rows, -1), response_mask.long()], -1)
    return DataProto.from_dict(
        tensors={"input_ids": full, "attention_mask": attention, "position_ids": attention.cumsum(-1),
                 "responses": responses, "response_mask": response_mask},
        non_tensors={"uid": np.array(["a"] * rows, dtype=object), "extra_info": np.array(extras, dtype=object),
                     "replay_blocks_json": np.array(blocks, dtype=object)})


def test_leave_one_out_prompts_and_unchanged_responses():
    data = batch("hir")
    active = torch.tensor([[True, False, True]] * 4)
    replay, pairs = build_criterion_replay(data, active, TOK, max_prompt_length=200)
    assert pairs.tolist() == [[r, c] for r in range(4) for c in (0, 2)]
    minus = json.loads(data.non_tensor_batch["extra_info"][0]["minus_prompts_json"])
    for i, (row, criterion) in enumerate(pairs.tolist()):
        prompt = decode(replay.batch["input_ids"][i, :200], replay.batch["attention_mask"][i, :200])
        assert prompt == TOK.apply_chat_template([{"role": "user", "content": minus[criterion]}])
        assert CRITERIA[criterion] not in prompt
        assert torch.equal(replay.batch["responses"][i], data.batch["responses"][row])
        assert torch.equal(replay.batch["full_input_ids"][i], data.batch["input_ids"][row])


def test_privileged_blocks_skip_empty_and_drop_one_block():
    data = batch("sql")
    active = torch.ones(4, 3, dtype=torch.bool)
    replay, pairs = build_criterion_replay(data, active, TOK, max_prompt_length=50)
    assert [1, 1] not in pairs.tolist() and [0, 1] in pairs.tolist()   # empty error block is not replayed
    width = 50 + 1024
    for i, (row, criterion) in enumerate(pairs.tolist()):
        blocks = json.loads(data.non_tensor_batch["replay_blocks_json"][row])
        full = decode(replay.batch["full_input_ids"][i, :width], replay.batch["full_attention_mask"][i, :width])
        minus = decode(replay.batch["input_ids"][i, :width], replay.batch["attention_mask"][i, :width])
        kept = [b for b in blocks if b]
        assert full == TOK.apply_chat_template([{"role": "user", "content": "P0\n\n" + "\n\n".join(kept)}])
        others = [b for k, b in enumerate(blocks) if k != criterion and b]
        assert minus == TOK.apply_chat_template([{"role": "user", "content": "P0\n\n" + "\n\n".join(others)}])


def test_criterion_js_tensor_places_replayed_rows():
    active = torch.tensor([[True, True], [False, False], [True, False]])
    pairs = active.nonzero()
    js = torch.arange(len(pairs) * 5, dtype=torch.float32).reshape(-1, 5)
    dense = criterion_js_tensor(js, pairs, active, 5)
    assert torch.equal(dense[0, 1], js[1]) and torch.equal(dense[2, 0], js[2])
    assert torch.equal(dense[1], torch.zeros(2, 5)) and torch.equal(dense[2, 1], torch.zeros(5))
