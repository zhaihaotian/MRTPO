"""The typed replay reference (scripts/fetch_data.py) and how the replay contexts and the reward use it."""
import json
import unittest

import torch

from _load import fetch_data, replay, rlla

INSTRUCTION = """You are a helpful assistant.

**Available Tools**
In your response, you can use the following tools:
1. Name: book_room
Description: Book a room.
Parameters: {"capacity": {"description": "people", "type": "int"}, "budget": {"description": "max price", "type": "float, optional"}, "quiet": {"description": "quiet floor", "type": "boolean"}, "city": {"description": "city", "type": "string"}}
2. Name: get_time
Description: Current time.
Parameters: {}

**Steps for Each Turn**
1. **Think:** ..."""


def label(calls):
    return '<think> plan </think>\n<tool_call>\n' + '\n'.join(json.dumps(c) for c in calls) + '\n</tool_call>'


class TypedLabel(unittest.TestCase):
    def test_converts_declared_numeric_and_boolean_strings(self):
        raw = label([{'name': 'book_room', 'parameters': {'capacity': '8', 'budget': '2.5e2', 'quiet': 'True',
                                                         'city': '100'}}])
        typed = fetch_data.typed_label(raw, INSTRUCTION)
        call = json.loads(typed.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
        self.assertEqual(call['parameters'], {'capacity': 8, 'budget': 250.0, 'quiet': True, 'city': '100'})
        self.assertTrue(typed.startswith('<think> plan </think>\n<tool_call>\n'))

    def test_leaves_non_literals_typed_values_and_unknown_tools(self):
        raw = label([{'name': 'book_room', 'parameters': {'capacity': 'eight', 'budget': 3.0}},
                     {'name': 'unknown_tool', 'parameters': {'capacity': '8'}},
                     {'name': 'get_time', 'parameters': {}}])
        self.assertEqual(fetch_data.typed_label(raw, INSTRUCTION), raw)  # nothing to convert -> byte-identical

    def test_direct_answer_label_unchanged(self):
        raw = '<think> no tool </think>\n<response> Hello </response>'
        self.assertEqual(fetch_data.typed_label(raw, INSTRUCTION), raw)

    def test_schema_parser(self):
        schemas = fetch_data.tool_schemas(INSTRUCTION)
        self.assertEqual(set(schemas), {'book_room', 'get_time'})
        self.assertEqual(fetch_data.declared_kind(schemas['book_room']['budget']), 'float')


class ReferenceUse(unittest.TestCase):
    def setUp(self):
        self.raw = label([{'name': 'book_room', 'parameters': {'capacity': '8', 'city': 'Oslo'}}])
        self.typed = fetch_data.typed_label(self.raw, INSTRUCTION)
        self.row = dict(instruction=INSTRUCTION, input='**Dialogue Records History**\n<user> book </user>',
                        output=self.raw, teacher_output=self.typed)

    def test_correctness_block_shows_typed_reference(self):
        full, ok = replay.context_messages(self.row, 'full')
        corr, _ = replay.context_messages(self.row, 'correctness')
        fmt, _ = replay.context_messages(self.row, 'format')
        self.assertTrue(ok)
        self.assertIn('Parameter `capacity`: 8.', corr[1]['content'])
        self.assertNotIn('"8"', corr[1]['content'])
        self.assertIn('Parameter `capacity`: 8.', full[1]['content'])
        self.assertNotIn('correctness reference', fmt[1]['content'])
        self.assertEqual(full[0]['content'], INSTRUCTION)
        untyped, _ = replay.context_messages(dict(self.row, teacher_output=None), 'correctness')
        self.assertIn('Parameter `capacity`: "8".', untyped[1]['content'])

    def test_direct_answer_rows_have_no_correctness_context(self):
        row = dict(self.row, output='<think> a </think>\n<response> b </response>', teacher_output=None)
        messages, ok = replay.context_messages(row, 'correctness')
        self.assertFalse(ok)
        self.assertEqual(messages[1]['content'], row['input'])

    def test_reward_scores_the_original_label(self):
        quoted = '<|im_start|>assistant\n' + self.raw + '<|im_end|>'
        typed = '<|im_start|>assistant\n' + self.typed + '<|im_end|>'
        self.assertEqual(rlla.compute_score(quoted, self.raw), (4.0, 1.0, 3.0))
        total, fmt, correctness = rlla.compute_score(typed, self.raw)
        self.assertEqual(fmt, 1.0)
        self.assertLess(correctness, 3.0)  # the typed value 8 does not match the label's "8"


class ContextTensors(unittest.TestCase):
    def test_splice_into_truncated_rollout_prompt(self):
        class Tok:
            pad_token_id, eos_token_id = 0, 1

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                text = ''.join(f'[{m["role"]}]{m["content"]}[end]' for m in messages) + '[assistant]'
                return [2 + ord(ch) % 50 for ch in text]

        tok = Tok()
        row = dict(instruction='sys', input='question', output=label([{'name': 'get_time', 'parameters': {}}]))
        base = tok.apply_chat_template(replay._messages(row['instruction'], row['input']))[3:]  # left-truncated
        prompts = torch.tensor([base])
        responses = torch.tensor([[5, 6, 7]])
        tensors, applicable = replay.context_tensors(
            tokenizer=tok, responses=responses, response_mask=torch.ones_like(responses), extra_info=[row],
            context='full', base_prompts=prompts, base_prompt_mask=torch.ones_like(prompts))
        full = tok.apply_chat_template(replay.context_messages(row, 'full')[0])
        ids = tensors['input_ids'][0][tensors['attention_mask'][0].bool()].tolist()
        self.assertTrue(applicable.item())
        self.assertEqual(ids, full[3:] + [5, 6, 7])


if __name__ == '__main__':
    unittest.main()
