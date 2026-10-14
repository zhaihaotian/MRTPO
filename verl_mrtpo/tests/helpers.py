"""Shared CPU fixtures: random criterion batches and a one-token-per-character tokenizer."""
import json

import numpy as np
import torch

from verl import DataProto


class CharTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [3 + ord(c) % 1000 for c in text]
        if return_tensors == "pt":
            ids = torch.tensor([ids])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        return {"input_ids": ids}


def criterion_batch(seed=0, groups=4, n=8, criteria=3, length=12, step=20):
    """Random binary criterion scores, ragged response lengths and JS values in [0, ln 2]."""
    rng = np.random.default_rng(seed)
    rows = groups * n
    scores = (rng.uniform(size=(rows, criteria)) < 0.5).astype(int)
    lengths = rng.integers(1, length + 1, size=rows)
    mask = (torch.arange(length)[None, :] < torch.tensor(lengths)[:, None]).float()
    js = torch.tensor(rng.uniform(0, 0.69, size=(rows, criteria, length)), dtype=torch.float32)
    js = js * (torch.tensor(rng.uniform(size=(rows, criteria, length))) < 0.3)
    data = DataProto.from_dict(
        tensors={"response_mask": mask, "criterion_js": js},
        non_tensors={"uid": np.array([f"u{i // n}" for i in range(rows)], dtype=object),
                     "criteria_scores": np.array([json.dumps(s.tolist()) for s in scores], dtype=object)},
        meta_info={"global_steps": step},
    )
    return data
