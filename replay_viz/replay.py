"""Replay responses with a base model and show where each reward's weight lands.

For every reward k the same response token ids are scored under two prompts:
  P_all     : the prompt P_0 with every non-empty verifier block appended to the end of the user message
              (blank-line separated, in the order of the rewards), or P_0 itself when nothing is appended;
  P_minus_k : the same prompt without block k, or, for a reward that removes information the prompt already shows,
              the given prompt without that information.
At each response position the Jensen-Shannon divergence between the two next-token distributions gives
q = sqrt(JS / ln 2) and the MRTPO token weight w = (1 - beta) + beta * q / mean(q).

usage: python replay.py --model PATH --input examples/text_to_sql.json --out viz/ [--sample N] [--png]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

try:
    import mrtpo  # noqa: F401  (installed with verl_mrtpo)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verl_mrtpo"))
from mrtpo.advantage import allocate_criterion_advantage
from mrtpo.js import chunked_distribution_js
from render import render_html, render_png

LOG_TWO = math.log(2.0)


# ---------------------------------------------------------------------------------------------- contexts

def _user_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    raise ValueError("The prompt needs a user message")


def append_blocks(messages, blocks):
    """A copy of `messages` whose last user message ends with the blocks, each after a blank line."""
    out = copy.deepcopy(messages)
    if blocks:
        i = _user_index(out)
        out[i]["content"] = out[i]["content"] + "\n\n" + "\n\n".join(blocks)
    return out


def _minus_prompt(messages, minus):
    """`minus_prompt` is either a list of messages or the text of the last user message."""
    if isinstance(minus, list):
        return copy.deepcopy(minus)
    out = copy.deepcopy(messages)
    out[_user_index(out)]["content"] = minus
    return out


def build_contexts(messages, rewards, blocks=None):
    """P_all and one P_minus_k per reward (None when reward k has an empty block and is not replayed).

    `rewards` are dicts with a `name` and either `block` (text added in the replay) or `minus_prompt` (the prompt
    without the information that reward checks). `blocks` maps reward names to blocks that replace the given ones.
    """
    blocks = blocks or {}
    unknown = set(blocks) - {r["name"] for r in rewards}
    if unknown:
        raise ValueError(f"Blocks given for unknown rewards: {sorted(unknown)}")
    texts = []
    for reward in rewards:
        if ("block" in reward) == ("minus_prompt" in reward):
            raise ValueError(f"Reward {reward['name']!r} needs exactly one of 'block' or 'minus_prompt'")
        texts.append(blocks.get(reward["name"], reward.get("block")) or "")
    added = [text for text in texts if text]
    full = append_blocks(messages, added)
    minus = []
    for k, reward in enumerate(rewards):
        if "minus_prompt" in reward:
            minus.append(append_blocks(_minus_prompt(messages, reward["minus_prompt"]), added))
        elif not texts[k]:
            minus.append(None)
        else:
            minus.append(append_blocks(messages, [text for j, text in enumerate(texts) if j != k and text]))
    return full, minus, texts


# ---------------------------------------------------------------------------------------------- weights

def token_weights(js, beta):
    """q = sqrt(JS / ln 2) and MRTPO's weights w = (1 - beta) + beta * q / mean(q) (training code, warm-up done)."""
    js = js.float().reshape(1, -1)
    q = torch.sqrt(js.clamp(min=0.0, max=LOG_TWO) / LOG_TWO)
    ones = torch.ones_like(js)
    _, w = allocate_criterion_advantage(ones, js, ones.bool(), torch.ones(1, dtype=torch.bool),
                                        schedule=1.0, beta=beta)
    return q[0], w[0]


# ---------------------------------------------------------------------------------------------- model

class Replayer:
    def __init__(self, model, device="auto", dtype="bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype)).to(device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        self.eos_ids.discard(None)

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def prompt_ids(self, messages):
        """Token ids of a prompt, rendered and tokenized the way the training dataset does."""
        return self.tokenizer(self.render(messages), add_special_tokens=False)["input_ids"]

    def response_ids(self, text):
        """Tokenize a given response once and end it with the end-of-turn token, as a sampled response ends."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        end = self.tokenizer.eos_token_id
        return ids if end is None or (ids and ids[-1] in self.eos_ids) else ids + [end]

    @torch.no_grad()
    def sample(self, messages, n, max_new_tokens, temperature=1.0, top_p=1.0, top_k=0, seed=0):
        """Sample n responses under P_0 from the raw policy and keep the exact token ids."""
        torch.manual_seed(seed)
        ids = torch.tensor([self.prompt_ids(messages)], device=self.device)
        out = self.model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=temperature, top_p=top_p,
            top_k=top_k, repetition_penalty=1.0, max_new_tokens=max_new_tokens, num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
        )
        responses = []
        for row in out[:, ids.shape[1]:].tolist():
            kept = []
            for token in row:
                kept.append(token)
                if token in self.eos_ids:
                    break
            responses.append(kept)
        return responses

    @torch.no_grad()
    def response_logits(self, prompt_ids, response_ids):
        """Logits [T, V] that predict each response token; prompt ids and response ids are spliced, never re-tokenized."""
        full = torch.tensor([prompt_ids + response_ids], device=self.device)
        return self.model(full, logits_to_keep=len(response_ids) + 1).logits[0, :-1]

    @torch.no_grad()
    def replay(self, full, minus, response_ids, temperature=1.0, chunk_size=128):
        """Per-token JS between P_all and each P_minus_k on the same response (None where not replayed)."""
        full_logits = self.response_logits(self.prompt_ids(full), response_ids)
        out = []
        for messages in minus:
            if messages is None:
                out.append(None)
                continue
            logits = self.response_logits(self.prompt_ids(messages), response_ids)
            out.append(chunked_distribution_js(full_logits, logits, chunk_size=chunk_size,
                                               temperature=temperature).cpu())
            del logits
        return out

    def token_strings(self, response_ids):
        return [self.tokenizer.decode([token]) for token in response_ids]


# ---------------------------------------------------------------------------------------------- run

def _responses(example):
    for item in example.get("responses", []):
        yield (item, {}) if isinstance(item, str) else (item["text"], item.get("blocks", {}))


def run(example, replayer, *, beta=0.5, sample=0, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=1024, seed=0):
    """Replay every response of one example; returns the JSON record that the page is rendered from."""
    messages, rewards = example["messages"], example["rewards"]
    if sample:
        items = [(ids, {}, "sampled") for ids in
                 replayer.sample(messages, sample, max_new_tokens, temperature, top_p, top_k, seed)]
    else:
        items = [(replayer.response_ids(text), blocks, "given") for text, blocks in _responses(example)]
        if not items:
            raise ValueError("The input has no responses; give some or pass --sample N")
    record = {
        "name": example.get("name", ""), "model": str(replayer.model.name_or_path), "beta": beta,
        "temperature": temperature, "p0": replayer.render(messages), "responses": [],
    }
    for response_ids, blocks, source in items:
        full, minus, texts = build_contexts(messages, rewards, blocks)
        js = replayer.replay(full, minus, response_ids, temperature=temperature)
        rows = []
        for reward, block, values, minus_messages in zip(rewards, texts, js, minus):
            row = {"name": reward["name"], "block": block, "replayed": values is not None}
            if values is None:
                row.update(js=None, q=None, w=[1.0] * len(response_ids), p_minus=None)
            else:
                q, w = token_weights(values, beta)
                row.update(js=values.tolist(), q=q.tolist(), w=w.tolist(), p_minus=replayer.render(minus_messages))
            rows.append(row)
        record["responses"].append({
            "source": source,
            "text": replayer.tokenizer.decode(response_ids),
            "response_ids": response_ids,
            "tokens": replayer.token_strings(response_ids),
            "p_all": replayer.render(full),
            "rewards": rows,
        })
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--input", required=True, type=Path, help="example JSON (see README.md)")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--sample", type=int, default=0, help="sample N responses under P_0 instead of the given ones")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature; the replay divides logits by it, as training does")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 = off")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--png", action="store_true", help="also write one PNG per response (needs matplotlib)")
    args = parser.parse_args(argv)
    if not 0.0 <= args.beta < 1.0:
        parser.error("--beta must lie in [0, 1)")

    example = json.loads(args.input.read_text())
    replayer = Replayer(args.model, device=args.device, dtype=args.dtype)
    record = run(example, replayer, beta=args.beta, sample=args.sample, temperature=args.temperature,
                 top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    (args.out / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    (args.out / f"{stem}.html").write_text(render_html(record))
    written = [args.out / f"{stem}.json", args.out / f"{stem}.html"]
    if args.png:
        written += render_png(record, args.out / stem)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
