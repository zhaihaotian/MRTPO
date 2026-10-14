"""Training split: drop questions whose gold SQL fails and prompts longer than SQL-R1's 4,096-token limit.

Prompt length = tokens of the chat-templated prompt (with the generation prompt) under the Qwen2.5-Coder-7B-Instruct
tokenizer. Spider train: 7,000 -> 6,915 (3 failing gold SQL, 82 prompts over 4,096 tokens).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT, help='data root holding datasets/ (default: this folder)')
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-Coder-7B-Instruct',
                        help='path or Hugging Face id of Qwen2.5-Coder-7B-Instruct (only the tokenizer is loaded)')
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    datasets = args.root / 'datasets'
    kept = dropped_gold = dropped_length = 0
    with open(datasets / 'train.jsonl') as source, open(datasets / 'train_filtered.jsonl', 'w') as out:
        for line in source:
            record = json.loads(line)
            if not record['gold_ok']:
                dropped_gold += 1
                continue
            n = len(tokenizer.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True))
            if n > args.max_prompt_tokens:
                dropped_length += 1
                continue
            record['prompt_tokens'] = n
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
    print(json.dumps({'kept': kept, 'dropped_gold_failed': dropped_gold,
                      f'dropped_prompt_over_{args.max_prompt_tokens}': dropped_length}))


if __name__ == '__main__':
    main()
