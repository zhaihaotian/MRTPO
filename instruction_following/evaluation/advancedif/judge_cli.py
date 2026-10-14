"""Run the official AdvancedIF judge (facebookresearch/AdvancedIF, unmodified) against any OpenAI-compatible server.
Two settings of the official client only matter for a local judge server and are set here:
  * the judges run in asyncio's default thread pool, which Python caps at 32 threads, so --max_concurrency above 32
    has no effect; the pool is sized to --max-concurrency instead;
  * openai.OpenAI() is created with the default 600 s timeout; a long judgement on a fully loaded local server can take
    longer, so the timeout is raised to 3600 s.
The official CLI also lacks the --temperature option its evaluate() requires (the value is unused by the API call), so
evaluate() is called directly. --presence-penalty, --temperature and --thinking-token-budget are only used when
re-judging failed calls; omitted, the official request is unchanged.
The official request asks for {"type": "json_object"}; the judge then sometimes writes JSON booleans for the
rubric answers (the official parser expects strings ending in Yes/No and fails) or malformed JSON. The request is sent
with the official output schema instead (rubric answers are strings, the overall verdict is Yes/No), so the server's
structured decoding guarantees parseable output; what the judge writes inside each answer is unchanged.
usage: OPENAI_BASE_URL=http://127.0.0.1:8300/v1 OPENAI_API_KEY=local PYTHONPATH=<dir containing AdvancedIF> \
       python judge_cli.py --input advancedif.jsonl --output advancedif_judged.jsonl --model judge --max-concurrency 512"""
import argparse
import asyncio.base_events
import concurrent.futures
import json
import logging
from pathlib import Path

import openai

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
ap.add_argument('--model', default='judge'); ap.add_argument('--max-concurrency', type=int, default=512)
ap.add_argument('--max-completion-tokens', type=int, default=16384); ap.add_argument('--log-file', type=Path)
ap.add_argument('--presence-penalty', type=float, choices=(0.0, 1.5), default=0.0)
ap.add_argument('--temperature', type=float, choices=(0.0, 1.0))
ap.add_argument('--thinking-token-budget', type=int,
                help='Limit reasoning tokens while leaving room for the final judgement')
args = ap.parse_args()
if args.thinking_token_budget is not None:
    if not 0 < args.thinking_token_budget < args.max_completion_tokens:
        ap.error('--thinking-token-budget must be positive and smaller than --max-completion-tokens')

_run_in_executor = asyncio.base_events.BaseEventLoop.run_in_executor
def run_in_executor(loop, executor, func, *a):
    if executor is None and loop._default_executor is None:
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(args.max_concurrency, thread_name_prefix='judge'))
    return _run_in_executor(loop, executor, func, *a)
asyncio.base_events.BaseEventLoop.run_in_executor = run_in_executor

OUTPUT_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['rubrics_check', 'SATISFIED_ALL_REQUIREMENTS'],
                 'properties': {'rubrics_check': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                                'SATISFIED_ALL_REQUIREMENTS': {'type': 'string', 'enum': ['Yes', 'No']}}}
_OpenAI = openai.OpenAI
def make_client(**kw):
    client = _OpenAI(timeout=3600.0, **kw)
    create = client.chat.completions.create
    def create_with_schema(**req):
        req['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'advancedif_judgement', 'schema': OUTPUT_SCHEMA}}
        if args.presence_penalty:
            req['presence_penalty'] = args.presence_penalty
        if args.temperature is not None:
            req['temperature'] = args.temperature
        if args.thinking_token_budget is not None:
            req.setdefault('extra_body', {})['thinking_token_budget'] = args.thinking_token_budget
        resp = create(**req)
        text = resp.choices[0].message.content or ''
        if args.thinking_token_budget is not None:
            usage = getattr(resp, 'usage', None)
            details = getattr(usage, 'completion_tokens_details', None)
            logging.getLogger('judge_cli').info(
                'bounded judge metadata: thinking_token_budget=%s completion_tokens=%s '
                'reasoning_tokens=%s final_chars=%d',
                args.thinking_token_budget, getattr(usage, 'completion_tokens', None),
                getattr(details, 'reasoning_tokens', None), len(text))
        try:
            json.loads(text)
        except ValueError:   # the official code drops the raw output of a failed parse; keep its shape for diagnosis
            logging.getLogger('judge_cli').warning('unparseable judge output: finish_reason=%s chars=%d head=%r tail=%r',
                                                   resp.choices[0].finish_reason, len(text), text[:300], text[-300:])
            message = resp.choices[0].message
            reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
            logging.getLogger('judge_cli').warning(
                'failed judge metadata: completion_tokens=%s reasoning_chars=%d reasoning_exclamation_only=%s',
                getattr(resp.usage, 'completion_tokens', None), len(reasoning),
                bool(reasoning) and not reasoning.strip('! \t\r\n'))
        return resp
    client.chat.completions.create = create_with_schema
    return client
openai.OpenAI = make_client

from AdvancedIF.cli import evaluate  # noqa: E402  (after the patches above)

evaluate.callback(input_file=args.input, output_file=args.output, task_filter=None, api_key=None, model=args.model,
                  max_completion_tokens=args.max_completion_tokens, max_concurrency=args.max_concurrency,
                  temperature=0.0, log_level='INFO', log_file=args.log_file)
