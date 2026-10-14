"""BFCL V4 single-turn inference and scoring for one checkpoint on one GPU (resumes saved cases)."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from protocol import categories, expected_counts

DECODING = dict(temperature=0.6, top_p=0.95, top_k=-1, repetition_penalty=1, seed=0, max_new_tokens=8192,
                max_context_length=32768)
CONCURRENCY = 48


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Base model or <run>/actor/global_step_<step> directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--server-python', required=True, help='Interpreter of the vLLM 0.11.0 inference environment')
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()
    out = a.output.resolve()
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    os.environ.update(BFCL_PROJECT_ROOT=str(out), LOCAL_SERVER_ENDPOINT='127.0.0.1', LOCAL_SERVER_PORT=str(a.port),
                      BFCL_MAX_NEW_TOKENS=str(DECODING['max_new_tokens']))
    sys.path.insert(0, str(Path(__file__).parent / 'v4'))
    from adapter import register, RLLAHandler
    register()
    from bfcl_eval._llm_response_generation import multi_threaded_inference
    from bfcl_eval.utils import make_json_serializable, load_dataset_entry
    from transformers import AutoTokenizer
    cases = {c: load_dataset_entry(c) for c in categories()}
    counts = expected_counts()
    for c, entries in cases.items():
        if len(entries) != counts[c]:
            raise ValueError(f'{c}: wrong benchmark version or dataset size')
    config = dict(version='v4', source_commit='6ea57973c7a6097fd7c5915698c54c17c5b1b6c8',
                  model=str(Path(a.model).resolve()) if Path(a.model).exists() else a.model,
                  counts=counts, concurrency=CONCURRENCY, **DECODING)
    record = out / 'inference.json'
    if record.exists() and json.loads(record.read_text()) != config:
        raise ValueError('Output directory belongs to a different model or protocol')
    record.write_text(json.dumps(config, indent=2) + '\n')
    command = [a.server_python, '-m', 'vllm.entrypoints.openai.api_server', '--model', a.model,
               '--served-model-name', 'eval', '--host', '127.0.0.1', '--port', str(a.port),
               '--dtype', 'bfloat16', '--tensor-parallel-size', '1', '--gpu-memory-utilization', '0.85',
               '--max-model-len', str(DECODING['max_context_length']), '--max-num-seqs', '64',
               '--max-num-batched-tokens', '8192', '--generation-config', 'vllm', '--seed', '0', '--enforce-eager',
               '--disable-log-requests']
    env = os.environ.copy()
    env.pop('VLLM_PORT', None)  # BFCL uses this name for its endpoint; vLLM for internal rendezvous
    with (out / 'server.log').open('a') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 480
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Server unavailable; inspect server.log')
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health', timeout=2).close()
                    break
                except OSError:
                    time.sleep(2)
            handler = RLLAHandler('rlla-eval', DECODING['temperature'])
            handler.tokenizer = AutoTokenizer.from_pretrained(a.model)
            handler.max_context_length = DECODING['max_context_length']
            handler.model_path_or_id = 'eval'
            handler.backend = 'vllm'
            handler.is_fc_model = False
            errors, streams, pending = [], {}, []
            try:
                for cat, entries in cases.items():
                    path = out / 'raw' / f'{cat}.jsonl'
                    saved = {}
                    if path.exists():
                        for line in path.read_text().splitlines():
                            try:
                                row = json.loads(line)
                                saved[row['id']] = row
                            except json.JSONDecodeError:
                                pass  # an interrupted append is regenerated
                        path.write_text(''.join(json.dumps(r) + '\n' for r in saved.values()))
                    streams[cat] = path.open('a', buffering=1)
                    pending.extend((cat, copy.deepcopy(e)) for e in entries if e['id'] not in saved)
                with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                    futures = {pool.submit(multi_threaded_inference, handler, entry, entry['id'].endswith('_0'), False):
                               (cat, entry['id']) for cat, entry in pending}
                    for future in as_completed(futures):
                        cat, case_id = futures[future]
                        try:
                            result = make_json_serializable(future.result())
                            value = result.get('result')
                            error = isinstance(value, str) and value.startswith('Error during inference:')
                            context = error and any(s in value.lower() for s in ('maximum context length',
                                                                                 'maximum model length'))
                            if error and not context:
                                errors.append(result)  # infrastructure error: retried on the next run
                            else:
                                streams[cat].write(json.dumps(result) + '\n')
                        except Exception as error:
                            errors.append(dict(id=case_id, error=repr(error)))
            finally:
                for stream in streams.values():
                    stream.close()
            (out / 'inference_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
            if errors:
                raise RuntimeError(f'{len(errors)} infrastructure errors; rerun the same command to complete them')
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    subprocess.run([sys.executable, str(Path(__file__).with_name('score.py')), '--output', str(out)], check=True)


if __name__ == '__main__':
    main()
