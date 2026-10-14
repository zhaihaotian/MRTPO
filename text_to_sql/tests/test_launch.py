"""CPU checks of the training launcher's resolved settings (no GPU, no model)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / 'training' / 'launch.py'


def dry_run(*arguments):
    out = subprocess.run([sys.executable, str(LAUNCH), *arguments, '--seed', '42', '--model', '/m/Qwen2.5-Coder-7B-Instruct',
                          '--output', '/tmp/run', '--dry-run'], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.mark.parametrize('method,estimator,saw', [('grpo', 'grpo', False), ('gdpo', 'gdpo', False),
                                                  ('mrtpo', 'mrtpo', False), ('opsd', 'opsd', False),
                                                  ('gdpo_saw', 'gdpo', True), ('mrtpo_saw', 'mrtpo', True)])
def test_methods(method, estimator, saw):
    s = dry_run('--method', method)['settings']
    assert s['algorithm.adv_estimator'] == estimator
    assert s['algorithm.saw'] is saw
    assert s['algorithm.mrtpo.reallocation_bound'] is True and s['algorithm.mrtpo.reallocation_ratio'] == 1.0
    assert s['algorithm.mrtpo.beta'] == 0.5 and s['algorithm.mrtpo.ramp_steps'] == 10
    assert s['actor_rollout_ref.actor.use_kl_loss'] is (method != 'opsd')
    assert s['data.train_files'].endswith('train_opsd.parquet' if method == 'opsd' else 'parquet/train.parquet')
    assert (s['data.train_batch_size'], s['actor_rollout_ref.rollout.n']) == (32, 8)
    assert (s['data.max_prompt_length'], s['data.max_response_length']) == (4096, 2048)
    assert (s['actor_rollout_ref.rollout.temperature'], s['actor_rollout_ref.rollout.top_p'],
            s['actor_rollout_ref.rollout.top_k']) == (1.1, 1.0, -1)
    assert s['trainer.total_training_steps'] == 100 and s['trainer.n_gpus_per_node'] == 8
    for key in ('+data.seed', '+actor_rollout_ref.actor.seed', '+actor_rollout_ref.rollout.seed', 'trainer.seed'):
        assert s[key] == 42


def test_opsd_teacher():
    s = dry_run('--method', 'opsd')['settings']
    assert s['algorithm.opsd.teacher_template'] == 'reference_field_turn_repeat'
    assert (s['algorithm.opsd.topk'], s['algorithm.opsd.token_clip']) == (64, 2.0)
    assert s['actor_rollout_ref.ref.log_prob_max_token_len_per_gpu'] == 12288


def test_beta_is_an_mrtpo_option():
    assert dry_run('--method', 'mrtpo')['settings']['algorithm.mrtpo.beta'] == 0.5
    assert dry_run('--method', 'mrtpo', '--beta', '0.25')['settings']['algorithm.mrtpo.beta'] == 0.25
    with pytest.raises(subprocess.CalledProcessError):
        dry_run('--method', 'gdpo', '--beta', '0.25')
