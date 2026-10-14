"""Launcher settings for every method (no GPU)."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'training'))
import launch  # noqa: E402


def settings(method, size='1.5b', no_bound=False):
    args = SimpleNamespace(method=method, seed=4, model='Qwen/Qwen2.5-1.5B-Instruct', model_size=size,
                           data_dir=REPO / 'data/rlla_4k', output=Path('/tmp/mrtpo-test'), save_freq=100,
                           no_reallocation_bound=no_bound)
    return launch.configuration(args)


class Launch(unittest.TestCase):
    def test_shared_settings(self):
        for method in launch.METHODS:
            s, env, cmd = settings(method)
            self.assertEqual(s['algorithm.adv_estimator'], method)
            self.assertEqual((s['data.train_batch_size'], s['actor_rollout_ref.rollout.n']), (512, 4))
            self.assertEqual(s['actor_rollout_ref.actor.ppo_mini_batch_size'], 128)
            self.assertEqual(s['actor_rollout_ref.actor.optim.lr'], 1e-6)
            self.assertEqual(s['actor_rollout_ref.actor.entropy_coeff'], 0.001)
            self.assertEqual(s['algorithm.kl_ctrl.kl_coef'], 0.001)
            self.assertFalse(s['actor_rollout_ref.actor.use_kl_loss'])
            self.assertEqual((s['data.max_prompt_length'], s['data.max_response_length']), (2048, 1024))
            self.assertEqual((s['trainer.total_training_steps'], s['trainer.n_gpus_per_node']), (100, 4))
            self.assertEqual(env['SEED'], '4')
            self.assertEqual(cmd[1:4], ['-u', '-m', 'verl.trainer.main_ppo'])
            self.assertEqual(any(k.startswith('algorithm.mrtpo.') for k in s), method == 'mrtpo')

    def test_mrtpo_defaults_and_bound_flag(self):
        s, _, _ = settings('mrtpo')
        self.assertEqual((s['algorithm.mrtpo.beta'], s['algorithm.mrtpo.ramp_steps']), (0.5, 10))
        self.assertTrue(s['algorithm.mrtpo.reallocation_bound'])
        self.assertEqual(s['algorithm.mrtpo.reallocation_ratio'], 1.0)
        s, _, _ = settings('mrtpo', no_bound=True)
        self.assertFalse(s['algorithm.mrtpo.reallocation_bound'])

    def test_3b_allocator(self):
        self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', settings('gdpo')[1])
        self.assertEqual(settings('gdpo', '3b')[1]['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')

    def test_dry_run(self):
        out = subprocess.run([sys.executable, str(REPO / 'training/launch.py'), '--method', 'mrtpo', '--seed', '0',
                              '--model-size', '3b', '--model', 'Qwen/Qwen2.5-3B-Instruct', '--output', '/tmp/x',
                              '--dry-run'], capture_output=True, text=True, check=True)
        record = json.loads(out.stdout)
        self.assertEqual(record['settings']['algorithm.adv_estimator'], 'mrtpo')


if __name__ == '__main__':
    unittest.main()
