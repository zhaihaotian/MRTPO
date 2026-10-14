"""Launch one tool-calling run (GRPO, GDPO or MRTPO) on a single 4-GPU node."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FRAMEWORK = REPO / 'vendor' / 'verl'
METHODS = ('grpo', 'gdpo', 'mrtpo')
GPUS = 4


def configuration(args):
    output = Path(args.output).resolve()
    data = Path(args.data_dir).resolve()
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    settings = {
        'algorithm.adv_estimator': args.method,
        'data.train_files': str(data / 'train.parquet'),
        'data.val_files': str(data / 'test.parquet'),
        'data.train_batch_size': 512,
        'data.val_batch_size': 128,
        'data.max_prompt_length': 2048,
        'data.max_response_length': 1024,
        'actor_rollout_ref.model.path': model,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 128,
        'actor_rollout_ref.actor.ppo_micro_batch_size': 64,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 16384,
        'actor_rollout_ref.actor.clip_ratio': 0.2,
        'actor_rollout_ref.actor.entropy_coeff': 0.001,
        'actor_rollout_ref.actor.grad_clip': 1.0,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.grad_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.seed': 0,
        'actor_rollout_ref.rollout.n': 4,
        'actor_rollout_ref.rollout.temperature': 1.0,
        'actor_rollout_ref.rollout.top_p': 1,
        'actor_rollout_ref.rollout.top_k': -1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.6,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'algorithm.kl_penalty': 'kl',
        'algorithm.kl_ctrl.kl_coef': 0.001,
        'trainer.logger': ['console', 'jsonl'],
        'trainer.project_name': 'mrtpo-tool-calling',
        'trainer.experiment_name': f'qwen2.5-{args.model_size}-instruct-{args.method}-s{args.seed}',
        'trainer.n_gpus_per_node': GPUS,
        'trainer.nnodes': 1,
        'trainer.save_freq': args.save_freq,
        'trainer.test_freq': 10,
        'trainer.default_local_dir': str(output),
        'trainer.total_training_steps': 100,
        'trainer.total_epochs': 15,
    }
    if args.method == 'mrtpo':
        settings.update({
            'algorithm.mrtpo.beta': 0.5,
            'algorithm.mrtpo.ramp_steps': 10,
            'algorithm.mrtpo.reallocation_bound': not args.no_reallocation_bound,
            'algorithm.mrtpo.reallocation_ratio': 1.0,
            'algorithm.mrtpo.replay_micro_batch_size': 16,
            'algorithm.mrtpo.js_vocab_chunk_size': 128,
        })
    environment = dict(SEED=str(args.seed), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', VLLM_ATTENTION_BACKEND='XFORMERS',
                       RAY_USAGE_STATS_ENABLED='0', RAY_DISABLE_DOCKER_CPU_WARNING='1')
    environment['RAY_TMPDIR'] = f'/tmp/mrtpo{os.getuid()}p{os.getpid()}'
    if args.model_size == '3b':
        environment['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    command = [sys.executable, '-u', '-m', 'verl.trainer.main_ppo']
    command += [f'{key}={json.dumps(value, separators=(",", ":"))}' for key, value in settings.items()]
    return settings, environment, command


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--no-reallocation-bound', action='store_true',
                   help='MRTPO only: switch off the reallocation bound (on by default)')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--model', required=True, help='Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct path or model id')
    p.add_argument('--model-size', choices=['1.5b', '3b'], required=True)
    p.add_argument('--data-dir', type=Path, default=REPO / 'data' / 'rlla_4k')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-freq', type=int, choices=[10, 100], default=100,
                   help='Save a Hugging Face model every N training steps')
    p.add_argument('--dry-run', action='store_true', help='Print the resolved settings and exit')
    return p


def main():
    args = parser().parse_args()
    if args.no_reallocation_bound and args.method != 'mrtpo':
        raise SystemExit('--no-reallocation-bound applies to --method mrtpo only')
    settings, environment, command = configuration(args)
    record = dict(method=args.method, seed=args.seed, model_size=args.model_size, gpus=GPUS,
                  settings=settings, environment=environment, command=command)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    for name in ('train.parquet', 'test.parquet'):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(f'{args.data_dir / name}; run python scripts/fetch_data.py first')
    if (args.output / 'metrics.jsonl').exists():
        raise FileExistsError(f'Existing training results: {args.output}; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('RAY_ADDRESS', None)
    env.update(environment)
    env['PYTHONPATH'] = str(FRAMEWORK) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run(command, cwd=FRAMEWORK, env=env)
    (args.output / 'exit.code').write_text(str(result.returncode) + '\n')
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
