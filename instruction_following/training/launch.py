"""Launch one instruction-following run (HIR-16K, 8 GPUs) with GRPO, GDPO, MRTPO or CoRT."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
STACK = REPO / 'verl_mrtpo'
DATA = REPO / 'instruction_following' / 'data' / 'hir_if'
METHODS = ('grpo', 'gdpo', 'mrtpo', 'cort')


def configuration(args):
    output = args.output.resolve()
    parquet = str((args.data_dir / 'train.parquet').resolve())
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    settings = {
        'algorithm.adv_estimator': args.method,
        'algorithm.norm_adv_by_std_in_grpo': True,
        'algorithm.use_kl_in_reward': False,
        'data.train_files': parquet,
        'data.val_files': parquet,
        'data.train_batch_size': 64,
        'data.val_batch_size': 64,
        'data.max_prompt_length': 2048,
        'data.max_response_length': 4096,
        'data.shuffle': True,
        '++data.seed': args.seed,
        'data.filter_overlong_prompts': True,
        'data.truncation': 'error',
        'data.dataloader_num_workers': 4,
        'custom_reward_function.path': str(STACK / 'mrtpo' / 'reward' / 'hir_reward.py'),
        'custom_reward_function.name': 'compute_score',
        'reward_model.reward_manager': 'naive',
        'actor_rollout_ref.model.path': model,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.use_fused_kernels': False,
        'actor_rollout_ref.actor.strategy': 'fsdp',
        '++actor_rollout_ref.actor.seed': args.seed,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.optim.lr_warmup_steps': 15,
        'actor_rollout_ref.actor.optim.warmup_style': 'constant',
        'actor_rollout_ref.actor.optim.weight_decay': 0.1,
        'actor_rollout_ref.actor.grad_clip': 1.0,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 64,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.clip_ratio_low': 0.2,
        'actor_rollout_ref.actor.clip_ratio_high': 0.27,
        'actor_rollout_ref.actor.entropy_coeff': 0,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
        'actor_rollout_ref.actor.global_token_loss': True,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 16384,
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size': 1,
        'actor_rollout_ref.actor.entropy_from_logits_with_chunking': True,
        'actor_rollout_ref.actor.checkpoint.save_contents': ['model', 'optimizer', 'extra', 'hf_model'],
        'actor_rollout_ref.actor.checkpoint.load_contents': ['model', 'optimizer', 'extra'],
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.mode': 'sync',
        'actor_rollout_ref.rollout.dtype': 'bfloat16',
        'actor_rollout_ref.rollout.calculate_log_probs': True,
        'actor_rollout_ref.rollout.n': 8,
        'actor_rollout_ref.rollout.temperature': 1.0,
        'actor_rollout_ref.rollout.top_p': 0.99,
        'actor_rollout_ref.rollout.top_k': 100,
        '++actor_rollout_ref.rollout.seed': args.seed,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.5,
        'actor_rollout_ref.rollout.max_model_len': 6144,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 16384,
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs': 128,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 16384,
        'trainer.total_training_steps': args.steps,
        'trainer.n_gpus_per_node': 8,
        'trainer.nnodes': 1,
        'trainer.seed': args.seed,
        'trainer.test_freq': -1,
        'trainer.val_before_train': False,
        'trainer.save_freq': args.save_freq,
        'trainer.resume_mode': 'auto',
        'trainer.hf_export_dir': str(output / 'hf'),
        'trainer.max_actor_ckpt_to_keep': 1,
        'trainer.project_name': 'instruction_following',
        'trainer.experiment_name': output.name,
        'trainer.default_local_dir': str(output),
        'trainer.logger': ['console', 'jsonl'],
        'ray_init.num_cpus': 32,
    }
    if args.method == 'mrtpo':
        settings.update({
            'algorithm.mrtpo.beta': args.beta,
            'algorithm.mrtpo.ramp_steps': 10,
            'algorithm.mrtpo.reallocation_bound': not args.no_reallocation_bound,
            'algorithm.mrtpo.reallocation_ratio': 1.0,
            'algorithm.mrtpo.js_chunk_size': 128,
        })
    elif args.beta != 0.5 or args.no_reallocation_bound:
        raise ValueError('--beta and --no-reallocation-bound apply to --method mrtpo')
    if args.method == 'cort':
        settings.update({'algorithm.cort.eta': 0.5, 'algorithm.cort.tau': 1.0,
                         'algorithm.cort.center': 0.0, 'algorithm.cort.ramp_steps': 100})
    environment = dict(TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       RAY_USAGE_STATS_ENABLED='0')
    command = [sys.executable, '-u', '-m', 'verl.trainer.main_ppo']
    command += [f'{key}={json.dumps(value, separators=(",", ":"))}' for key, value in settings.items()]
    return settings, environment, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model', required=True, help='Qwen2.5-7B-Instruct or Qwen3-4B-Instruct-2507 path or model ID')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--save-freq', type=int, default=50, help='Save (and export a bf16 HF model) every N steps')
    parser.add_argument('--beta', type=float, default=0.5, help='MRTPO beta')
    parser.add_argument('--no-reallocation-bound', action='store_true', help='MRTPO without the reallocation bound')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved settings and command, then exit')
    args = parser.parse_args()
    settings, environment, command = configuration(args)
    record = dict(method=args.method, seed=args.seed, model=args.model, steps=args.steps, gpus=8,
                  prompts_per_step=64, rollouts_per_prompt=8, settings=settings, environment=environment,
                  command=command)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    if not (args.data_dir / 'train.parquet').is_file():
        raise FileNotFoundError(args.data_dir / 'train.parquet')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    for key in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'RAY_ADDRESS'):
        env.pop(key, None)
    env.update(environment)
    with open(args.output / 'train.log', 'ab') as log:
        process = subprocess.Popen(command, cwd=args.output, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    (args.output / 'exit.code').write_text(f'{returncode}\n')
    sys.exit(returncode)


if __name__ == '__main__':
    main()
