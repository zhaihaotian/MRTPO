"""Launch one Text-to-SQL training run (Spider train, Qwen2.5-Coder-3B/7B-Instruct) on one 8-GPU node."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
STACK = HERE.parent / 'verl_mrtpo'
REWARD = STACK / 'mrtpo' / 'reward' / 'sql_reward.py'
METHODS = ('grpo', 'gdpo', 'mrtpo', 'opsd', 'gdpo_saw', 'mrtpo_saw')
STEPS, SAVE_FREQ, GPUS = 100, 20, 8


def configuration(args):
    estimator = args.method.removesuffix('_saw')
    if estimator != 'mrtpo' and (args.beta != 0.5 or args.no_reallocation_bound):
        raise ValueError('--beta and --no-reallocation-bound are MRTPO options')
    data = args.data_dir.resolve()
    output = args.output.resolve()
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    train = data / 'parquet' / ('train_opsd.parquet' if estimator == 'opsd' else 'train.parquet')
    settings = {
        'algorithm.adv_estimator': estimator,
        'algorithm.norm_adv_by_std_in_grpo': True,
        'algorithm.use_kl_in_reward': False,
        'algorithm.batch_norm': 'response',
        'algorithm.saw': args.method.endswith('_saw'),
        'algorithm.mrtpo.beta': args.beta,
        'algorithm.mrtpo.ramp_steps': 10,
        'algorithm.mrtpo.reallocation_bound': not args.no_reallocation_bound,
        'algorithm.mrtpo.reallocation_ratio': 1.0,
        'algorithm.mrtpo.js_chunk_size': 128,
        'data.train_files': str(train),
        'data.val_files': str(train),
        'data.train_batch_size': 32,
        'data.val_batch_size': 32,
        'data.max_prompt_length': 4096,
        'data.max_response_length': 2048,
        'data.shuffle': True,
        '+data.seed': args.seed,
        'data.filter_overlong_prompts': True,
        'data.truncation': 'error',
        'data.dataloader_num_workers': 4,
        'custom_reward_function.path': str(REWARD),
        'custom_reward_function.name': 'compute_score',
        'reward_model.reward_manager': 'naive',
        'actor_rollout_ref.model.path': model,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.use_fused_kernels': False,
        'actor_rollout_ref.actor.strategy': 'fsdp',
        '+actor_rollout_ref.actor.seed': args.seed,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.optim.lr_warmup_steps': 15,
        'actor_rollout_ref.actor.optim.warmup_style': 'constant',
        'actor_rollout_ref.actor.optim.weight_decay': 0.1,
        'actor_rollout_ref.actor.grad_clip': 1.0,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 32,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.clip_ratio_low': 0.2,
        'actor_rollout_ref.actor.clip_ratio_high': 0.2,
        'actor_rollout_ref.actor.entropy_coeff': 0,
        'actor_rollout_ref.actor.use_kl_loss': estimator != 'opsd',
        'actor_rollout_ref.actor.kl_loss_coef': 0.001,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
        'actor_rollout_ref.actor.global_token_loss': True,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': 7168,
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size': 1,
        'actor_rollout_ref.actor.entropy_from_logits_with_chunking': True,
        'actor_rollout_ref.actor.checkpoint.save_contents': ['model', 'optimizer', 'extra', 'hf_model'],
        'actor_rollout_ref.actor.checkpoint.load_contents': ['model', 'optimizer', 'extra'],
        'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': 12288 if estimator == 'opsd' else 7168,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.mode': 'sync',
        'actor_rollout_ref.rollout.dtype': 'bfloat16',
        'actor_rollout_ref.rollout.calculate_log_probs': True,
        'actor_rollout_ref.rollout.n': 8,
        'actor_rollout_ref.rollout.temperature': 1.1,
        'actor_rollout_ref.rollout.top_p': 1.0,
        'actor_rollout_ref.rollout.top_k': -1,
        '+actor_rollout_ref.rollout.seed': args.seed,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.5,
        'actor_rollout_ref.rollout.max_model_len': 6144,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 6144,
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs': 128,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': 7168,
        'trainer.total_training_steps': STEPS,
        'trainer.n_gpus_per_node': GPUS,
        'trainer.nnodes': 1,
        'trainer.seed': args.seed,
        'trainer.test_freq': -1,
        'trainer.val_before_train': False,
        'trainer.save_freq': SAVE_FREQ,
        'trainer.resume_mode': 'disable',
        'trainer.max_actor_ckpt_to_keep': 1,
        'trainer.project_name': 'spider_sql',
        'trainer.experiment_name': output.name,
        'trainer.default_local_dir': str(output),
        'trainer.hf_export_dir': str(output / 'hf'),
        'trainer.rollout_data_dir': str(output / 'rollouts'),
        'trainer.logger': ['console', 'jsonl'],
        'ray_init.num_cpus': 32,
    }
    if estimator == 'opsd':
        settings.update({
            'algorithm.opsd.topk': 64,
            'algorithm.opsd.token_clip': 2.0,
            'algorithm.opsd.emitted_think_tokens': True,
            'algorithm.opsd.teacher_max_prompt_length': 7168,
            'algorithm.opsd.teacher_template': 'reference_field_turn_repeat',
        })
    environment = {'SQL_DB_ROOT': str(data / 'raw'), 'TOKENIZERS_PARALLELISM': 'false',
                   'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'PYTHONNOUSERSITE': '1'}
    command = [sys.executable, '-u', '-m', 'verl.trainer.main_ppo']
    command += [f'{key}={json.dumps(value, separators=(",", ":"))}' for key, value in settings.items()]
    return settings, environment, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, required=True, help='42, 1 or 2 in the paper')
    parser.add_argument('--model', required=True, help='Qwen2.5-Coder-3B-Instruct or Qwen2.5-Coder-7B-Instruct path or id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data', help='data root with parquet/ and raw/')
    parser.add_argument('--beta', type=float, default=0.5, help='MRTPO beta')
    parser.add_argument('--no-reallocation-bound', action='store_true', help='MRTPO without the reallocation bound')
    parser.add_argument('--dry-run', action='store_true', help='print the resolved settings and exit')
    args = parser.parse_args()
    settings, environment, command = configuration(args)
    record = dict(method=args.method, seed=args.seed, steps=STEPS, save_freq=SAVE_FREQ, gpus=GPUS,
                  settings=settings, environment=environment, command=command)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    for path in (Path(settings['data.train_files']), Path(environment['SQL_DB_ROOT']) / 'spider_data/database'):
        if not path.exists():
            raise FileNotFoundError(f'{path} (run the data preparation in text_to_sql/README.md first)')
    if (args.output / 'metrics.jsonl').exists():
        raise FileExistsError(f'Existing training results: {args.output}; choose a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    for key in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'RAY_ADDRESS'):
        env.pop(key, None)
    env.update(environment)
    with open(args.output / 'train.log', 'ab') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            log.write(line)
        code = process.wait()
    (args.output / 'exit.code').write_text(f'{code}\n')
    sys.exit(code)


if __name__ == '__main__':
    main()
