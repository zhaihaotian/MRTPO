# verl_mrtpo: the training stack for instruction following and Text-to-SQL

This is [verl](https://github.com/volcengine/verl) 0.5.0 (`verl/`, Apache-2.0) with the `mrtpo` package (`mrtpo/`)
that adds the methods of the paper: GDPO, MRTPO (with the reallocation bound), SAW, CoRT and OPSD. GRPO is verl's own estimator. The launchers in
[`../instruction_following`](../instruction_following) and [`../text_to_sql`](../text_to_sql) call
`python -m verl.trainer.main_ppo` with all settings on the command line.

```bash
bash environments/install.sh mrtpo       # conda env from environments/requirements.lock.txt, then the CPU tests
```

## How a step runs

1. vLLM samples the rollouts; the reward function returns a score vector per response
   (`criteria_scores`, plus `replay_blocks_json` for Text-to-SQL).
2. `mrtpo.advantage.prepare_criteria` normalizes every criterion within its prompt's rollouts.
3. MRTPO only: `RayPPOTrainer._compute_criterion_js` builds the replay batch (`mrtpo.criteria`), and the actor scores
   each sampled response under the full context and under the context without one criterion
   (`DataParallelPPOActor.compute_criterion_js`, JS over the full vocabulary in `mrtpo.js`).
4. `compute_advantage` → `mrtpo.advantage.compute_criterion_advantage`: token weights
   `w = (1 − β) + β·q / mean(q)`, `q = sqrt(JS / ln 2)`, β ramped by a smoothstep over `ramp_steps`, the reallocation
   bound, then division by the batch standard deviation of the per-response advantages. GDPO is the same with `w = 1`.
5. The PPO update uses a token-mean loss over all tokens of the optimizer mini-batch (`global_token_loss`).

## Configuration (`verl/trainer/config/ppo_trainer.yaml`)

| key | default | meaning |
|---|---|---|
| `algorithm.adv_estimator` | | `grpo`, `gdpo`, `mrtpo`, `cort`, `opsd` |
| `algorithm.batch_norm` | `response` | GDPO / MRTPO normalization by the std of the per-response advantage |
| `algorithm.saw` | `false` | SAW: criterion k's advantage × n·CV_k / Σ CV (GDPO and MRTPO) |
| `algorithm.mrtpo.beta` | 0.5 | share of each criterion's credit placed by q |
| `algorithm.mrtpo.ramp_steps` | 10 | smoothstep ramp of β |
| `algorithm.mrtpo.reallocation_bound` | `true` | cap the token residual's second moment at ρ × that of the GDPO advantage |
| `algorithm.mrtpo.reallocation_ratio` | 1.0 | ρ |
| `algorithm.mrtpo.js_chunk_size` | 128 | tokens per fp32 JS chunk |
| `algorithm.cort.{eta,tau,center,ramp_steps}` | 0.5, 1.0, 0.0, 100 | CoRT weights `1 + η·λ·(sigmoid(τ(Δ − center)) − ½)`, normalized per response |
| `algorithm.opsd.{topk,token_clip}` | 64, 2.0 | teacher top-K forward KL, per-entry clip |
| `algorithm.opsd.teacher_template` | `reference_field_turn_repeat` | teacher context: prompt, reference solution, prompt |
| `algorithm.opsd.{emitted_think_tokens,teacher_max_prompt_length}` | `true`, 7168 | |
| `actor_rollout_ref.actor.global_token_loss` | `false` | token-mean over the whole update (the launchers set `true`) |
| `trainer.seed`, `trainer.hf_export_dir` | | process seed; bf16 HF export of every saved step |

Metrics go to `<trainer.default_local_dir>/metrics.jsonl` with `trainer.logger=[console,jsonl]`; method metrics use the
prefixes `mrtpo/`, `saw/`, `cort/`, `opsd/`.

## Code map

| file | content |
|---|---|
| `mrtpo/advantage.py` | GDPO / MRTPO advantages, token weights, reallocation bound, SAW |
| `mrtpo/criteria.py` | criterion scores, group normalization, leave-one-out replay |
| `mrtpo/js.py` | chunked full-vocabulary Jensen–Shannon divergence |
| `mrtpo/cort.py` | CoRT replay prompt and token weights |
| `mrtpo/opsd.py` | OPSD teacher prompts and top-K forward KL |
| `mrtpo/hir.py` | HIR's prompt layouts (used by the data builder and the criteria-free replay) |
| `mrtpo/reward/` | instruction-following reward (IFEval / IFBench checkers) and Text-to-SQL reward |
| `mrtpo/sql/` | SQL-R1 prompt, replay blocks, SQLite execution |
| `mrtpo/checkpoint.py` | bf16 HF export and data-order restore on resume |
| `verl/trainer/ppo/ray_trainer.py` | `compute_advantage` branches and the replay / teacher / CoRT passes |
| `verl/workers/actor/dp_actor.py`, `verl/workers/fsdp_workers.py` | replay JS, OPSD teacher top-K and loss, global token loss |

## Changes to upstream verl 0.5.0

Added for the paper's methods:
- `trainer/ppo/ray_trainer.py`: `compute_advantage` branches for `gdpo`, `mrtpo`, `cort`, `opsd`; the MRTPO replay,
  CoRT replay and OPSD teacher passes; criterion / advantage / truncation metrics; bf16 HF export of every saved step
  and data-order restore on resume (`mrtpo/checkpoint.py`).
- `workers/actor/dp_actor.py`: `compute_criterion_js` (replay JS), OPSD teacher top-K and loss, token-mean loss over
  the whole optimizer mini-batch (`global_token_loss`).
- `workers/fsdp_workers.py`: `compute_criterion_js` and `compute_ref_topk` worker methods, actor seed, timing
  reduction on the CPU group (`utils/profiler/performance.py`).
- `trainer/main_ppo.py`: reference worker for OPSD, `trainer.seed`; `config/ppo_trainer.yaml`,
  `config/actor/dp_actor.yaml`: the keys above; `utils/tracking.py`: the `jsonl` logger.

Removed, because the runs use only FSDP actors, synchronous vLLM rollouts and rule-based rewards: Megatron (workers,
models, checkpointing, dispatch modes), SGLang, async / agent-loop rollouts, HF and naive rollouts, critic and
reward-model workers, `tools/`, `interactions/`, `experimental/`, `model_merger/`, the SFT, generation and evaluation
entry points and their configs, verl's built-in dataset scorers and the batch / DAPO / PRIME reward managers, Ascend
NPU patches, the ModelScope hub patch and utilities nothing imports. `main_ppo.py` and `ray_trainer.py` raise
`NotImplementedError` for another strategy, `rollout.mode=async` or `reward_model.enable=True`;
`utils.reward_score.default_compute_score` raises, so `custom_reward_function.path` must be set (the launchers do).

## Tests

```bash
python -m pytest -q tests        # CPU only, no model weights
```
