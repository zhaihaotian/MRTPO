# Training configuration

All methods use the configuration below; only `--method` differs. The data are the ToolRL `rlla_4k` parquet files
written by `scripts/fetch_data.py` (original prompts, labels, order and split, plus `extra_info.teacher_output`):
3,920 training and 80 validation rows.

| Parameter | Value |
|---|---|
| Backbone | Qwen2.5-1.5B-Instruct / Qwen2.5-3B-Instruct |
| Hardware per run | 1 node, 4 × NVIDIA A100 40GB (or larger) |
| Training / rollout | FSDP / vLLM 0.6.3 (xFormers attention backend), tensor parallel 1, bf16, eager mode |
| Python / PyTorch / Transformers / Ray | 3.10.21 / 2.4.0+cu121 / 4.47.1 / 2.10.0 |
| FlashAttention / xFormers | 2.6.3 / 0.0.27.post2 |
| Training steps | 100 policy updates (one per step) |
| Validation | before training and every 10 steps (80 RLLA rows, greedy) |
| Evaluated checkpoint | step 100 (Hugging Face actor and tokenizer) |
| Prompts × responses per step | 512 × 4 = 2,048 responses |
| PPO mini-batch / micro-batch | 128 / 64 prompts (4 optimizer updates per step) |
| PPO epochs / clip ratio | 1 / 0.2 |
| Optimizer | AdamW, learning rate 1e-6 constant (no warm-up), weight decay 0.01, gradient clipping 1.0 |
| Entropy coefficient | 0.001 |
| KL | reward-KL penalty with coefficient 0.001 (`kl` estimator); no actor KL loss |
| Rollout temperature / top-p / top-k | 1.0 / 1.0 / −1; vLLM seed 0 |
| Prompt / response limit | 2,048 / 1,024 tokens (prompts truncated from the left) |
| Dynamic batching | on; 16,384 tokens per GPU |
| Rollout GPU memory utilization | 0.6 |
| Actor param / grad / optimizer offload | off / off / off |
| Reference param offload | on |
| Gradient checkpointing / padding removal | on / on |
| Data order / torch RNG | shuffled by the `SEED` environment variable (= `--seed`) |
| Seeds | Qwen2.5-1.5B-Instruct: 0, 1, 2, 4, 5; Qwen2.5-3B-Instruct: 0, 1, 2 |

MRTPO (`algorithm.mrtpo.*`):

| Parameter | Value |
|---|---|
| Mixture strength β | 0.5 for both rewards |
| Warm-up | b = β · smoothstep(step / K), K = 10 steps (3p² − 2p³) |
| Token score | q = sqrt(JS / ln 2), JS over the full vocabulary at temperature 1 |
| Reallocation bound | on, ρ = 1 (`--no-reallocation-bound` turns it off) |
| Weight cap | none |
| Replay micro-batch | 16 responses per forward pass; JS computed in chunks of 128 tokens |

GRPO takes the group z-score of the summed reward after the reward-KL penalty. GDPO and MRTPO read the two rewards
separately, without the KL penalty. The policy loss is the token mean of each micro-batch divided by the nominal
accumulation factor 2 (mini-batch / micro-batch). For 3B runs the launcher sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. On 4 × A100 40GB a 1.5B GRPO or GDPO run takes about 2.5 hours and a
3B run 4–5 hours; MRTPO adds the replay forward passes (at most three per response and step).

## Rewards

`vendor/verl/verl/utils/reward_score/rlla.py`, applied to the assistant turn of the decoded sequence:

| Reward | Range | Definition |
|---|---|---|
| format | {0, 1} | 1 iff the response matches the structure the label uses: `<think>…</think>\n<tool_call>\n…\n</tool_call>`, `<think>…</think>\n<response>…</response>`, or both blocks; one block of each kind |
| correctness | [−3, 3] | 0 for labels without `<tool_call>`. Otherwise −3 if the response has no parsable `<tool_call>` block; 3 if the predicted calls equal the reference; else −3 + 6 · S / M with S = name match (intersection over union of the call-name multisets) + for each reference call the best unused predicted call with the same name (parameter-name match + one point per exactly equal value) and M = 1 + Σ (1 + number of reference parameters) |

The total reward is format + correctness. Both are scored against the original label (`reward_model.ground_truth`).

## Outputs

Each run directory contains `launch.json` (arguments, resolved settings, environment and command), the full Hydra
`config.json`, `metrics.jsonl` (one record per step), `exit.code` and `actor/global_step_100`. With `--save-freq 10`,
models are also written at steps 10, 20, …, 90. These are Hugging Face weights for evaluation; optimizer and data-loader
state are not saved, so an interrupted run is restarted in a new output directory with the same seed.
