# MRTPO on Tool Calling

This folder reproduces the tool-calling experiments of the paper. The model learns to call tools from a dialogue and a
list of tool descriptions, answering in this format:

```
<think> reasoning </think>
<tool_call>
{"name": ..., "parameters": {...}}
</tool_call>
```

(or `<think> … </think>\n<response> … </response>` when no tool is needed). Training uses the two ToolRL / GDPO rewards:

- **Format**: 0 or 1, the response has exactly the tag structure of the label.
- **Correctness**: in [−3, 3], tool names, parameter names and parameter values compared with the reference calls.

Example: the label is
`{"name": "book_room", "parameters": {"capacity": "8", "city": "Oslo"}}`. A response
`<think> … </think>\n<tool_call>\n{"name": "book_room", "parameters": {"capacity": "8", "city": "Oslo"}}\n</tool_call>`
gets format 1 and correctness 3. Writing `"capacity": 8` instead gets correctness 1.5: the score is
(name match 1 + parameter-name match 1 + equal values 1) / (1 + 1 + 2 parameters) = 3/4 of the range, and values are
compared exactly.

MRTPO keeps each reward's GDPO group advantage per response and spreads it over the response tokens. For every reward it
replays the sampled response under the prompt with and without the information that reward depends on (format block,
reference calls), measures per token how much the next-token distribution moves (Jensen–Shannon distance), and gives
tokens that depend more on that information a larger share. For the correctness reward, the reference calls shown in the
replay are the label with numeric and boolean values written in the types the tool schema declares
(`extra_info.teacher_output`, built by `scripts/fetch_data.py`); the reward itself always scores the original label.
See [`docs/ALGORITHMS.md`](./docs/ALGORITHMS.md).

The code is the verl implementation released with [GDPO](https://github.com/NVlabs/GDPO), vendored in
[`vendor/verl`](./vendor/verl), with GRPO, GDPO and MRTPO as advantage estimators. All methods share the same launcher
and configuration.

## Installation

Three conda environments, because training and the BFCL evaluation need different versions of vLLM and PyTorch. All
versions are pinned in [`environments/`](./environments).

```bash
bash environments/install_training.sh mrtpo-tc-train         # training: Python 3.10, PyTorch 2.4.0, vLLM 0.6.3
bash environments/install_inference.sh mrtpo-tc-inference    # vLLM 0.11.0 server used during evaluation
bash environments/install_evaluation.sh mrtpo-tc-bfcl        # BFCL V4 (gorilla @ 6ea57973)
```

The training installer finishes by running the CPU tests (`bash scripts/test.sh`).

## Data

```bash
conda activate mrtpo-tc-train
python scripts/fetch_data.py
```

This downloads ToolRL's `rlla_4k` split at commit `8cee13e` (3,920 training and 80 validation rows; sha256 checked:
train `f2e728ad…e9b1c8`, test `f7f300eb…da7554`) and writes `data/rlla_4k/{train,test}.parquet`: all original
columns unchanged, plus `extra_info.teacher_output`. In `teacher_output`, every `int`, `float` or `boolean` argument
value that the label writes as a plain-literal string (`"8"`, `"2.5"`, `"true"`) is converted to the declared type. This
changes 393 of 3,920 training rows (419 int, 471 float and 40 boolean values; 6 non-literal strings stay as written) and
10 of 80 validation rows. Prompts, labels and the reward are unchanged.

Models: `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`; download them anywhere and pass the path.

## Training

One node with 4 GPUs (A100 40GB or larger) per run; 100 steps of 512 prompts × 4 responses. The final model is saved to
`<output>/actor/global_step_100`.

```bash
conda activate mrtpo-tc-train
python training/launch.py --method mrtpo --seed 0 --model-size 1.5b \
  --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-mrtpo-s0
```

`--method` is `grpo`, `gdpo` or `mrtpo`. MRTPO uses β = 0.5, warm-up K = 10 steps and the reallocation bound
(ρ = 1); `--no-reallocation-bound` switches the bound off. Add `--dry-run` to print every resolved setting, and
`--save-freq 10` to also keep the models at steps 10, 20, …, 90.

Seeds: 0, 1, 2, 4, 5 for Qwen2.5-1.5B-Instruct and 0, 1, 2 for Qwen2.5-3B-Instruct.

```bash
for m in grpo gdpo mrtpo; do
  for s in 0 1 2 4 5; do
    python training/launch.py --method $m --seed $s --model-size 1.5b \
      --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-$m-s$s
  done
  for s in 0 1 2; do
    python training/launch.py --method $m --seed $s --model-size 3b \
      --model /models/Qwen2.5-3B-Instruct --output outputs/3b-$m-s$s
  done
done
```

Each run writes `metrics.jsonl` (one line per step): per-reward training scores `critic/{format,correctness}_score/mean`,
validation scores `val/test_*/rlla` every 10 steps, and for MRTPO `mrtpo/*` (mixture strength, reallocation scale, share
of responses replayed). All settings are listed in [`docs/TRAINING.md`](./docs/TRAINING.md).

## Evaluation

BFCL V4 single-turn AST: Non-Live (Simple Python / Java / JavaScript, Multiple, Parallel, Parallel Multiple; 1,150 cases)
and Live (Simple, Multiple, Parallel, Parallel Multiple; 1,351 cases), 2,501 cases in total, one GPU per checkpoint. The
script starts a vLLM server from the inference environment, runs every case through the ToolRL prompt and parser, and
scores it with the official BFCL checkers.

```bash
conda activate mrtpo-tc-bfcl
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --model outputs/1p5b-mrtpo-s0/actor/global_step_100 \
  --server-python /path/to/envs/mrtpo-tc-inference/bin/python \
  --port 8000 --output eval/1p5b-mrtpo-s0
```

Decoding: temperature 0.6, top-p 0.95, seed 0, at most 8,192 new tokens, 32,768-token context. An interrupted run is
resumed by repeating the command. `python evaluation/score.py --output DIR` rescores saved outputs on CPU. Metrics are the
official accuracies (no type coercion), in `summary.json` under `accuracy`: Non-Live = mean of Simple (mean of the three
languages), Multiple, Parallel and Parallel Multiple; Live = case-weighted mean of the four Live categories. The format
rate (share of outputs with the exact RLLA tag structure) is stored under `format` with the same grouping. The paper
table is mean ± sample SD over seeds (`--metric format` for the format rate):

```bash
python evaluation/aggregate.py --row 'MRTPO 1.5B' eval/1p5b-mrtpo-s{0,1,2,4,5} --row 'GDPO 1.5B' eval/1p5b-gdpo-s{0,1,2,4,5}
```

Details: [`docs/EVALUATION.md`](./docs/EVALUATION.md).

## Code map

| What | Where |
|---|---|
| Rewards (also the BFCL format rate) | `vendor/verl/verl/utils/reward_score/rlla.py`, `evaluation/format_rate.py` |
| Replay contexts (P_0, P_F, P_C, P_FC) | `vendor/verl/verl/utils/rlla_replay.py` |
| Which responses are replayed, replay batch | `vendor/verl/verl/trainer/ppo/ray_trainer.py::RayPPOTrainer._mrtpo_replay` |
| Replay forward passes and JS | `vendor/verl/verl/trainer/ppo/mrtpo.py::replay_js`, called by `workers/actor/dp_actor.py::compute_replay_js` |
| Token weights, reallocation bound | `vendor/verl/verl/trainer/ppo/mrtpo.py::token_weights`, `reallocation_bound`, `combine` |
| GRPO / GDPO / MRTPO advantages | `vendor/verl/verl/trainer/ppo/ray_trainer.py::compute_advantage` |
| Settings | `training/launch.py`, `vendor/verl/verl/trainer/config/ppo_trainer.yaml` (`algorithm.mrtpo.*`) |
