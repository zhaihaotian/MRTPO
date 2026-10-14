# MRTPO on Instruction Following

This folder reproduces the instruction-following experiments in the paper. Each training prompt is a question with
several verifiable constraints, for example (one of the 6,998 training prompts, six constraints):

```
What are the key features of a well-designed website?

- There should be 2 paragraphs. Paragraphs are separated with the markdown divider: ***
- Your response should contain at least 3 bullet points. Use the markdown bullet points such as: * This is point 1.
- Highlight at least 2 sections in your answer with markdown, i.e. *highlighted section*.
- In your response, the word *design* should appear at least 2 times.
- Do not include keywords *layout* in the response.
- Finish your response with this exact phrase *Is there anything else I can help with?* No other words should follow this phrase.
```

Each constraint is one reward: 0 or 1 from the official strict checker of IFEval (`source=type1` prompts) or IFBench
(`type2`). A response to the prompt above gets a vector of six scores, e.g. `[1, 1, 0, 1, 1, 1]`.
- **GRPO** uses the mean of the vector as a single reward.
- **GDPO** normalizes each constraint's score within the prompt's 8 rollouts, sums the six advantages and divides by
  the batch standard deviation of that sum.
- **MRTPO** starts from the same per-constraint advantages and moves each one between the response's tokens: the policy
  scores the same response twice, with the full prompt and with that constraint left out, and tokens whose
  next-token distribution changes more (Jensen–Shannon distance) receive more of that constraint's credit. Each
  response keeps its GDPO total.
- **CoRT** keeps the GRPO advantage and reweights tokens by the log-probability contrast between the full prompt and the
  question without any constraint.

The training code is the shared stack in [`../verl_mrtpo`](../verl_mrtpo) (verl 0.5.0 with the `mrtpo` package).

## Installation

```bash
bash ../verl_mrtpo/environments/install.sh mrtpo          # training and IFEval / IFBench / MulDimIF evaluation
bash environments/install_judge.sh mrtpo-judge             # vLLM 0.19.0, only for serving the AdvancedIF judge
```

The first environment is Python 3.11 with PyTorch 2.6.0 (CUDA 12.4), vLLM 0.8.5, transformers 4.53.2,
flash-attn 2.7.4.post1 and Ray 2.47.1; every package is pinned in
[`../verl_mrtpo/environments/requirements.lock.txt`](../verl_mrtpo/environments/requirements.lock.txt). The installer also
downloads the NLTK data the checkers use and runs the CPU tests. The judge environment is pinned in
[`environments/judge.lock.txt`](environments/judge.lock.txt) (Python 3.10, PyTorch 2.10, CUDA 12.8).

## Data

```bash
conda activate mrtpo
python data/prepare_hir.py
```

This downloads `HIR_trainv1.jsonl` from [`sastpg/HIR-16K`](https://huggingface.co/datasets/sastpg/HIR-16K) (revision
`2a95f69e`, 16,968 rows, sha256 `465a01c1…183b`) into `data/raw/` and writes `data/hir_if/train.parquet`:
- keeps the 6,549 `type1` and 449 `type2` rows (IFEval and IFBench checkers) and drops the other 9,970;
- joins question and constraints with one of HIR's six layouts (inline, newline, numbered rules, bullets,
  constraints first, one per line), chosen per row with `random.Random(20260927 + row_index)`;
- drops prompts longer than 2,048 chat-template tokens (Qwen3-4B-Instruct-2507 and Qwen2.5-7B-Instruct tokenizers; none
  is dropped);
- stores, for every constraint, the same prompt with that constraint left out (`extra_info.minus_prompts_json`), which
  MRTPO replays.

Result: 6,998 prompts with 43,496 constraints (5 / 6 / 7 / 8 / ≥9 constraints per prompt: 1,479 / 3,507 / 1,386 / 420 /
206). The script checks a sha256 over the rows (`d2ffa5d5…62dd`) and stops if the training set differs from the paper's.
The prompt is one user message under the model's chat template (Qwen2.5 adds its default system prompt).

Models: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (revision `a09a3545`) and
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (revision `cdbee75f`). Download them
wherever you like and pass the path.

## Training

Each run uses one node with 8 H200 GPUs and trains for 300 steps. A bf16 Hugging Face model is
exported every 50 steps to `<output>/hf/global_step_<N>`; the paper reports `global_step_300`.

```bash
conda activate mrtpo
python training/launch.py --method mrtpo --seed 42 \
  --model /models/Qwen2.5-7B-Instruct --output runs/qwen25-7b-mrtpo-s42
```

`--method` is `grpo`, `gdpo`, `mrtpo` or `cort`. Add `--dry-run` to print the resolved settings and the full trainer
command without training. The main comparison uses seeds 42–46 for both models:

```bash
for model in Qwen2.5-7B-Instruct Qwen3-4B-Instruct-2507; do
  for m in grpo gdpo cort mrtpo; do
    for s in 42 43 44 45 46; do
      python training/launch.py --method $m --seed $s --model /models/$model --output runs/$model-$m-s$s
    done
  done
done
```

Settings shared by all methods:
- 64 prompts × 8 rollouts per step; vLLM sampling with temperature 1.0, top-p 0.99, top-k 100; at most 2,048 prompt and
  4,096 response tokens.
- AdamW, learning rate 1e-6 with 15 warm-up steps and then constant, weight decay 0.1, gradient clipping 1.0.
- One PPO update per step (mini-batch = the whole step), clip range [0.2, 0.27], no entropy bonus, no KL term.
- Token-mean loss over all response tokens of the update (across GPUs and micro-batches).
- Seed `s` sets the data order, the rollout sampler and the trainer seed.

Method settings:
- **GDPO / MRTPO**: per-constraint group normalization `(r − mean) / (std + 1e-6)` over the 8 rollouts; the summed
  advantage is divided by its batch standard deviation (one value per response).
- **MRTPO**: β = 0.5, ramped in with a cubic smoothstep over the first K = 10 steps; token weight
  `w = (1 − β) + β·q / mean(q)` with `q = sqrt(JS / ln 2)`, JS between the full-prompt and leave-one-out next-token
  distributions (full vocabulary, fp32 in chunks of 128 tokens); the reallocation bound is on (ρ = 1): if the token
  residual (MRTPO advantage minus the GDPO advantage) has a larger second moment over the batch's tokens than the GDPO
  advantage, it is scaled down by one common factor; no cap on individual weights.
- **CoRT**: η = 0.5, τ = 1.0, center 0, 100-step smoothstep ramp; the replay prompt is the question without any
  constraint.

Every run writes `metrics.jsonl` (one line per step). Rewards are `reward/criterion_micro_pass_rate` and
`reward/prompt_all_pass_rate`; MRTPO logs `mrtpo/schedule`, `mrtpo/weight_{min,max,std}`, `mrtpo/js_mean`,
`mrtpo/replay_rows` and `mrtpo/residual_scale` (1 when the bound did not act); CoRT logs `cort/*`.

## Evaluation

**IFEval, IFBench, MulDimIF** (one GPU per checkpoint):

```bash
conda activate mrtpo
CUDA_VISIBLE_DEVICES=0 bash evaluation/run.sh runs/qwen25-7b-mrtpo-s42/hf/global_step_300 eval/qwen25-7b-mrtpo-s42
```

- Benchmarks (vendored in [`evaluation/data`](evaluation/data) from HIR's evaluation copy): IFEval 541 prompts,
  IFBench 294, MulDimIF 1,200.
- Decoding: vLLM bf16, the model's chat template with one user turn, 5 samples per prompt, temperature 0.7, top-p 0.8,
  top-k 20, at most 4,096 new tokens, seed 0, `max_model_len` 8,192.
- Metrics (computed on each of the 5 samples, then averaged): IFEval and IFBench **prompt-level strict** (share of
  prompts whose instructions all pass the official strict checker) and **instruction-level strict** (share of
  instructions that pass); MulDimIF **all-pass** (share of prompts whose constraints all pass). Results are in
  `metrics.json` (`ifeval.prompt_strict`, `ifeval.inst_strict`, `ifbench.prompt_strict`, `ifbench.inst_strict`,
  `muldimif.all_pass`).

**AdvancedIF** ([`facebook/AdvancedIF`](https://huggingface.co/datasets/facebook/AdvancedIF), 1,645 prompts in three
subsets, each prompt with human-written rubrics), judged by
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (thinking on) with the official judge code:

```bash
bash evaluation/advancedif/fetch.sh          # official judge code (commit f9d30137) and the data (revision e20cba9b)
conda activate mrtpo
bash evaluation/advancedif/run.sh runs/qwen25-7b-mrtpo-s42/hf/global_step_300 eval/qwen25-7b-mrtpo-s42/advancedif
JUDGE=/models/Qwen3.6-35B-A3B VLLM=$(conda run -n mrtpo-judge which vllm) \
  bash evaluation/advancedif/judge.sh eval/qwen25-7b-mrtpo-s42/advancedif 0,1,2,3 8300
```

- Generation uses all visible GPUs (one shard each) with the same decoding as above (5 samples, 8,225 responses).
- The judge runs as vLLM servers of two GPUs each (tensor parallel 2) with `--reasoning-parser qwen3`; the official
  client is called unchanged except for a thread pool sized to 512 concurrent requests, a 3,600 s timeout and the
  official output format passed as a JSON schema. Failed judge calls are re-judged with presence penalty 1.5 and up to
  16,384 / 32,768 / 49,152 completion tokens.
- Metrics (`summary.json`, `advancedif.overall`): **Overall** = share of responses for which all rubrics are satisfied
  (`all_rubrics_pass`); **Rubric** = share of all checked rubrics that are satisfied (`micro_rubric_pass`). A response
  whose judge call still fails counts as not passed.

We report the step-300 checkpoint, averaged over training seeds.

## Code map

- [`training/launch.py`](training/launch.py): every training setting.
- [`../verl_mrtpo/mrtpo/advantage.py`](../verl_mrtpo/mrtpo/advantage.py): GDPO advantages, MRTPO's token weights
  (`allocate_criterion_advantage`) and the reallocation bound (`bound_reallocation`).
- [`../verl_mrtpo/mrtpo/criteria.py`](../verl_mrtpo/mrtpo/criteria.py): per-constraint group normalization and the
  leave-one-out replay; [`../verl_mrtpo/mrtpo/js.py`](../verl_mrtpo/mrtpo/js.py): the JS divergence.
- [`../verl_mrtpo/verl/trainer/ppo/ray_trainer.py`](../verl_mrtpo/verl/trainer/ppo/ray_trainer.py)
  (`_compute_criterion_js`) and `verl/workers/actor/dp_actor.py` (`compute_criterion_js`): where the replay runs.
- [`../verl_mrtpo/mrtpo/cort.py`](../verl_mrtpo/mrtpo/cort.py): CoRT; [`../verl_mrtpo/mrtpo/reward/hir_reward.py`](../verl_mrtpo/mrtpo/reward/hir_reward.py):
  the per-constraint reward.
- [`evaluation/`](evaluation): generation, official checkers and scoring; [`evaluation/advancedif/`](evaluation/advancedif):
  AdvancedIF generation, judging and summary.
