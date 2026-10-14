# Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL

<p align="center">
🧩 <a href="./instruction_following">Instruction following</a> &nbsp;|&nbsp; 🗄️ <a href="./text_to_sql">Text-to-SQL</a> &nbsp;|&nbsp; 🛠️ <a href="./tool_calling">Tool calling</a>
</p>

This repository contains the implementation of **MRTPO**, a multi-reward reinforcement learning method that places each
reward's advantage on the tokens relevant to that reward. It includes training and evaluation for the three settings
of the paper: multi-constraint instruction following, text-to-SQL and tool calling.

## Method

GDPO normalizes every reward separately within its rollout group, so the learner knows which rewards a response earned,
but each reward's advantage is still spread evenly over the whole response. MRTPO keeps GDPO's per-reward advantages and
moves each one onto the tokens relevant to its reward.

For each reward k, the rollout policy replays its own response under two contexts that differ only in the information
the reward's verifier uses (the format rule, a reference answer, an execution error, or one constraint of the
instruction): the context with all verifier blocks, and the same context without block k. At every response position,
the Jensen–Shannon distance q between the two next-token distributions measures how much the token depends on that
information, and becomes a mean-preserving weight for reward k's advantage:

```python
# A[k]: GDPO group-normalized advantage of reward k, shape (B,), one value per response
# js[k]: per-token Jensen-Shannon divergence between the two replays, shape (B, T)
token_adv = 0
for k in range(K):
    q = (js[k] / math.log(2)).sqrt()                     # Jensen-Shannon distance in [0, 1]
    w = (1 - b) + b * q / q.mean(dim=-1, keepdim=True)   # b = beta * smoothstep(step / N); mean of w is 1
    token_adv = token_adv + A[k][:, None] * w
# then the reallocation bound and GDPO's batch rescaling
```

![Leave-one-out replay on a text-to-SQL response](replay_viz/figures/text_to_sql.png)

- β = 0.5 and a 10-step smoothstep warm-up in all settings. With β = 0, MRTPO is exactly GDPO.
- Each response keeps its GDPO advantage on average; only its distribution over tokens changes.
- The **reallocation bound** scales the token-level reallocation down by one common factor whenever its second moment
  over the batch exceeds that of the GDPO advantage.
- The replay reuses the rollout policy and samples nothing new. It runs only for (response, reward) pairs whose group
  varies on that reward; otherwise the advantage is zero and the weights cannot change it.

## 📁 What's in this repo

| Setting | Models | Data | Rewards | Methods | Code |
|---|---|---|---|---|---|
| [Instruction following](./instruction_following) | Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507 | HIR-16K (first two subsets, 6,998 prompts) | one per constraint, {0, 1}, 5–13 per prompt | GRPO, GDPO, CoRT, MRTPO | [`verl_mrtpo/`](./verl_mrtpo) |
| [Text-to-SQL](./text_to_sql) | Qwen2.5-Coder-3B/7B-Instruct | Spider train (6,915 prompts) | format, executable, execution, {0, 1} | GRPO, GDPO, OPSD, MRTPO, GDPO + SAW, MRTPO + SAW | [`verl_mrtpo/`](./verl_mrtpo) |
| [Tool calling](./tool_calling) | Qwen2.5-1.5B/3B-Instruct | ToolRL RLLA (3,920 prompts) | format {0, 1}, correctness [−3, 3] | GRPO, GDPO, MRTPO | [`tool_calling/vendor/verl`](./tool_calling/vendor/verl) |

The three settings run on two verl-based training stacks:

- [`verl_mrtpo/`](./verl_mrtpo) is verl 0.5.0 with the `mrtpo` package (MRTPO, GDPO, SAW, CoRT, OPSD, and the two task
  rewards). Instruction following and text-to-SQL share it; their folders hold the data preparation, the launcher and
  the evaluation.
- [`tool_calling/`](./tool_calling) is self-contained: the verl code base released with GDPO, with GRPO, GDPO and
  MRTPO as advantage estimators, plus BFCL-v4 evaluation.

[`replay_viz/`](./replay_viz) replays a model's responses under your verifier blocks without training and draws each
reward's token weights (see "Using MRTPO on your own task").

## 🧩 Instruction following

- **Environment.** One conda environment from [`verl_mrtpo/environments`](./verl_mrtpo/environments): Python 3.11,
  PyTorch 2.6.0 (CUDA 12.4), vLLM 0.8.5, transformers 4.53.2, flash-attn 2.7.4.post1, Ray 2.47.1. A second environment
  with vLLM 0.19.0 serves the AdvancedIF judge.
- **Data.** `data/prepare_hir.py` downloads HIR-16K, keeps the 6,549 prompts checked by IFEval checkers and the 449
  checked by IFBench checkers (43,496 constraints), joins question and constraints with HIR's six layouts, and stores
  each prompt without each of its constraints for the replay.
- **Rewards and replay.** Each constraint is a binary reward from its official strict checker. The verifier block of a
  constraint is the constraint itself: MRTPO replays the response under the prompt with and without that constraint.
- **Training.** 300 steps, 64 prompts × 8 rollouts, 8 H200 GPUs per run; seeds 42–46.
- **Evaluation.** IFEval, IFBench and MulDimIF (5 samples, prompt- and instruction-level strict accuracy, MulDimIF
  all-pass) and AdvancedIF judged by Qwen3.6-35B-A3B with the official judge code (Overall and Rubric).

```bash
bash verl_mrtpo/environments/install.sh mrtpo && conda activate mrtpo
cd instruction_following
python data/prepare_hir.py
python training/launch.py --method mrtpo --seed 42 --model /models/Qwen2.5-7B-Instruct --output runs/qwen25-7b-mrtpo-s42
```

See [`instruction_following/README.md`](./instruction_following/README.md).

## 🗄️ Text-to-SQL

- **Environment.** The same environment as instruction following.
- **Data.** `data/prepare.sh` downloads Spider, BIRD dev, Spider-Syn, Spider-Realistic, Spider-DK, SQL-R1's schema
  caches and OmniSQL's reasoning (OPSD only) at pinned versions, builds SQL-R1-style prompts whose output format is
  `<think>…</think><answer>```sql …```</answer>`, and drops the 3 training questions whose gold query fails and the 82
  prompts longer than 4,096 tokens (7,000 → 6,915).
- **Rewards and replay.** Format, executable and execution match, each 0 or 1. The verifier blocks are the format rule,
  the response's own SQLite error, and the reference query.
- **Training.** 100 steps, 32 prompts × 8 rollouts, 8 H20 GPUs per run; seeds 42, 1, 2.
- **Evaluation.** Execution accuracy with greedy decoding on Spider dev / test / Syn / Realistic / DK and BIRD dev.

```bash
bash verl_mrtpo/environments/install.sh mrtpo && conda activate mrtpo
cd text_to_sql
bash data/prepare.sh
python training/launch.py --method mrtpo --seed 42 --model /models/Qwen2.5-Coder-7B-Instruct --output outputs/coder7B-mrtpo-s42
```

See [`text_to_sql/README.md`](./text_to_sql/README.md).

## 🛠️ Tool calling

- **Environment.** Three conda environments from [`tool_calling/environments`](./tool_calling/environments): training
  (Python 3.10, PyTorch 2.4.0, vLLM 0.6.3), the vLLM 0.11.0 server used during evaluation, and BFCL-v4 (gorilla
  `6ea57973`).
- **Data.** `scripts/fetch_data.py` downloads ToolRL's `rlla_4k` (3,920 training and 80 validation rows) at a pinned
  commit and adds the replay reference: the label with numeric and boolean argument values written in the types the
  tool schema declares. The reward scores the original label.
- **Rewards and replay.** Format {0, 1} and correctness [−3, 3], as in ToolRL and GDPO. The verifier blocks are the
  required response form and the reference calls.
- **Training.** 100 steps, 512 prompts × 4 rollouts, 4 A100 GPUs per run; seeds 0, 1, 2, 4, 5 for 1.5B and 0, 1, 2 for 3B.
- **Evaluation.** BFCL-v4 single-turn AST (Non-Live 1,150 and Live 1,351 cases) with the official checkers.

```bash
cd tool_calling
bash environments/install_training.sh mrtpo-tc-train && conda activate mrtpo-tc-train
python scripts/fetch_data.py
python training/launch.py --method mrtpo --seed 0 --model-size 1.5b --model /models/Qwen2.5-1.5B-Instruct --output outputs/1p5b-mrtpo-s0
```

See [`tool_calling/README.md`](./tool_calling/README.md).

## 🧪 Using MRTPO on your own task

MRTPO needs two things from a task:

1. **A vector of rewards**, each scored by its own verifier, as in GDPO.
2. **For every reward, a text block that carries the information its verifier uses.** The two replay contexts differ
   only in this block, so the block decides which tokens receive the reward's advantage.

Whatever the task, the replay compares two contexts for each reward k: **P_all**, which contains the blocks of all
rewards, and **P_\k**, the same context with block k removed. Where the blocks come from depends on whether the rollout
prompt already shows the verifier information:

- **The prompt does not show it** (tool calling, text-to-SQL). The verifier holds information the policy never sees
  while sampling: the reference calls, the reference query, the response's own SQLite error. We write each piece as a
  block and append the blocks to the rollout prompt P_0, so P_all = P_0 + all blocks and P_\k = P_0 + all blocks
  except block k. The rollout itself still uses P_0.
- **The prompt already shows it** (instruction following). Every constraint the verifiers check is part of the
  instruction, so the rollout prompt is already P_all. Nothing is added; P_\k is the rollout prompt with constraint k
  deleted.

The blocks of the three settings:

| Setting | Reward | Block |
|---|---|---|
| Tool calling | format | ``For the final user request, respond in this form: …`` (the tag structure of the label) |
| Tool calling | correctness | ``The correctness reference for the final user request contains these ordered operations:`` followed by ``1. Operation name: `book_room`.``, ``Parameter `capacity`: 8.`` and ``Parameter `city`: "Oslo".`` |
| Text-to-SQL | format | ``For the final user request, respond in this form: <think>...</think> <answer> ```sql ... ``` </answer> …`` |
| Text-to-SQL | executable | ``When this query was executed on this database, SQLite raised: `no such column: head_age`.`` (empty if the query ran) |
| Text-to-SQL | execution | ``The reference query for this question is: `SELECT count(*) FROM head WHERE age > 56`.`` |
| Instruction following | each constraint | the constraint itself, e.g. ``Finish your response with this exact phrase Is there anything else I can help with?`` |

How we write a block:

- **One reward, one block.** Each constraint, rubric item or verifier gets its own block; never merge several into one.
  The main contrast is always leave-one-out: P_all against P_all without block k.
- **Write the information itself, specific to this example.** Build every block by code from the example's label or
  the verifier's own output: the actual tool names and argument values, the actual reference query, the actual error
  message. A generic requirement ("the query must execute") carries no information about this response. Give only the
  fields the verifier compares; do not copy the reasoning or the whole text of a reference answer.
- **Write it as a plain statement, like the rest of the prompt.** No instructions to the model ("use this
  information", "do not mention this"), no headings or introductions ("Verifier information for this problem"), no
  labels ("Grading note"), no words about training, replay, scoring or what counts as success, and no separator lines.
  When the information is something to avoid, say so directly (an injected instruction "should not be followed"), not
  only what would happen if it were followed.
- **Use only this response's own feedback.** A block that depends on the response, such as an execution error, quotes
  the error of this response, not of other responses in the group. If there is none, the block is empty; never invent
  one. An empty block leaves that reward with uniform token weights on this response.
- **Write references in the form the policy is asked to produce.** Argument values follow the types of the tool
  schema, and a reference answer uses the same tags and code fences as the required output.
- **Keep the prompt, the rewards and the blocks consistent.** The output format shown in the prompt is exactly what the
  format reward checks, under the same chat template and thinking mode used for training, and no block contradicts the
  prompt or the reward. Do not reuse a template from another task (for example a math `\boxed{}` instruction in a SQL
  prompt).
- **Keep the rollout prompt and the response as sampled.** P_0 is the training prompt exactly as the policy saw it
  (system message, chat template, any truncation). Added blocks go at the end of the user message, after a blank line,
  in a fixed order; the system message is unchanged. The response is replayed with its sampled token ids; never decode
  and re-tokenize it.
- **Do not make up references.** Blocks come from the data and the verifiers only; no synthetic wrong answers or
  references taken from other rollouts.
- **A reward may also keep GDPO's uniform weights altogether.** Some rewards carry no information that a replay can
  place on particular tokens, such as a word count or "answer in at most three sentences", which the response decides
  as a whole. Our code does not single these out and has no switch for it: in the instruction-following runs every
  constraint is replayed. On your own data you can mark such rewards (by type, or with a simple classifier over their
  text) and skip their reallocation, so they are spread over the tokens as in GDPO.

Before training, sample real rollouts with the real P_0, print the rendered contexts of every kind of example (each
response form, empty and non-empty blocks), check that block k belongs to reward k, and run the format reward on every
reference answer you show. Then look at the token weights with [`replay_viz/`](./replay_viz) (below).

The blocks of each setting are built in
[`verl_mrtpo/mrtpo/sql/constructions.py`](./verl_mrtpo/mrtpo/sql/constructions.py) and
[`verl_mrtpo/mrtpo/reward/sql_reward.py`](./verl_mrtpo/mrtpo/reward/sql_reward.py) (text-to-SQL),
[`verl_mrtpo/mrtpo/criteria.py`](./verl_mrtpo/mrtpo/criteria.py) (instruction following) and
[`tool_calling/vendor/verl/verl/utils/rlla_replay.py`](./tool_calling/vendor/verl/verl/utils/rlla_replay.py) (tool calling).

**Which rewards fit.** MRTPO helps when a reward's information, written as text, changes what the policy would write
at the positions its verifier reads: a tool name and its arguments, a table or column, a required phrase. Some rewards
carry little such information. In math reasoning, the usual rewards are the final answer and the response length: the
answer block points only at the last number, and a length limit is not decided by particular tokens. The coding setting
of GDPO pairs test-case pass rate with a length reward and a no-runtime-or-compile-error reward. We did not evaluate
MRTPO on these settings.

**Hyperparameters.** β = 0.5 works in all our settings and is a good default. Some tasks may prefer a smaller β.
The reallocation bound is there for stability. Without it, long training runs vary more: as training goes on, MRTPO
starts to behave partly like on-policy self-distillation, and on some models, datasets and seeds performance keeps
rising while on others it drops after many steps. The bound keeps the runs stable, so we use it in all settings. For
short training, MRTPO without the bound also works well (`--no-reallocation-bound` in every launcher).

**Check the replay before training.** [`replay_viz/`](./replay_viz) replays responses of the base model zero-shot under
your blocks and draws each reward's token weights, as in the example figures of the paper. If you are unsure whether a
block works, look at a few responses there first. The weights do not always land where one expects: sometimes, as in
on-policy self-distillation, a large share falls on the reasoning part of the response.

## 🙏 Acknowledgements

The tool-calling code starts from the verl implementation released with [GDPO](https://github.com/NVlabs/GDPO), and
the other two settings build on [verl](https://github.com/volcengine/verl). We also thank the teams behind
[vLLM](https://github.com/vllm-project/vllm), [HIR](https://github.com/sastpg/HIR), [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
[IFBench](https://github.com/allenai/IFBench), [MulDimIF](https://github.com/Junjie-Ye/MulDimIF),
[AdvancedIF](https://github.com/facebookresearch/AdvancedIF), [SQL-R1](https://github.com/DataArcTech/SQL-R1),
[OmniSQL](https://github.com/RUCKBReasoning/OmniSQL), [Spider](https://yale-lily.github.io/spider),
[BIRD](https://bird-bench.github.io), [ToolRL](https://github.com/qiancheng0/ToolRL) and the
[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla). The CoRT, OPSD and SAW baselines
follow their papers.

## License

Apache-2.0. Third-party code and data keep their original licenses; see [`THIRD_PARTY.md`](./THIRD_PARTY.md).
