# MRTPO on Text-to-SQL

This folder reproduces the Text-to-SQL experiments in the paper. Given a SQLite schema and a question, the model
reasons and then writes one SQL query:

~~~
<think>
reasoning
</think>
<answer>
```sql
SELECT ...
```
</answer>
~~~

The prompt is SQL-R1's inference prompt (task overview, `SQLite`, a `CREATE TABLE` schema with example values, the
question, instructions). Its Output Format section shows the tagged form above, so the visible instruction and the
format reward agree.

Training uses three rewards, each 0 or 1, in this order:
- **Format**: `<think>`, `</think>`, `<answer>`, `</answer>` each appear exactly once, in this order, and `<answer>`
  contains a non-empty ```` ```sql ```` block.
- **Executable**: the last ```` ```sql ```` block after the last `<answer>` runs on the database without error within 30 s.
- **Correct**: its result-row set equals the result-row set of the gold SQL.

GRPO uses the mean of the three; GDPO, MRTPO and SAW use them as three criteria.

**MRTPO replay.** For each response the reward also returns one context block per criterion:

- format: the format rule
  ~~~
  For the final user request, respond in this form:

  <think>...</think>
  <answer>
  ```sql
  ...
  ```
  </answer>

  Use exactly one <think> block followed by exactly one <answer> block, and put the final SQL query in a ```sql code block inside <answer>.
  ~~~
- executable: this response's SQLite error, ``When this query was executed on this database, SQLite raised: `<error>`.``
  (or ``This response contains no ```sql block inside <answer>.``), and nothing if the query ran;
- correct: ``The reference query for this question is: `<gold SQL>`.``

The full replay context is the prompt followed by every non-empty block (separated by blank lines); the context for
criterion k leaves out block k. The policy's next-token distributions on the same response under the two contexts give
the per-token Jensen–Shannon distance that spreads criterion k's advantage over tokens. A criterion whose block is
empty (the executable criterion when the query ran) keeps uniform token weights.

**Example** (Spider train, `department_management`). Question: *How many heads of the departments are older than 56?*
Gold SQL: `SELECT count(*) FROM head WHERE age  >  56`. Response:

~~~
<think>
Count the heads older than 56.
</think>
<answer>
```sql
SELECT count(*) FROM head WHERE head_age > 56
```
</answer>
~~~

Rewards [format, executable, correct] = [1, 0, 0] (GRPO score 1/3). The executable block is
``When this query was executed on this database, SQLite raised: `no such column: head_age`.`` With `age` instead of
`head_age` the rewards are [1, 1, 1] and the executable block is empty.

The training code is the shared stack in [`../verl_mrtpo`](../verl_mrtpo) (verl fork + `mrtpo` package): the reward,
the prompt construction, MRTPO, SAW and OPSD live there. This folder holds the data preparation, the launcher and the
evaluation.

## Installation

One conda environment for training and evaluation (Python 3.11, PyTorch 2.6.0+cu124, vLLM 0.8.5, transformers 4.53.2,
flash-attn 2.7.4.post1, ray 2.47.1, pysqlite3-binary 0.5.4.post2; all versions pinned in
[`../verl_mrtpo/environments/requirements.lock.txt`](../verl_mrtpo/environments/requirements.lock.txt)):

```bash
bash verl_mrtpo/environments/install.sh mrtpo && conda activate mrtpo
cd text_to_sql
python -m pytest -q tests      # CPU checks of the data code, the launcher and the scoring
```

## Data

```bash
bash data/prepare.sh            # optional: bash data/prepare.sh DATA_ROOT /path/to/Qwen2.5-Coder-7B-Instruct
```

This runs four deterministic steps under `data/` (or `DATA_ROOT`):

1. `data/fetch_data.py` downloads the sources into `raw/` and checks their hashes:

   | source | version | check |
   |---|---|---|
   | Spider 1.0 `spider_data.zip` (train, dev, test, 166 + 206 databases) | official release (Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`) | md5 `b039fdb9e1e3adc386a91c5b2211667e` |
   | BIRD dev `dev.zip` (11 databases) | https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip | md5 `04b4af221c9186361f09b16abfd917ec` |
   | SQL-R1 schema and value caches (`db_info/{spiderdev,spidertest,bird}_db_id2{db_info,sampled_db_values}.json`) | DataArcTech/SQL-R1 @ `ccfb511e` | sha256 in the script |
   | Spider-Syn `Spider-Syn/dev.json` | ygan/Spider-Syn @ `0b996a57` | sha256 in the script |
   | Spider-Realistic `spider-realistic.json` | Zenodo record 5205322 | md5 `40252e199d88e33ebf13825fa6ac4ca5` |
   | Spider-DK `Spider-DK.json`, `tables.json`, 3 new databases | ygan/Spider-DK @ `5e16fa1f` | sha256 in the script |
   | OmniSQL chain of thought for Spider train `data/spider/train_spider_enhanced_with_cot.json` (OPSD only) | seeklhy/OmniSQL-datasets @ `e1db8bb1`, read out of `data.zip` with range requests (33 MB of the 22 GB archive) | sha256 `4a6f0d27…a144` |

2. `data/build_jsonl.py` writes one JSONL file per set to `datasets/`. Every record holds the SQL-R1 prompt as a
   single user message, the gold SQL, the database path and `gold_ok` (the gold SQL runs within 30 s; BIRD dev
   questions 518 and 701 are also marked `gold_ok = false`, as in the paper's evaluation). Schemas:
   Spider dev, Syn and Realistic use SQL-R1's Spider-dev cache, Spider test its Spider-test cache, BIRD dev its BIRD
   cache (the question is prefixed by the BIRD evidence when present); the training set and Spider-DK read
   `tables.json` and take up to 2 distinct non-empty values per column from the database, as SQL-R1 does. Every
   column shows at most 6 example values.
3. `data/filter_train.py` drops training questions whose gold SQL fails (3) and prompts longer than 4,096 tokens under
   the Qwen2.5-Coder-7B-Instruct tokenizer with its chat template (82): 7,000 → **6,915**. Only the tokenizer is loaded.
4. `data/make_parquet.py` writes `parquet/train.parquet` (GRPO, GDPO, MRTPO, SAW) and `parquet/train_opsd.parquet`
   (OPSD: the same 6,915 rows plus the reference solution). The reference solution is
   `<think>\n{reasoning}\n</think>\n<answer>\n```sql\n{gold SQL}\n```\n</answer>`, where `{reasoning}` is OmniSQL's
   chain of thought for the question up to its first SQL block. The chain of thought is looked up by question text
   (first OmniSQL record with that question) and kept only if its last SQL block returns the gold row set of that
   record on that record's database; otherwise `{reasoning}` is empty (6,564 of 6,915 keep it).

| set | file | questions | scored (`gold_ok`) | sha256 of the JSONL |
|---|---|---|---|---|
| Spider train | `datasets/train_filtered.jsonl` | 6,915 | – | `4ef0eafae36c7ad15efd8822e0e9fbc156b814838f920cac00e13b2d3a5788b4` |
| Spider dev | `datasets/spider_dev.jsonl` | 1,034 | 1,034 | `7069dda6bcbee782abb30c71fac2000cec38243d33674f53449c06f1b7424a96` |
| Spider test | `datasets/spider_test.jsonl` | 2,147 | 2,147 | `161d4127091c5f453df1e9e0c7e86d19c06f1e6868619246a3a15e5cca8df849` |
| Spider-Syn | `datasets/spider_syn.jsonl` | 1,034 | 1,034 | `1e25bf3b37db65cf96d873360519639bfdb2ce1c271f800d37c591cdac5aa991` |
| Spider-Realistic | `datasets/spider_realistic.jsonl` | 508 | 508 | `44bdd2714761579582997f8798828143803dc703d26563e135342987646a4792` |
| Spider-DK | `datasets/spider_dk.jsonl` | 535 | 534 | `61696ea00bdda509d99d2eec2ac9ccc7294504a314e10c180ff2f4f5ff8fbdb9` |
| BIRD dev | `datasets/bird_dev.jsonl` | 1,534 | 1,532 | – |

The models are `Qwen/Qwen2.5-Coder-3B-Instruct` and `Qwen/Qwen2.5-Coder-7B-Instruct`; download them wherever you like
and pass the path.

## Training

Each run uses one node with 8 H20 GPUs and trains for 100 steps. HF copies of the model are saved
every 20 steps to `<output>/hf/global_step_{20,40,60,80,100}`; the per-step metrics go to `<output>/metrics.jsonl`.

```bash
conda activate mrtpo
python training/launch.py --method mrtpo --seed 42 \
  --model /models/Qwen2.5-Coder-7B-Instruct --output outputs/coder7B-mrtpo-s42
```

`--method` is `grpo`, `gdpo`, `mrtpo`, `opsd`, `gdpo_saw` or `mrtpo_saw`. Add `--dry-run` to print every setting and
the full command without training. The main comparison runs every method on both models with seeds 42, 1 and 2:

```bash
for size in 3B 7B; do
  for m in grpo gdpo mrtpo opsd gdpo_saw mrtpo_saw; do
    for s in 42 1 2; do
      python training/launch.py --method $m --seed $s \
        --model /models/Qwen2.5-Coder-$size-Instruct --output outputs/coder$size-$m-s$s
    done
  done
done
```

Settings shared by all methods (identical for both models):

| setting | value |
|---|---|
| batch | 32 prompts × 8 rollouts per step, one PPO update per step (mini-batch 32 prompts) |
| sampling | temperature 1.1, top-p 1.0, no top-k |
| lengths | prompt 4,096 tokens (longer prompts were removed in the data), response 2,048 tokens |
| optimizer | AdamW, lr 1e-6, 15 warm-up steps then constant, weight decay 0.1, gradient clipping 1.0 |
| loss | clipped PPO objective, clip range [0.2, 0.2], entropy coefficient 0, token-mean over the whole update |
| KL | KL loss `low_var_kl`, coefficient 0.001, for GRPO, GDPO and MRTPO (with or without SAW); none for OPSD |
| advantages | GRPO: group-normalized mean reward; GDPO: per-criterion group normalization, summed, then rescaled per response over the batch; MRTPO and SAW build on GDPO |
| MRTPO | token weights `w = (1 − β) + β·q / mean(q)`, `q = sqrt(JS / ln 2)`, β = 0.5 ramped in with a cubic smoothstep over the first 10 steps; reallocation bound on (ρ = 1.0) |
| SAW | criterion k's group advantage scaled by `n · CV_k / Σ_j CV_j` over the batch (n = 3) |
| OPSD | teacher = the frozen initial model whose context is the prompt, the reference solution as an assistant turn and the prompt again; forward KL on the teacher's top-64 tokens, each term clipped at 2.0 |
| seeds | 42, 1, 2 (data order, rollout sampling and process seed) |

## Evaluation

We evaluate the step-100 model on the six sets with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py \
  --model outputs/coder7B-mrtpo-s42/hf/global_step_100 --out eval/coder7B-mrtpo-s42
```

Pass the downloaded model as `--model` to evaluate a base model. Decoding is greedy (temperature 0, one sample) with
vLLM in bf16, the model's chat template, at most 2,048 new tokens, context 8,192, seed 0. The metric is execution
accuracy: the last ```` ```sql ```` block after the last `<answer>` must run within 30 s and return the same row set as
the gold SQL (gold SQL timeout 60 s). Only records with `gold_ok` are scored (the "scored" column above). Each set gets
`eval/<run>/<set>/result.json` (`accuracy`, plus executable rate, format rate, truncation rate and, for BIRD, accuracy
by difficulty) and `responses.jsonl`. If the run is interrupted, rerun the command; finished sets are skipped.

The tables report the mean over seeds 42, 1 and 2 for each set, and **Avg** is the mean of the six sets:

```bash
python evaluation/summarize.py eval/coder7B-mrtpo-s42 eval/coder7B-mrtpo-s1 eval/coder7B-mrtpo-s2
```

## Code map

| what | where |
|---|---|
| prompt, format rule, reference block | [`../verl_mrtpo/mrtpo/sql/constructions.py`](../verl_mrtpo/mrtpo/sql/constructions.py) |
| SQL extraction and execution | [`../verl_mrtpo/mrtpo/sql/runtime.py`](../verl_mrtpo/mrtpo/sql/runtime.py) |
| three rewards and the per-response replay blocks | [`../verl_mrtpo/mrtpo/reward/sql_reward.py`](../verl_mrtpo/mrtpo/reward/sql_reward.py) |
| leave-one-out replay contexts | `build_criterion_replay` in [`../verl_mrtpo/mrtpo/criteria.py`](../verl_mrtpo/mrtpo/criteria.py) |
| per-token JS between the two contexts | `compute_criterion_js` in [`../verl_mrtpo/verl/workers/actor/dp_actor.py`](../verl_mrtpo/verl/workers/actor/dp_actor.py) |
| token allocation, reallocation bound, SAW | `allocate_criterion_advantage`, `bound_reallocation`, `saw_criterion_weights` in [`../verl_mrtpo/mrtpo/advantage.py`](../verl_mrtpo/mrtpo/advantage.py) |
| OPSD teacher prompt and loss | [`../verl_mrtpo/mrtpo/opsd.py`](../verl_mrtpo/mrtpo/opsd.py) |
| data, launcher, evaluation | `data/`, `training/launch.py`, `evaluation/` in this folder |
