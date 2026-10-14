# BFCL evaluation protocol

## Inference

Each training run is evaluated at its step-100 model; the base models are evaluated with the same protocol. Every
checkpoint produces one output per case, with the same ToolRL prompt, parser, subsets and decoding for all methods.

| Item | Setting |
|---|---|
| Evaluator | BFCL V4, gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (source checkout, not the PyPI wheel) |
| Subsets | single-turn AST: Non-Live and Live, 2,501 cases in 10 categories |
| Prompt / parser | ToolRL RLLA handler on the V4 API (`evaluation/v4/toolrl_adapter.py`, `evaluation/v4/adapter.py`) |
| Interface | completion API, `is_fc_model=False` |
| Temperature / top-p / top-k | 0.6 / 0.95 / −1 |
| Repetition penalty / seed | 1 / 0 |
| Max new tokens / context | min(8,192, 32,768 − prompt tokens − 2) / 32,768 |
| Server | vLLM 0.11.0, TP 1, bf16, memory utilization 0.85, max 64 sequences, 8,192 batched tokens, eager mode, `--generation-config vllm` |
| Client concurrency | 48 cases |

A case that exceeds the context limit counts as a failure of the model.

## Accuracy

Per-category accuracy is the official BFCL AST checker's (correct cases / cases); argument values must have the type the
function schema declares, with no coercion.

| Group | Category | Cases |
|---|---|---:|
| Non-Live | `simple_python` / `simple_java` / `simple_javascript` | 400 / 100 / 50 |
| Non-Live | `multiple` / `parallel` / `parallel_multiple` | 200 / 200 / 200 |
| Live | `live_simple` / `live_multiple` / `live_parallel` / `live_parallel_multiple` | 258 / 1,053 / 16 / 24 |

$$
A_{\text{simple}}=\tfrac{1}{3}(A_{\text{Python}}+A_{\text{Java}}+A_{\text{JavaScript}}),\qquad
A_{\text{Non-Live}}=\tfrac{1}{4}(A_{\text{simple}}+A_{\text{multiple}}+A_{\text{parallel}}+A_{\text{parallel multiple}}),\qquad
A_{\text{Live}}=\frac{\sum_{c\in \text{Live}} N_c A_c}{1351}.
$$

`single_turn` = $(1150\,A_{\text{Non-Live}} + 1351\,A_{\text{Live}})/2501$ combines the two groups by their case counts.
`evaluation/protocol.py` computes these columns; `summary.json` stores them under `accuracy` (fractions) together with
the official per-category counts.

## Format

The assistant text of each output (after the Qwen assistant marker, whitespace trimmed) scores 1 if it matches one of
these structures exactly, with tag order, closing tags, newlines, and one block of each kind; otherwise 0.

| Output type | Structure |
|---|---|
| Direct answer | `<think>...</think>\n<response>...</response>` |
| Tool call | `<think>...</think>\n<tool_call>\n...\n</tool_call>` |
| Tool call and answer | `<think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>` |

The check is the training format reward (`rlla.py::format_reward`), applied for each of the three label types; an output
passes if any matches (`evaluation/format_rate.py`). For case $i$ with $m_i$ outputs, $F_i = \frac{1}{m_i}\sum_j f_{ij}$
(0 for a case without output); the category format rate is the mean of $F_i$ over cases. Non-Live, Live and
`single_turn` format use the same formulas as accuracy. `summary.json` stores them under `format`, and the per-category
values under `categories.<category>.format`.

## Seeds and aggregation

The reported values are mean ± sample standard deviation (ddof = 1) over training seeds of the step-100 models, in
percent (`evaluation/aggregate.py`, `--metric accuracy` or `--metric format`). Seeds: 0, 1, 2, 4, 5 for Qwen2.5-1.5B-Instruct and 0, 1, 2 for
Qwen2.5-3B-Instruct. The inference seed is 0.
