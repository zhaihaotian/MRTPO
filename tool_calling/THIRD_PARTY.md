# Third-party code and data

| Component | Source | License | Use here |
|---|---|---|---|
| verl | https://github.com/volcengine/verl (Bytedance) | Apache-2.0 | training stack in `vendor/verl`; original copyright headers retained, `vendor/verl/LICENSE` |
| GDPO (verl-GDPO) | https://github.com/NVlabs/GDPO | Apache-2.0 | the verl copy with the GDPO estimator and the RLLA reward that `vendor/verl` is derived from |
| ToolRL | https://github.com/qiancheng0/ToolRL | see repository | `rlla_4k` data (downloaded by `scripts/fetch_data.py` at commit `8cee13e`, not redistributed); RLLA prompt and parser in `evaluation/v4/toolrl_adapter.py`; reward definition in `vendor/verl/verl/utils/reward_score/rlla.py` |
| BFCL (gorilla) | https://github.com/ShishirPatil/gorilla, commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | Apache-2.0 | evaluator and benchmark data, installed separately by `environments/install_evaluation.sh` |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | rollout (0.6.3, glue code in `vendor/verl/verl/third_party/vllm`) and evaluation server (0.11.0) |
| FlashAttention | https://github.com/Dao-AILab/flash-attention, release 2.6.3 | BSD-3-Clause | training dependency (prebuilt wheel) |

Model weights (Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct), training data and BFCL data are obtained from their sources
and are not part of this repository.
