# Third-party code and data in instruction_following

| path | source | license |
|---|---|---|
| `evaluation/official/ifeval/` | IFEval checkers ([google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)), from HIR's evaluation folder | Apache-2.0 |
| `evaluation/official/ifbench/` | IFBench checkers ([allenai/IFBench](https://github.com/allenai/IFBench)), from HIR's evaluation folder; changes in `evaluation/official/NOTICE.md` | Apache-2.0 |
| `evaluation/muldim_official.py`, `evaluation/muldim_lib/` | MulDimIF evaluation code ([Junjie-Ye/MulDimIF](https://github.com/Junjie-Ye/MulDimIF)), from HIR's evaluation folder; `Length_Sentences` counts sentence terminators directly (same count, linear time) | Apache-2.0 |
| `evaluation/data/ifeval/input_data.jsonl` | IFEval prompts (541) | Apache-2.0 |
| `evaluation/data/ifbench/IFBench_test.jsonl` | IFBench test set (294) | ODC-BY-1.0 |
| `evaluation/data/muldimif/test.json` | MulDimIF test set (1,200), [Junjie-Ye/MulDimIF](https://github.com/Junjie-Ye/MulDimIF) | CC-BY-4.0 |
| `evaluation/advancedif/judge_cli.py` | calls the official [AdvancedIF](https://github.com/facebookresearch/AdvancedIF) judge, which `fetch.sh` downloads (not vendored) | AdvancedIF: CC BY-NC 4.0 |
| `data/prepare_hir.py` | builds prompts from [HIR-16K](https://huggingface.co/datasets/sastpg/HIR-16K) with HIR's layouts | HIR-16K: Apache-2.0 |

The evaluation files come from the evaluation folder of [HIR](https://github.com/sastpg/HIR).
