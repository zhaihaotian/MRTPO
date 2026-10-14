# Third-party code in verl_mrtpo

| path | source | license |
|---|---|---|
| `verl/` | [volcengine/verl](https://github.com/volcengine/verl) v0.5.0, pruned and modified as listed in README.md | Apache-2.0 (`LICENSE`, `Notice.txt`) |
| `mrtpo/hir.py` | `_concat_question` from [HIR](https://github.com/sastpg/HIR) (`HIR/verl/trainer/ppo/ray_trainer.py`, a verl fork) | Apache-2.0 (HIR/LICENSE) |
| `mrtpo/reward/if_score/ifeval/` | IFEval checkers ([google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval)), as copied in HIR | Apache-2.0 |
| `mrtpo/reward/if_score/ifbench/` | IFBench checkers ([allenai/IFBench](https://github.com/allenai/IFBench)), as copied in HIR; returns False instead of raising on word-less responses | Apache-2.0 |
| `mrtpo/sql/constructions.py`, `mrtpo/sql/sql_r1_reserved_words.json` | prompt template and schema format of [SQL-R1](https://github.com/DataArcTech/SQL-R1) | Apache-2.0 |
