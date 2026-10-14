# Third-party code and data

This repository vendors or downloads code and data from other projects. Each part keeps its original license; the
details are listed per folder:

- [`verl_mrtpo/THIRD_PARTY.md`](./verl_mrtpo/THIRD_PARTY.md): verl 0.5.0, HIR's prompt layouts, the IFEval and IFBench
  checkers used as rewards, SQL-R1's prompt format.
- [`instruction_following/THIRD_PARTY.md`](./instruction_following/THIRD_PARTY.md): IFEval, IFBench and MulDimIF
  evaluation code and test sets, the AdvancedIF judge (downloaded, not vendored), HIR-16K.
- [`tool_calling/THIRD_PARTY.md`](./tool_calling/THIRD_PARTY.md): verl and GDPO's verl code base, ToolRL data and
  prompt, BFCL (gorilla), vLLM.

Text-to-SQL downloads Spider, BIRD dev, Spider-Syn, Spider-Realistic, Spider-DK, SQL-R1's schema caches and OmniSQL's
data from their original sources (`text_to_sql/data/fetch_data.py`); none of them is redistributed here.
