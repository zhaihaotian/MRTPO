# Vendored verl (GDPO code base)

The training runtime used by `training/launch.py`: verl as released with GDPO, reduced to the FSDP + vLLM 0.6.3 path and
the GRPO, GDPO and MRTPO advantage estimators for RLLA tool calling. Install with
`pip install --no-deps --no-build-isolation -e vendor/verl` after the locked training dependencies
(`environments/install_training.sh` does both).

verl is Apache-2.0 (`LICENSE`); original copyright notices are kept in the source files. See `../../THIRD_PARTY.md`.
