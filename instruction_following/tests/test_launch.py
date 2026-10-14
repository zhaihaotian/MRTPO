"""The launcher's settings compose with the trainer config, and each method gets the paper settings."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "instruction_following" / "training" / "launch.py"
CONFIG = ROOT / "verl_mrtpo" / "verl" / "trainer" / "config"


def dry_run(*args):
    out = subprocess.run([sys.executable, str(LAUNCH), "--seed", "42", "--model", "/models/m", "--output", "/tmp/run",
                          "--dry-run", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("args", [["--method", "grpo"], ["--method", "gdpo"], ["--method", "cort"],
                                  ["--method", "mrtpo"], ["--method", "mrtpo", "--beta", "0.25"]])
def test_overrides_compose(args):
    record = dry_run(*args)
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        config = compose("ppo_trainer", overrides=record["command"][4:])
    assert config.algorithm.adv_estimator == args[1]
    assert config.data.train_batch_size == 64 and config.actor_rollout_ref.rollout.n == 8
    assert config.trainer.total_training_steps == 300 and config.trainer.seed == 42
    assert config.actor_rollout_ref.actor.clip_ratio_high == 0.27 and config.actor_rollout_ref.rollout.top_k == 100
    assert config.algorithm.mrtpo.reallocation_bound is True
    if "--beta" in args:
        assert config.algorithm.mrtpo.beta == 0.25


def test_beta_requires_mrtpo():
    result = subprocess.run([sys.executable, str(LAUNCH), "--method", "gdpo", "--seed", "42", "--model", "m",
                             "--output", "/tmp/run", "--beta", "0.25", "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
