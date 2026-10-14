"""Load single source files of the release by path, so the tests need only torch (no ray, vLLM or GPU)."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERL = REPO / 'vendor' / 'verl' / 'verl'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mrtpo = load('mrtpo_under_test', VERL / 'trainer/ppo/mrtpo.py')
rlla = load('rlla_under_test', VERL / 'utils/reward_score/rlla.py')
replay = load('rlla_replay_under_test', VERL / 'utils/rlla_replay.py')
fetch_data = load('fetch_data_under_test', REPO / 'scripts/fetch_data.py')
