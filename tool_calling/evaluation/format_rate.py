"""BFCL format rate: whether each output has one of the exact RLLA tag structures (the training format reward)."""
import importlib.util
from pathlib import Path
import statistics

SOURCE = Path(__file__).resolve().parents[1] / 'vendor/verl/verl/utils/reward_score/rlla.py'
_spec = importlib.util.spec_from_file_location('rlla_format_rate', SOURCE)
rlla = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlla)
# A label with a tool call, a direct answer, or both: an output passes if it matches any of the three structures.
SELECTORS = ('<tool_call>', '<response>', '<tool_call><response>')


def flatten(value):
    if isinstance(value, str):
        return [value]
    return [text for item in value for text in flatten(item)]


def case_format(row):
    """Share of the case's assistant outputs that pass; 0 for a case without output."""
    texts = [rlla.assistant_turn(text) for text in flatten(row['result'])]
    if not texts:
        return 0.0
    passing = sum(any(rlla.format_reward(text, selector) == 1.0 for selector in SELECTORS) for text in texts)
    return passing / len(texts)


def category_format(rows):
    """Category format rate: mean over cases."""
    return statistics.mean(case_format(row) for row in rows)
