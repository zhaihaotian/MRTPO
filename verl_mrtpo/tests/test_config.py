from pathlib import Path

from omegaconf import OmegaConf

CONFIG = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config" / "ppo_trainer.yaml"


def test_algorithm_defaults_are_the_paper_settings():
    algorithm = OmegaConf.load(CONFIG).algorithm
    assert algorithm.batch_norm == "response" and algorithm.saw is False
    assert dict(algorithm.mrtpo) == {"beta": 0.5, "ramp_steps": 10, "reallocation_bound": True,
                                     "reallocation_ratio": 1.0, "js_chunk_size": 128}
    assert dict(algorithm.cort) == {"eta": 0.5, "tau": 1.0, "center": 0.0, "ramp_steps": 100}
    assert algorithm.opsd.topk == 64 and algorithm.opsd.token_clip == 2.0
    assert algorithm.opsd.teacher_template == "reference_field_turn_repeat"
    assert algorithm.opsd.emitted_think_tokens is True and algorithm.opsd.teacher_max_prompt_length == 7168
