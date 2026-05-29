from __future__ import annotations

from experiments import config


def test_experiment_config_matches_decisions() -> None:
    assert "synthetic" in config.BENCHMARKS
    assert "flat_equal_budget" in config.CONDITIONS
    assert config.OBS_BUDGETS["synthetic"] == 512
    assert config.MODEL == "Qwen/Qwen3-32B"
    assert 20 in config.STEP_BUDGETS
    assert 50 in config.STEP_BUDGETS
