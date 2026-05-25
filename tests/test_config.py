from __future__ import annotations

from experiments import config


def test_experiment_config_matches_phase_five_decisions() -> None:
    assert config.STEP_BUDGETS == [10, 20, 50, 100]
    assert config.CONDITIONS == ["dual_stream", "flat_sliding_window", "flat_sink_recent", "verifier_off"]
    assert config.BENCHMARKS == ["webarena", "tau_bench"]
    assert config.OBS_BUDGETS == {"webarena": 4096, "tau_bench": 4096}
    assert config.MODEL == "Qwen/Qwen3-32B"
