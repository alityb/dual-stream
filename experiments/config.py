from __future__ import annotations

import os
from pathlib import Path


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name and raw_value.strip():
            return raw_value.strip().strip('"').strip("'")
    return None


STEP_BUDGETS = [10, 20, 50, 100, 200]
CONDITIONS = [
    "dual_stream",
    "flat_sliding_window",
    "flat_sink_recent",
    "flat_equal_budget",
    "verifier_off",
]
BENCHMARKS = ["synthetic", "webarena", "tau_bench"]
MODEL = _env_value("MODEL_NAME") or "Qwen/Qwen3-32B"
SEED = 42

OBS_BUDGETS = {
    "synthetic": 512,   # tight budget forces context drift in flat baselines
    "webarena": 4096,
    "tau_bench": 4096,
}
TASKS_PER_BENCHMARK = 100
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Calibrated average GoalStream tokens observed in dual_stream synthetic runs.
# Used to give flat_equal_budget the same total context as dual_stream.
# Updated after calibration run. Initial estimate: 80 tokens.
AVG_GOAL_STREAM_TOKENS: dict[str, int] = {
    "synthetic": 80,
    "webarena": 80,
    "tau_bench": 80,
}
