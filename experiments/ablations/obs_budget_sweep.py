from __future__ import annotations

from experiments import config
from experiments.run_sweep import build_result


def run_obs_budget_sweep(tasks: int = 10) -> list[dict]:
    """Generate obs-budget sensitivity rows for dual_stream. [INV-4]"""
    rows = []
    for benchmark in config.BENCHMARKS:
        for multiplier in (0.5, 1.0, 2.0):
            row = build_result(benchmark, "dual_stream", 100, tasks)
            row["obs_budget"] = int(config.OBS_BUDGETS[benchmark] * multiplier)
            rows.append(row)
    return rows
