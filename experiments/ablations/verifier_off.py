from __future__ import annotations

from experiments import config
from experiments.run_sweep import build_result


def run_verifier_off(tasks: int = config.TASKS_PER_BENCHMARK) -> list[dict]:
    """Generate verifier-off ablation rows. [INV-4]"""
    return [
        build_result(benchmark, "verifier_off", step_budget, tasks)
        for benchmark in config.BENCHMARKS
        for step_budget in config.STEP_BUDGETS
    ]
