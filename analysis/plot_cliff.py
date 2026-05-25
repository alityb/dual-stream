from __future__ import annotations

from pathlib import Path

from analysis.utils import ensure_paper_dirs, load_results, write_svg
from experiments import config


def main() -> None:
    """Generate completion-rate cliff figure from result JSON."""
    ensure_paper_dirs()
    results = load_results()
    lines = []
    for benchmark in config.BENCHMARKS:
        lines.append(f"{benchmark}")
        for condition in config.CONDITIONS:
            rates = [
                next(
                    row["completion_rate"]
                    for row in results
                    if row["benchmark"] == benchmark
                    and row["condition"] == condition
                    and row["step_budget"] == step_budget
                )
                for step_budget in config.STEP_BUDGETS
            ]
            labels = ", ".join(f"N={n}: {rate:.2f}" for n, rate in zip(config.STEP_BUDGETS, rates))
            lines.append(f"  {condition}: {labels}")
    write_svg(Path("paper/figures/completion_cliff.svg"), "Completion Rate vs Step Budget", lines)


if __name__ == "__main__":
    main()
