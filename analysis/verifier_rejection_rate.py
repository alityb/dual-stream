from __future__ import annotations

from pathlib import Path

from analysis.utils import ensure_paper_dirs, load_results, write_svg
from experiments import config


def main() -> None:
    """Generate verifier rejection-rate figure."""
    ensure_paper_dirs()
    results = [row for row in load_results() if row["condition"] == "dual_stream"]
    lines = []
    for benchmark in config.BENCHMARKS:
        values = [
            next(row["verifier_rejection_rate"] for row in results if row["benchmark"] == benchmark and row["step_budget"] == n)
            for n in config.STEP_BUDGETS
        ]
        lines.append(f"{benchmark}: " + ", ".join(f"N={n}: {value:.3f}" for n, value in zip(config.STEP_BUDGETS, values)))
    write_svg(Path("paper/figures/verifier_rejection_rate.svg"), "Verifier Rejection Rate", lines)


if __name__ == "__main__":
    main()
