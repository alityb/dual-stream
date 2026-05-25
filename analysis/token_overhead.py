from __future__ import annotations

from pathlib import Path

from analysis.utils import ensure_paper_dirs, load_results, write_svg
from experiments import config


def main() -> None:
    """Generate GoalStream overhead figure and LaTeX table."""
    ensure_paper_dirs()
    results = [row for row in load_results() if row["condition"] == "dual_stream"]
    lines = []
    table = [
        "\\begin{tabular}{lrrrr}",
        "Benchmark & N=20 & N=50 & N=100 & N=200 " + "\\\\",
        "\\hline",
    ]
    for benchmark in config.BENCHMARKS:
        values = [
            next(
                row["goal_stream_overhead_pct"]
                for row in results
                if row["benchmark"] == benchmark and row["step_budget"] == n
            )
            for n in config.STEP_BUDGETS
        ]
        lines.append(
            f"{benchmark}: "
            + ", ".join(
                f"N={n}: {value:.2f}%" for n, value in zip(config.STEP_BUDGETS, values)
            )
        )
        table.append(
            f"{benchmark} & "
            + " & ".join(f"{value:.2f}" for value in values)
            + " "
            + "\\\\"
        )
    table.append("\\end{tabular}")
    write_svg(Path("paper/figures/goal_stream_overhead.svg"), "GoalStream Overhead", lines)
    Path("paper/tables/token_overhead.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
