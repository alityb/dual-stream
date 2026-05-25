from __future__ import annotations

from pathlib import Path

from analysis.utils import load_results


def require(text: str, value: str) -> None:
    """Require one result-derived value in the paper source."""
    if value not in text:
        raise SystemExit(f"paper/main.tex is missing result value {value}")


def main() -> None:
    """Verify paper-visible experiment numbers trace to result JSON fields."""
    paper = Path("paper/main.tex").read_text(encoding="utf-8")
    results = load_results()

    def row(benchmark: str, condition: str, step_budget: int) -> dict:
        return next(
            item
            for item in results
            if item["benchmark"] == benchmark
            and item["condition"] == condition
            and item["step_budget"] == step_budget
        )

    traced_values = [
        f"{row('webarena', 'dual_stream', 50)['completion_rate']:.2f}",
        f"{row('webarena', 'dual_stream', 100)['completion_rate']:.2f}",
        f"{row('webarena', 'flat_sliding_window', 50)['completion_rate']:.2f}",
        f"{row('webarena', 'flat_sliding_window', 100)['completion_rate']:.2f}",
        f"{row('webarena', 'dual_stream', 100)['goal_stream_overhead_pct']:.2f}",
        f"{row('webarena', 'dual_stream', 100)['verifier_rejection_rate']:.2f}",
        str(row('webarena', 'dual_stream', 100)["step_budget"]),
        str(row('webarena', 'dual_stream', 100)["n_tasks"]),
    ]
    for value in traced_values:
        require(paper, value)


if __name__ == "__main__":
    main()
