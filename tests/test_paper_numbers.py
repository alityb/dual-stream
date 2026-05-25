from __future__ import annotations

from analysis.verify_paper_numbers import main


def test_verify_paper_numbers(monkeypatch) -> None:
    rows = []
    for condition in ("dual_stream", "flat_sliding_window"):
        for step_budget in (50, 100):
            rows.append(
                {
                    "benchmark": "webarena",
                    "condition": condition,
                    "step_budget": step_budget,
                    "completion_rate": {
                        ("dual_stream", 50): 0.76,
                        ("dual_stream", 100): 0.72,
                        ("flat_sliding_window", 50): 0.70,
                        ("flat_sliding_window", 100): 0.30,
                    }[(condition, step_budget)],
                    "goal_stream_overhead_pct": 3.78,
                    "verifier_rejection_rate": 0.04,
                    "n_tasks": 50,
                }
            )
    monkeypatch.setattr("analysis.verify_paper_numbers.load_results", lambda: rows)
    main()
