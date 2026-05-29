from __future__ import annotations

from benchmarks.tau_bench.harness import TauBenchHarness
from benchmarks.webarena.harness import WebArenaHarness
from src.core import AgentResult, GoalStream, ObservationStream


def test_webarena_fixture_fallback_scores_correctly() -> None:
    """Fixture tasks score correctly when WebArena is not set up."""
    harness = WebArenaHarness()
    tasks = harness._fixture_tasks(5)
    scores = []
    for task in tasks:
        result = AgentResult(
            task.expected_answer,
            1,
            GoalStream(),
            ObservationStream(),
            final_tool_call=task.expected_tool_call,
        )
        scores.append(harness.score(task, result))
    assert scores == [1.0] * 5


def test_tau_bench_harness_scores_expected_answers() -> None:
    harness = TauBenchHarness()
    scores = []
    for task in harness.load_tasks(5):
        result = AgentResult(
            task.expected_answer,
            1,
            GoalStream(),
            ObservationStream(),
            final_tool_call=task.expected_tool_call,
        )
        scores.append(harness.score(task, result))
    assert scores == [1.0] * 5


def test_webarena_category_key_term_fallback() -> None:
    harness = WebArenaHarness()
    task = harness._fixture_tasks(1)[0]
    task.key_terms = []
    spec = harness.extract_task_spec(task)
    assert spec.key_terms == ["navigation"]


def test_webarena_loads_live_tasks_or_fixture() -> None:
    """load_tasks returns tasks regardless of whether WebArena is configured."""
    harness = WebArenaHarness()
    tasks = harness.load_tasks(3)
    assert len(tasks) == 3
    for task in tasks:
        assert task.instruction
        assert task.key_terms
