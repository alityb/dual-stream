from __future__ import annotations

from benchmarks.tau_bench.harness import TauBenchHarness
from benchmarks.webarena.harness import WebArenaHarness
from benchmarks.synthetic.harness import SyntheticHarness
from src.core import AgentResult, GoalStream, ObservationStream


def test_webarena_fixture_scores_correctly() -> None:
    harness = WebArenaHarness()
    tasks = harness._fixture_tasks(5)
    for task in tasks:
        result = AgentResult(
            task.expected_answer, 1, GoalStream(), ObservationStream(),
            final_tool_call=task.expected_tool_call,
        )
        assert harness.score(task, result) == 1.0


def test_tau_bench_harness_scores_expected_answers() -> None:
    harness = TauBenchHarness()
    for task in harness.load_tasks(5):
        result = AgentResult(
            task.expected_answer, 1, GoalStream(), ObservationStream(),
            final_tool_call=task.expected_tool_call,
        )
        assert harness.score(task, result) == 1.0


def test_webarena_category_key_term_fallback() -> None:
    harness = WebArenaHarness()
    task = harness._fixture_tasks(1)[0]
    task.key_terms = []
    spec = harness.extract_task_spec(task)
    assert spec.key_terms == ["navigation"]


def test_webarena_loads_tasks() -> None:
    harness = WebArenaHarness()
    tasks = harness.load_tasks(3)
    assert len(tasks) == 3


def test_synthetic_loads_tasks() -> None:
    harness = SyntheticHarness()
    tasks = harness.load_tasks(9)
    assert len(tasks) == 9
    assert all(t.instruction for t in tasks)
