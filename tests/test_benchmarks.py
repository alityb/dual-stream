from __future__ import annotations

from benchmarks.tau_bench.harness import TauBenchHarness
from benchmarks.webarena.harness import WebArenaHarness
from dual_stream.types import AgentResult, GoalStream, ObservationStream


def test_fixture_compatible_harnesses_score_expected_answers() -> None:
    for harness in (WebArenaHarness(), TauBenchHarness()):
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
    task = harness.load_tasks(1)[0]
    task.key_terms = []
    spec = harness.extract_task_spec(task)
    assert spec.key_terms == ["navigation"]
