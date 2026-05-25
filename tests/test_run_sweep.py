from __future__ import annotations

import json

from benchmarks.base import BenchmarkHarness
from dual_stream.types import AgentResult, BenchmarkTask, GoalStream, ObservationStream, TaskSpec
from experiments import config
from experiments.run_sweep import build_result, write_sweep


class FakeBackend:
    def complete(self, context: str, model: str) -> str:
        return "<goal_update>handle target task</goal_update><final_answer>passed</final_answer>"


class FakeHarness(BenchmarkHarness):
    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        count = n or 3
        return [
            BenchmarkTask(
                id=f"fake-{index}",
                instruction=f"handle target task {index}",
                key_terms=["target", "task"],
                expected_answer="passed",
            )
            for index in range(count)
        ]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        return TaskSpec(id=task.id, text=task.instruction, key_terms=task.key_terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        return 1.0 if result.answer == task.expected_answer else 0.0

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None


def test_build_result_schema(monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda benchmark: FakeHarness())
    result = build_result(
        "webarena", "dual_stream", 100, 10, backend_factory=lambda: FakeBackend()
    )
    assert set(result) == {
        "benchmark",
        "condition",
        "step_budget",
        "model",
        "seed",
        "obs_budget",
        "n_tasks",
        "completion_rate",
        "mean_steps_used",
        "mean_goal_stream_tokens",
        "mean_total_context_tokens",
        "goal_stream_overhead_pct",
        "verifier_rejection_rate",
        "tasks",
    }
    assert len(result["tasks"]) == 10


def test_write_sweep_creates_32_json_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda benchmark: FakeHarness())
    paths = write_sweep(3, tmp_path, backend_factory=lambda: FakeBackend())
    assert len(paths) == 32
    assert len(list(tmp_path.glob("*.json"))) == 32
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["model"] == config.MODEL


def test_write_sweep_filters_dimensions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda benchmark: FakeHarness())
    paths = write_sweep(
        3,
        tmp_path,
        backend_factory=lambda: FakeBackend(),
        benchmarks=["webarena"],
        conditions=["verifier_off"],
        step_budgets=[50],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["benchmark"] == "webarena"
    assert payload["condition"] == "verifier_off"
    assert payload["step_budget"] == 50
