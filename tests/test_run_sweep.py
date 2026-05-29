from __future__ import annotations

import json

from benchmarks.base import BenchmarkHarness
from src.core import AgentResult, BenchmarkTask, GoalStream, ObservationStream, TaskSpec
from experiments import config
from experiments.run_sweep import build_result, write_sweep


class FakeBackend:
    def complete(self, context: str, model: str) -> str:
        return "<goal_update>work on task</goal_update><tool_call>finish()</tool_call>"


class FakeHarness(BenchmarkHarness):
    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        count = n or 3
        return [
            BenchmarkTask(
                id=f"fake-{i}",
                instruction=f"do task {i}",
                key_terms=["task"],
                expected_answer="done",
            )
            for i in range(count)
        ]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        return TaskSpec(id=task.id, text=task.instruction, key_terms=task.key_terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        return 1.0

    def make_tool_executor(self, task: BenchmarkTask):
        def ex(call: str) -> str:
            task.metadata["last_score"] = "1.0"
            return "done"
        return ex

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None


def test_build_result_schema(monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda b: FakeHarness())
    result = build_result(
        "synthetic", "dual_stream", 20, 5, backend_factory=lambda: FakeBackend()
    )
    required_keys = {
        "benchmark", "condition", "step_budget", "model", "seed", "obs_budget",
        "n_tasks", "completion_rate", "mean_score", "mean_steps_used",
        "mean_goal_stream_tokens", "mean_total_context_tokens",
        "goal_stream_overhead_pct", "verifier_rejection_rate", "tasks",
    }
    assert required_keys.issubset(set(result))
    assert result["benchmark"] == "synthetic"


def test_write_sweep_skips_existing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda b: FakeHarness())
    paths = write_sweep(
        3, tmp_path,
        backend_factory=lambda: FakeBackend(),
        benchmarks=["synthetic"],
        conditions=["dual_stream"],
        step_budgets=[20],
    )
    assert len(paths) == 1
    # Second run should skip (file exists)
    paths2 = write_sweep(
        3, tmp_path,
        backend_factory=lambda: FakeBackend(),
        benchmarks=["synthetic"],
        conditions=["dual_stream"],
        step_budgets=[20],
    )
    assert len(paths2) == 1


def test_write_sweep_filters_dimensions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda b: FakeHarness())
    paths = write_sweep(
        3, tmp_path,
        backend_factory=lambda: FakeBackend(),
        benchmarks=["synthetic"],
        conditions=["verifier_off"],
        step_budgets=[50],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["condition"] == "verifier_off"
    assert payload["step_budget"] == 50
