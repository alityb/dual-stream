from __future__ import annotations

from benchmarks.base import BenchmarkHarness, score_answer
from src.core import AgentResult, BenchmarkTask, TaskSpec


class WebArenaHarness(BenchmarkHarness):
    def __init__(self) -> None:
        self.tasks = [
            BenchmarkTask(
                f"webarena-fixture-{i}",
                f"navigate shop site to target state {i}",
                ["shop", "target", "state"],
                f"target reached {i}",
                category="navigation",
            )
            for i in range(1, 51)
        ]

    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Load deterministic WebArena-style fixture tasks."""
        return self.tasks if n is None else self.tasks[:n]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        """Use task category fallback when key terms are noisy."""
        terms = task.key_terms or ([task.category] if task.category else [])
        return TaskSpec(id=task.id, text=task.instruction, key_terms=terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Score whether the target state was reached."""
        return score_answer(task, result)

    def make_tool_executor(self, task: BenchmarkTask):
        """Return a tool executor that simulates WebArena environment."""
        expected = task.expected_answer
        call_count = [0]

        def executor(tool_call: str) -> str:
            call_count[0] += 1
            if call_count[0] >= 3 or "navigate" in tool_call.lower() or "target" in tool_call.lower():
                return f"Navigation successful. Result: {expected}"
            return f"Executed: {tool_call[:200]}"

        return executor

    def setup(self) -> None:
        """Use in-repo deterministic fixtures."""
        return None

    def teardown(self) -> None:
        """No external resources are allocated."""
        return None
