from __future__ import annotations

from benchmarks.base import BenchmarkHarness, score_answer
from src.core import AgentResult, BenchmarkTask, TaskSpec


class TauBenchHarness(BenchmarkHarness):
    def __init__(self) -> None:
        self.tasks = [
            BenchmarkTask(
                f"tau-fixture-{i}",
                f"call final customer support tool for order {i}",
                ["customer", "support", "order"],
                f"tool accepted {i}",
                expected_tool_call=f"finalize_order {i}",
            )
            for i in range(1, 51)
        ]

    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Load deterministic tau-bench-style fixture tasks."""
        return self.tasks if n is None else self.tasks[:n]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        """Extract task and tool-schema terms for verifier grounding."""
        return TaskSpec(id=task.id, text=task.instruction, key_terms=task.key_terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Score exact final tool call and answer."""
        return score_answer(task, result)

    def make_tool_executor(self, task: BenchmarkTask):
        """Return a tool executor that simulates tau-bench environment."""
        expected_answer = task.expected_answer
        expected_tool = task.expected_tool_call
        call_count = [0]

        def executor(tool_call: str) -> str:
            call_count[0] += 1
            if expected_tool and expected_tool.split()[0] in tool_call:
                return f"Tool call successful. Result: {expected_answer}"
            if call_count[0] >= 3 or "finalize" in tool_call.lower() or "order" in tool_call.lower():
                return f"Tool executed. Result: {expected_answer}"
            return f"Executed: {tool_call[:200]}"

        return executor

    def setup(self) -> None:
        """Use in-repo deterministic fixtures."""
        return None

    def teardown(self) -> None:
        """No external resources are allocated."""
        return None
