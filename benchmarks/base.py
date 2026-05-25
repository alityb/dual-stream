from __future__ import annotations

from abc import ABC, abstractmethod

from core import AgentResult, BenchmarkTask, TaskSpec


class BenchmarkHarness(ABC):
    @abstractmethod
    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Load benchmark tasks."""
        return []

    @abstractmethod
    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        """Extract TaskSpec for verifier grounding."""
        return TaskSpec(text=task.instruction, key_terms=task.key_terms)

    @abstractmethod
    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Return binary score for one task."""
        return 0.0

    @abstractmethod
    def setup(self) -> None:
        """Prepare benchmark resources."""
        return None

    @abstractmethod
    def teardown(self) -> None:
        """Release benchmark resources."""
        return None


def score_answer(task: BenchmarkTask, result: AgentResult) -> float:
    """Score deterministic fixture answers via answer match or state observation."""
    expected = task.expected_answer.lower()
    # Check final answer text
    if result.answer and expected in result.answer.lower():
        return 1.0
    # State-based: check if the environment confirmed task completion in observations
    for entry in result.obs_stream_snapshot.window:
        if entry.role == "tool_output" and expected in entry.content.lower():
            return 1.0
    return 0.0
