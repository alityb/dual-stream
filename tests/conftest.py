from __future__ import annotations

import pytest

from dual_stream.types import GoalEntry, GoalStream, TaskSpec


class SimpleTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


@pytest.fixture
def task_spec() -> TaskSpec:
    return TaskSpec(text="download all blog posts from example.com", key_terms=["blog posts", "download"])


@pytest.fixture
def goal_stream(task_spec: TaskSpec) -> GoalStream:
    root = GoalEntry(id="root", text=task_spec.text, depth=0, step_created=0)
    return GoalStream(entries=[root], spec=task_spec)
