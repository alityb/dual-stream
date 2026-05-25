from __future__ import annotations

from src.context.goal_stream import append_verified, mark_complete
from src.core import GoalEntry, GoalStream, TaskSpec


def test_active_stack_returns_root_to_leaf(task_spec: TaskSpec) -> None:
    root = GoalEntry(id="root", text="download all blog posts", depth=0)
    child = GoalEntry(id="child", text="fetch blog posts page 1", parent_id="root", depth=1)
    stream = GoalStream(entries=[child, root], spec=task_spec)
    assert [entry.id for entry in stream.active_stack()] == ["root", "child"]


def test_depth_zero_root_is_valid(task_spec: TaskSpec) -> None:
    stream = GoalStream(spec=task_spec)
    root = GoalEntry(text="download all blog posts", depth=0)
    result = append_verified(stream, root, task_spec)
    assert result.valid is True
    assert stream.entries == [root]


def test_append_verified_rejects_without_mutating(goal_stream: GoalStream, task_spec: TaskSpec) -> None:
    proposed = GoalEntry(text="unrelated task", parent_id="root", depth=1)
    result = append_verified(goal_stream, proposed, task_spec)
    assert result.valid is False
    assert len(goal_stream.entries) == 1


def test_mark_complete_updates_without_removal(goal_stream: GoalStream) -> None:
    assert mark_complete(goal_stream, "root", 3) is True
    assert len(goal_stream.entries) == 1
    assert goal_stream.entries[0].status == "completed"
    assert goal_stream.entries[0].step_resolved == 3


def test_token_count_uses_full_history(goal_stream: GoalStream, tokenizer) -> None:
    goal_stream.entries.append(GoalEntry(text="fetch blog posts page 1", parent_id="root", depth=1))
    assert goal_stream.token_count(tokenizer) > 0
