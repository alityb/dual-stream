from __future__ import annotations

from core import GoalEntry, GoalStream, TaskSpec
from verifier.checks import (
    check_redundancy,
    check_scope_narrowing,
    check_spec_consistency,
)
from verifier.verifier import validate


def test_scope_narrowing_accepts_root(goal_stream: GoalStream) -> None:
    result = check_scope_narrowing(GoalEntry(text="root", depth=0), goal_stream)
    assert result.valid is True


def test_scope_narrowing_rejects_parent_substring(goal_stream: GoalStream) -> None:
    proposed = GoalEntry(text="download all blog posts from example.com now", parent_id="root", depth=1)
    result = check_scope_narrowing(proposed, goal_stream)
    assert result.valid is False
    assert result.failed_check == "scope_narrowing"


def test_scope_narrowing_rejects_depth_over_cap(goal_stream: GoalStream) -> None:
    goal_stream.max_depth = 1
    proposed = GoalEntry(text="fetch blog posts page 1", parent_id="root", depth=2)
    result = check_scope_narrowing(proposed, goal_stream)
    assert result.valid is False
    assert result.failed_check == "scope_narrowing"


def test_redundancy_checks_completed_only(goal_stream: GoalStream) -> None:
    goal_stream.entries.append(GoalEntry(text="fetch blog posts page 1", status="completed", depth=1))
    result = check_redundancy(GoalEntry(text="  FETCH  blog posts page 1 ", depth=1), goal_stream)
    assert result.valid is False
    assert result.failed_check == "redundancy"


def test_redundancy_allows_active_duplicate(goal_stream: GoalStream) -> None:
    goal_stream.entries.append(GoalEntry(text="fetch blog posts page 1", status="active", depth=1))
    result = check_redundancy(GoalEntry(text="fetch blog posts page 1", depth=1), goal_stream)
    assert result.valid is True


def test_spec_consistency_empty_terms_passes() -> None:
    result = check_spec_consistency(GoalEntry(text="anything"), TaskSpec(text="open ended"))
    assert result.valid is True


def test_spec_consistency_requires_overlap(task_spec: TaskSpec) -> None:
    result = check_spec_consistency(GoalEntry(text="buy groceries"), task_spec)
    assert result.valid is False
    assert result.failed_check == "spec_consistency"


def test_validate_returns_first_failure(goal_stream: GoalStream, task_spec: TaskSpec) -> None:
    proposed = GoalEntry(text="download all blog posts from example.com", parent_id="root", depth=1)
    result = validate(proposed, goal_stream, task_spec)
    assert result.valid is False
    assert result.failed_check == "scope_narrowing"
