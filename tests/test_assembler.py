from __future__ import annotations

from dual_stream.context.assembler import build_context, render_goal_stream, render_obs_stream
from dual_stream.context.obs_stream import append_observation
from dual_stream.types import GoalEntry, GoalStream, ObservationEntry, ObservationStream, TaskSpec


def test_render_goal_stream_exact_format(task_spec: TaskSpec) -> None:
    root = GoalEntry(text="Download all blog posts from example.com", depth=0)
    child = GoalEntry(text="Fetch blog post #14 from /blog/post-14", parent_id=root.id, depth=1)
    child.status = "completed"
    leaf = GoalEntry(text="Fetch blog post #15 from /blog/post-15", parent_id=child.id, depth=2)
    stream = GoalStream(entries=[root, child, leaf], spec=task_spec)
    assert render_goal_stream(stream) == (
        "[GOAL STATE]\n"
        "Root: Download all blog posts from example.com\n"
        "Active subgoal (depth 2): Fetch blog post #15 from /blog/post-15\n"
        "Completed: 1 subgoals"
    )


def test_render_obs_stream_exact_format() -> None:
    stream = ObservationStream(budget=2048)
    append_observation(
        stream,
        ObservationEntry(
            step=8,
            role="tool_output",
            content="GET /blog/post-13 → 200 OK, 4200 bytes",
            tokens=1847,
        ),
    )
    assert render_obs_stream(stream) == (
        "[OBSERVATIONS — 1847/2048 tokens]\n"
        "[step 8 | tool_output] GET /blog/post-13 → 200 OK, 4200 bytes"
    )


def test_build_context_goal_first(goal_stream: GoalStream) -> None:
    obs_stream = ObservationStream()
    context = build_context(goal_stream, obs_stream)
    assert context.startswith("[GOAL STATE]")
    assert "\n\n[OBSERVATIONS" in context
