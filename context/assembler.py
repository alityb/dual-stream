from __future__ import annotations

from core import GoalStream, ObservationStream


def render_goal_stream(goal_stream: GoalStream) -> str:
    """Render full goal state in prefix position. [INV-6]"""
    root = next((entry for entry in goal_stream.entries if entry.depth == 0), None)
    active_stack = goal_stream.active_stack()
    active = active_stack[-1] if active_stack else root
    completed = sum(1 for entry in goal_stream.entries if entry.status == "completed")
    root_text = root.text if root is not None else ""
    active_depth = active.depth if active is not None else 0
    active_text = active.text if active is not None else ""
    return "\n".join(
        [
            "[GOAL STATE]",
            f"Root: {root_text}",
            f"Active subgoal (depth {active_depth}): {active_text}",
            f"Completed: {completed} subgoals",
        ]
    )


def render_obs_stream(obs_stream: ObservationStream) -> str:
    """Render FIFO observation history currently in budget. [INV-1]"""
    lines = [f"[OBSERVATIONS — {obs_stream.tokens_used}/{obs_stream.budget} tokens]"]
    lines.extend(
        f"[step {entry.step} | {entry.role}] {entry.content}"
        for entry in obs_stream.window
    )
    return "\n".join(lines)


def build_context(goal_stream: GoalStream, obs_stream: ObservationStream) -> str:
    """Concatenate streams with GoalStream first. [INV-1, INV-6]"""
    goal_section = render_goal_stream(goal_stream)
    obs_section = render_obs_stream(obs_stream)
    return f"{goal_section}\n\n{obs_section}"
