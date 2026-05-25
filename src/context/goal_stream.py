from __future__ import annotations

from src.core import GoalEntry, GoalStream, TaskSpec, VerifierResult
from src.verifier.verifier import validate


def append_verified(
    stream: GoalStream, proposed: GoalEntry, spec: TaskSpec
) -> VerifierResult:
    """Append only verifier-approved goals. [INV-2, INV-6]"""
    result = validate(proposed, stream, spec)
    if result.valid:
        stream.entries.append(proposed)
    return result


def mark_complete(stream: GoalStream, goal_id: str, step: int) -> bool:
    """Mark an existing goal completed without removing it. [INV-6]"""
    for entry in stream.entries:
        if entry.id == goal_id:
            entry.status = "completed"
            entry.step_resolved = step
            return True
    return False
