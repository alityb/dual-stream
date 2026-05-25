from __future__ import annotations

import re

from src.core import GoalEntry, GoalStream, TaskSpec, VerifierResult


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def check_scope_narrowing(proposed: GoalEntry, stream: GoalStream) -> VerifierResult:
    """Ensure a proposed goal narrows its parent. [INV-2, INV-5]"""
    if proposed.depth == 0:
        return VerifierResult(valid=True)
    if proposed.depth > stream.max_depth:
        return VerifierResult(
            valid=False,
            failed_check="scope_narrowing",
            reason="Proposed subgoal exceeds GoalStream max_depth. Use the active subgoal instead.",
        )
    parent = next(
        (entry for entry in stream.entries if entry.id == proposed.parent_id), None
    )
    if parent is None:
        return VerifierResult(
            valid=False,
            failed_check="scope_narrowing",
            reason="Proposed subgoal is broader than or equal to parent. Narrow the scope.",
        )
    parent_text = _normalize(parent.text)
    proposed_text = _normalize(proposed.text)
    if proposed.depth != parent.depth + 1 or parent_text in proposed_text:
        return VerifierResult(
            valid=False,
            failed_check="scope_narrowing",
            reason="Proposed subgoal is broader than or equal to parent. Narrow the scope.",
        )
    return VerifierResult(valid=True)


def check_redundancy(proposed: GoalEntry, stream: GoalStream) -> VerifierResult:
    """Reject exact duplicates of completed goals. [INV-2, INV-5]"""
    proposed_text = _normalize(proposed.text)
    for entry in stream.entries:
        if entry.status == "completed" and _normalize(entry.text) == proposed_text:
            return VerifierResult(
                valid=False,
                failed_check="redundancy",
                reason=f"This subgoal duplicates a completed entry: '{entry.text}'",
            )
    return VerifierResult(valid=True)


def check_spec_consistency(proposed: GoalEntry, spec: TaskSpec) -> VerifierResult:
    """Require overlap with original task terms. [INV-2, INV-5]"""
    if not spec.key_terms:
        return VerifierResult(valid=True)
    proposed_text = proposed.text.lower()
    if any(term.lower() in proposed_text for term in spec.key_terms):
        return VerifierResult(valid=True)
    return VerifierResult(
        valid=False,
        failed_check="spec_consistency",
        reason=f"Proposed subgoal has no overlap with task spec terms: {spec.key_terms}",
    )
