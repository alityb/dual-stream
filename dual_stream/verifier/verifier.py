from __future__ import annotations

from dual_stream.types import GoalEntry, GoalStream, TaskSpec, VerifierResult
from dual_stream.verifier.checks import (
    check_redundancy,
    check_scope_narrowing,
    check_spec_consistency,
)


def validate(
    proposed: GoalEntry, stream: GoalStream, spec: TaskSpec, enabled: bool = True
) -> VerifierResult:
    """Run deterministic structural verifier checks. [INV-2, INV-5]"""
    if not enabled:
        return VerifierResult(valid=True)
    for check in (
        check_scope_narrowing(proposed, stream),
        check_redundancy(proposed, stream),
        check_spec_consistency(proposed, spec),
    ):
        if not check.valid:
            return check
    return VerifierResult(valid=True)
