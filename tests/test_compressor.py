from __future__ import annotations

import pytest

from compressor.sliding_window import trim
from context.obs_stream import append_observation
from core import GoalStream, ObservationEntry, ObservationStream


def test_trim_rejects_goal_stream() -> None:
    with pytest.raises(TypeError):
        trim(GoalStream())  # type: ignore[arg-type]


def test_trim_at_exact_budget_keeps_entries() -> None:
    stream = ObservationStream(budget=5)
    append_observation(stream, ObservationEntry(step=1, role="tool_output", content="a", tokens=5))
    trim(stream)
    assert stream.tokens_used == 5
    assert len(stream.window) == 1


def test_trim_one_token_over_budget_evicts_fifo() -> None:
    stream = ObservationStream(budget=5)
    append_observation(stream, ObservationEntry(step=1, role="tool_output", content="old", tokens=3))
    append_observation(stream, ObservationEntry(step=2, role="environment", content="new", tokens=3))
    trim(stream)
    assert stream.tokens_used == 3
    assert [entry.content for entry in stream.window] == ["new"]
