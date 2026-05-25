from __future__ import annotations

from dual_stream.context.obs_stream import append_observation
from dual_stream.types import ObservationEntry, ObservationStream


def test_append_observation_tracks_tokens() -> None:
    stream = ObservationStream(budget=10)
    append_observation(stream, ObservationEntry(step=1, role="tool_output", content="abc", tokens=3))
    assert stream.tokens_used == 3
    assert len(stream.window) == 1
