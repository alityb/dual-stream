from __future__ import annotations

from src.core import ObservationStream


def trim(stream: ObservationStream) -> None:
    """Trim obs_stream to budget. [INV-1: GoalStream not accepted as argument.]"""
    if not isinstance(stream, ObservationStream):
        raise TypeError("trim() accepts ObservationStream only")
    while stream.tokens_used > stream.budget and stream.window:
        evicted = stream.window.popleft()
        stream.tokens_used -= evicted.tokens
