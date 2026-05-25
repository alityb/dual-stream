from __future__ import annotations

from core import ObservationEntry, ObservationStream


def append_observation(stream: ObservationStream, entry: ObservationEntry) -> None:
    """Append an observation and update token accounting before trim. [INV-1]"""
    stream.window.append(entry)
    stream.tokens_used += entry.tokens
