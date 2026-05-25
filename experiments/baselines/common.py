from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from core import AgentConfig, AgentResult, GoalEntry, GoalStream, ObservationStream, TaskSpec


@dataclass
class FlatBufferEntry:
    step: int
    role: str
    content: str
    tokens: int


@dataclass
class FlatBuffer:
    window: deque[FlatBufferEntry] = field(default_factory=deque)
    budget: int = 2048
    tokens_used: int = 0


def append_flat(buffer: FlatBuffer, entry: FlatBufferEntry) -> None:
    """Append to a flat baseline buffer. [INV-3]"""
    buffer.window.append(entry)
    buffer.tokens_used += entry.tokens


def trim_flat_fifo(buffer: FlatBuffer) -> None:
    """FIFO trim for flat sliding-window baseline. [INV-3]"""
    while buffer.tokens_used > buffer.budget and buffer.window:
        evicted = buffer.window.popleft()
        buffer.tokens_used -= evicted.tokens


def trim_flat_sink_recent(buffer: FlatBuffer, sink_tokens: int = 4) -> None:
    """Retain first sink entry and most recent entries. [INV-3]"""
    while buffer.tokens_used > buffer.budget and len(buffer.window) > 1:
        removable_index = 1 if buffer.window[0].tokens <= sink_tokens else 0
        entries = list(buffer.window)
        evicted = entries.pop(removable_index)
        buffer.window = deque(entries)
        buffer.tokens_used -= evicted.tokens


def render_flat_context(buffer: FlatBuffer) -> str:
    """Render a single context buffer for baseline calls. [INV-3]"""
    lines = [f"[FLAT CONTEXT — {buffer.tokens_used}/{buffer.budget} tokens]"]
    lines.extend(f"[step {entry.step} | {entry.role}] {entry.content}" for entry in buffer.window)
    return "\n".join(lines)


def empty_result(answer: str | None, steps: int, timed_out: bool, task: TaskSpec) -> AgentResult:
    """Adapt baseline results to the shared result type. [INV-3]"""
    return AgentResult(
        answer=answer,
        steps_used=steps,
        goal_stream_snapshot=GoalStream(entries=[GoalEntry(text=task.text, depth=0)], spec=task),
        obs_stream_snapshot=ObservationStream(),
        timed_out=timed_out,
    )
