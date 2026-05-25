from __future__ import annotations

from dual_stream.types import AgentConfig, TaskSpec
from experiments.baselines.common import FlatBuffer, FlatBufferEntry, append_flat, trim_flat_fifo, trim_flat_sink_recent
from experiments.baselines.flat_sink_recent import FlatSinkRecentAgent
from experiments.baselines.flat_sliding_window import FlatSlidingWindowAgent


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, context: str, model: str) -> str:
        return self.responses.pop(0)


def test_flat_fifo_trims_oldest() -> None:
    buffer = FlatBuffer(budget=5)
    append_flat(buffer, FlatBufferEntry(1, "a", "old", 3))
    append_flat(buffer, FlatBufferEntry(2, "b", "new", 3))
    trim_flat_fifo(buffer)
    assert [entry.content for entry in buffer.window] == ["new"]


def test_flat_sink_recent_keeps_sink() -> None:
    buffer = FlatBuffer(budget=7)
    append_flat(buffer, FlatBufferEntry(0, "task", "sink", 1))
    append_flat(buffer, FlatBufferEntry(1, "a", "old", 4))
    append_flat(buffer, FlatBufferEntry(2, "b", "new", 4))
    trim_flat_sink_recent(buffer)
    assert [entry.content for entry in buffer.window] == ["sink", "new"]


def test_flat_sliding_window_agent_final_answer() -> None:
    agent = FlatSlidingWindowAgent(AgentConfig(max_steps=2), FakeBackend(["<final_answer>done</final_answer>"]))
    result = agent.run("download all blog posts", TaskSpec(text="download all blog posts", key_terms=["blog posts"]))
    assert result.answer == "done"


def test_flat_sink_recent_agent_final_answer() -> None:
    agent = FlatSinkRecentAgent(AgentConfig(max_steps=2), FakeBackend(["<final_answer>done</final_answer>"]))
    result = agent.run("download all blog posts", TaskSpec(text="download all blog posts", key_terms=["blog posts"]))
    assert result.answer == "done"
