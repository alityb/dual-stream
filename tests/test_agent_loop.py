from __future__ import annotations

from src.agent import DualStreamAgent, format_agent_prompt, parse_response
from src.core import AgentConfig, ParsedResponse, TaskSpec


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.contexts: list[str] = []

    def complete(self, context: str, model: str) -> str:
        self.contexts.append(context)
        return self.responses.pop(0)


def test_parse_response_extracts_supported_tags() -> None:
    parsed = parse_response(
        "<goal_update>fetch blog posts</goal_update><tool_call>http_get</tool_call>"
    )
    assert parsed == ParsedResponse(goal_update="fetch blog posts", tool_call="http_get")


def test_parse_response_accepts_terminal_unclosed_tool_call() -> None:
    parsed = parse_response(
        "<goal_update>inspect cart</goal_update>\n<tool_call>print('x')"
    )
    assert parsed.goal_update == "inspect cart"
    assert parsed.tool_call == "print('x')"


def test_format_agent_prompt_includes_xml_contract() -> None:
    prompt = format_agent_prompt("context")
    assert "<tool_call>" in prompt
    assert "WEBARENA" in prompt
    assert prompt.endswith("context")


def test_agent_bootstraps_root_and_final_answer() -> None:
    backend = FakeBackend(["<final_answer>done</final_answer>"])
    agent = DualStreamAgent(config=AgentConfig(max_steps=3), backend=backend)
    result = agent.run("download all blog posts", TaskSpec(text="download all blog posts", key_terms=["blog posts"]))
    assert result.answer == "done"
    assert result.steps_used == 1
    assert result.goal_stream_snapshot.entries[0].text == "download all blog posts"


def test_agent_verified_goal_update_and_tool_call() -> None:
    backend = FakeBackend(
        [
            "<goal_update>fetch blog posts page 1</goal_update><tool_call>http_get /blog/1</tool_call>",
            "<final_answer>done</final_answer>",
        ]
    )
    agent = DualStreamAgent(
        config=AgentConfig(max_steps=3),
        backend=backend,
        tool_executor=lambda call: f"output for {call}",
    )
    spec = TaskSpec(text="download all blog posts", key_terms=["blog posts"])
    result = agent.run(spec.text, spec)
    assert len(result.goal_stream_snapshot.entries) == 2
    assert result.obs_stream_snapshot.tokens_used > 0
    assert "[GOAL STATE]" in backend.contexts[0]


def test_agent_rejection_does_not_append_goal() -> None:
    backend = FakeBackend(
        [
            "<goal_update>buy groceries</goal_update>",
            "<final_answer>done</final_answer>",
        ]
    )
    agent = DualStreamAgent(config=AgentConfig(max_steps=3), backend=backend)
    spec = TaskSpec(text="download all blog posts", key_terms=["blog posts"])
    result = agent.run(spec.text, spec)
    assert len(result.goal_stream_snapshot.entries) == 1
    assert result.verifier_rejections == 1
    assert result.obs_stream_snapshot.window[0].role == "rejection_notice"


def test_parse_failure_counts_as_step_and_injects_notice() -> None:
    backend = FakeBackend(["unparseable", "<final_answer>done</final_answer>"])
    agent = DualStreamAgent(config=AgentConfig(max_steps=3), backend=backend)
    result = agent.run("download all blog posts", TaskSpec(text="download all blog posts", key_terms=["blog posts"]))
    assert result.steps_used == 2
    assert result.obs_stream_snapshot.window[0].content.startswith("[FORMAT ERROR]")


def test_agent_times_out() -> None:
    backend = FakeBackend(["<tool_call>x</tool_call>", "<tool_call>y</tool_call>"])
    agent = DualStreamAgent(config=AgentConfig(max_steps=2), backend=backend)
    result = agent.run("download all blog posts", TaskSpec(text="download all blog posts", key_terms=["blog posts"]))
    assert result.timed_out is True
    assert result.steps_used == 2
