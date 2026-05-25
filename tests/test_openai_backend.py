from __future__ import annotations

from agent import format_agent_prompt
from backends.openai import OpenAIBackend


def test_openai_backend_splits_system_marker() -> None:
    backend = OpenAIBackend(api_key="test")
    messages = backend._messages(format_agent_prompt("user context"))
    assert messages[0]["role"] == "system"
    assert "Every opening tag MUST have its matching closing tag." in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "user context"}
