from __future__ import annotations

from agent import DualStreamAgent
from context.assembler import render_goal_stream
from core import AgentConfig, TaskSpec


class QuickstartBackend:
    def __init__(self) -> None:
        self.responses = [
            "<goal_update>enumerate all blog post URLs</goal_update>",
            "<goal_update>fetch page 1</goal_update><tool_call>http_get /blog/page-1</tool_call>",
            "<tool_call>save post 1</tool_call>",
            "<tool_call>http_get /blog/page-2</tool_call>",
            "<tool_call>save post 2</tool_call>",
            "<tool_call>http_get /blog/page-3</tool_call>",
            "<tool_call>save post 3</tool_call>",
            "<tool_call>http_get /blog/page-4</tool_call>",
            "<tool_call>save post 4</tool_call>",
            "<final_answer>downloaded all blog posts from example.com</final_answer>",
        ]
        self.index = 0

    def complete(self, context: str, model: str) -> str:
        value = self.responses[self.index]
        self.index += 1
        return value


def tool_executor(tool_call: str) -> str:
    return f"{tool_call} → 200 OK, 4200 bytes"


def summarize_stack(agent_result) -> str:
    stack = agent_result.goal_stream_snapshot.active_stack()
    return " > ".join(f'"{entry.text}"' if entry.depth else f'root: "{entry.text}"' for entry in stack)


def main() -> None:
    spec = TaskSpec(
        text="download all blog posts from example.com",
        key_terms=["blog post", "download", "example.com", "page"],
    )
    agent = DualStreamAgent(
        config=AgentConfig(max_steps=10, obs_budget=2048),
        backend=QuickstartBackend(),
        tool_executor=tool_executor,
    )
    result = agent.run(spec.text, task_spec=spec)
    entries = result.goal_stream_snapshot.entries
    print(f'[step 0] GoalStream: [root: "{entries[0].text}"]')
    if len(entries) > 1:
        print(f'[step 1] GoalStream: [root > "{entries[1].text}"]')
    if len(entries) > 2:
        print(f'[step 2] GoalStream: [root > "{entries[1].text}" > "{entries[2].text}"]')
    print(
        f"[step 2] ObsStream: {result.obs_stream_snapshot.tokens_used} tokens "
        f"(budget: {result.obs_stream_snapshot.budget})"
    )
    depth = len(result.goal_stream_snapshot.active_stack())
    print(
        f"[step {result.steps_used}] COMPLETE. GoalStream depth: {depth}. "
        f"ObsStream tokens used: {result.obs_stream_snapshot.tokens_used}/"
        f"{result.obs_stream_snapshot.budget}."
    )
    total = result.goal_stream_snapshot.token_count(agent.tokenizer) + result.obs_stream_snapshot.tokens_used
    overhead = result.goal_stream_snapshot.token_count(agent.tokenizer) / total * 100
    print(f"GoalStream overhead: {overhead:.1f}% of total context.")
    render_goal_stream(result.goal_stream_snapshot)
    summarize_stack(result)


if __name__ == "__main__":
    main()
