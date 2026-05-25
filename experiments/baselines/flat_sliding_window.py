from __future__ import annotations

from dual_stream.agent import (
    WhitespaceTokenizer,
    extract_key_terms,
    format_agent_prompt,
    has_parsed_action,
    parse_response,
)
from dual_stream.backends.openai import OpenAIBackend
from dual_stream.types import AgentConfig, AgentResult, Backend, TaskSpec, ToolExecutor
from experiments.baselines.common import (
    FlatBuffer,
    FlatBufferEntry,
    append_flat,
    empty_result,
    render_flat_context,
    trim_flat_fifo,
)


class FlatSlidingWindowAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        backend: Backend | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.backend = backend or OpenAIBackend()
        self.tool_executor = tool_executor or self.default_tool_executor
        self.tokenizer = WhitespaceTokenizer()

    def run(self, task_description: str, task_spec: TaskSpec | None = None) -> AgentResult:
        """Run flat FIFO baseline loop. [INV-3]"""
        spec = task_spec or TaskSpec(
            text=task_description,
            key_terms=extract_key_terms(task_description),
        )
        buffer = FlatBuffer(budget=self.config.obs_budget + 256)
        step = 0
        append_flat(
            buffer,
            FlatBufferEntry(
                step=0,
                role="task",
                content=spec.text,
                tokens=len(self.tokenizer.encode(spec.text)),
            ),
        )
        trim_flat_fifo(buffer)

        while step < self.config.max_steps:
            context = render_flat_context(buffer)
            response = self.backend.complete(format_agent_prompt(context), model=self.config.model)
            step += 1

            parsed = parse_response(response)
            if not has_parsed_action(parsed):
                append_flat(buffer, FlatBufferEntry(step, "format_error", "[FORMAT ERROR] Expected supported tags.", 6))
                trim_flat_fifo(buffer)
                continue

            if parsed.goal_update is not None:
                append_flat(
                    buffer,
                    FlatBufferEntry(
                        step=step,
                        role="goal_update",
                        content=parsed.goal_update,
                        tokens=len(self.tokenizer.encode(parsed.goal_update)),
                    ),
                )
                trim_flat_fifo(buffer)

            if parsed.mark_complete is not None:
                append_flat(buffer, FlatBufferEntry(step, "mark_complete", parsed.mark_complete, 1))
                trim_flat_fifo(buffer)

            if parsed.tool_call is not None:
                output = self.tool_executor(parsed.tool_call)
                append_flat(
                    buffer,
                    FlatBufferEntry(
                        step=step,
                        role="tool_output",
                        content=output,
                        tokens=len(self.tokenizer.encode(output)),
                    ),
                )
                trim_flat_fifo(buffer)

            if parsed.final_answer is not None:
                return empty_result(parsed.final_answer, step, False, spec)

        return empty_result(None, step, True, spec)

    @staticmethod
    def default_tool_executor(tool_call: str) -> str:
        """Return deterministic local tool output for tests and dry runs."""
        return f"Executed tool call: {tool_call}"
