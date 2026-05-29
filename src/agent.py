from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from src.backends.openai import OpenAIBackend
from src.compressor.sliding_window import trim
from src.context.assembler import build_context
from src.context.goal_stream import mark_complete
from src.context.obs_stream import append_observation
from src.core import (
    AgentConfig,
    AgentResult,
    Backend,
    GoalEntry,
    GoalStream,
    ObservationEntry,
    ObservationStream,
    ParsedResponse,
    TaskSpec,
    ToolExecutor,
    VerifierResult,
)
from src.verifier.verifier import validate

LOGGER = logging.getLogger("dual_stream")
logging.basicConfig(level=os.getenv("DS_LOG_LEVEL", "WARNING"))

AGENT_PROTOCOL = """You are controlling an interactive agent loop.
Respond using ONLY these XML tags. Do not include any prose outside the tags.

<goal_update>one narrow subgoal</goal_update>
<tool_call>function_call_here</tool_call>
<mark_complete>goal_id</mark_complete>
<final_answer>answer text</final_answer>

Every opening tag MUST have a matching closing tag.
Use at least one tag per turn.

INSTRUCTIONS:
- The observation shows tool outputs from previous calls.
- In <tool_call>, write the exact function call to execute.
- Process items in order (1, 2, 3...). Do not skip items.
- Keep track of how many items you have processed using <goal_update>.
- When all items are done, call finish() or submit_answer(id) as appropriate.
- Example valid tool calls:
    get_product(5)
    record_price(5, 12.99)
    get_record(10)
    label(10, HIGH)
    check_item(3)
    submit_answer(3)
    finish()
- After calling finish() or submit_answer(), use <final_answer>done</final_answer>.
"""
SYSTEM_MARKER = "<<<DUAL_STREAM_SYSTEM>>>"
USER_MARKER = "<<<DUAL_STREAM_USER>>>"


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[int]:
        """Estimate tokens conservatively for local accounting."""
        if not text:
            return []
        estimate = max(len(text.split()), math.ceil(len(text) / 4))
        return list(range(estimate))


def extract_key_terms(text: str) -> list[str]:
    """Extract deterministic verifier terms without model calls. [INV-5]"""
    words = re.findall(r"[A-Za-z0-9_/-]+", text.lower())
    stop = {
        "a",
        "all",
        "an",
        "and",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    terms = [word for word in words if len(word) > 2 and word not in stop]
    return terms[:8]


def parse_response(response: str) -> ParsedResponse:
    """Parse exactly the XML fields supported by the agent loop."""
    # Strip Qwen3 thinking block before parsing XML tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    def tag(name: str, allow_unclosed: bool = False) -> str | None:
        match = re.search(rf"<{name}>(.*?)</{name}>", response, flags=re.DOTALL)
        if match is None and allow_unclosed:
            match = re.search(rf"<{name}>(.*?)\s*$", response, flags=re.DOTALL)
        if match is None:
            return None
        value = match.group(1).strip()
        return value or None

    return ParsedResponse(
        goal_update=tag("goal_update"),
        tool_call=tag("tool_call", allow_unclosed=True),
        final_answer=tag("final_answer"),
        mark_complete=tag("mark_complete"),
    )


def has_parsed_action(parsed: ParsedResponse) -> bool:
    return any(
        (
            parsed.goal_update,
            parsed.tool_call,
            parsed.final_answer,
            parsed.mark_complete,
        )
    )


def format_agent_prompt(context: str) -> str:
    """Prefix strict response protocol to model context. [INV-4]"""
    return f"{SYSTEM_MARKER}\n{AGENT_PROTOCOL}\n{USER_MARKER}\n{context}"


def make_rejection_notice(result: VerifierResult, step: int, tokenizer: WhitespaceTokenizer) -> ObservationEntry:
    """Create verifier rejection observations without mutating GoalStream. [INV-2]"""
    reason = result.reason or "Verifier rejected proposed goal."
    content = f"[VERIFIER REJECTED] {reason} | Check: {result.failed_check}"
    return ObservationEntry(
        step=step,
        role="rejection_notice",
        content=content,
        tokens=len(tokenizer.encode(reason)),
    )


class DualStreamAgent:
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
        """Run the dual-stream loop. [INV-1, INV-2, INV-4, INV-6]"""
        spec = task_spec or TaskSpec(
            text=task_description,
            key_terms=extract_key_terms(task_description),
        )
        goal_stream = GoalStream(spec=spec, max_depth=self.config.goal_max_depth)
        obs_stream = ObservationStream(budget=self.config.obs_budget)
        step = 0
        verifier_rejections = 0
        log_path = self._log_path(spec.id)

        goal_stream.entries.append(
            GoalEntry(text=spec.text, depth=0, step_created=0)
        )
        self._log_step(log_path, step, goal_stream, obs_stream, "init", None, False, None)

        while step < self.config.max_steps:
            context = build_context(goal_stream, obs_stream)
            response = self.backend.complete(
                format_agent_prompt(context), model=self.config.model
            )
            step += 1

            parsed = parse_response(response)
            if not has_parsed_action(parsed):
                entry = ObservationEntry(
                    step=step,
                    role="rejection_notice",
                    content="[FORMAT ERROR] Expected one of goal_update, tool_call, final_answer, mark_complete tags.",
                    tokens=12,
                )
                append_observation(obs_stream, entry)
                trim(obs_stream)
                self._log_step(log_path, step, goal_stream, obs_stream, "format_error", None, False, None)
                continue

            verifier_called = False
            verifier_result: VerifierResult | None = None
            if parsed.goal_update is not None:
                stack = goal_stream.active_stack()
                parent = stack[-1]
                proposed = GoalEntry(
                    text=parsed.goal_update,
                    parent_id=parent.id,
                    depth=len(stack),
                    step_created=step,
                )
                verifier_called = True
                verifier_result = validate(
                    proposed, goal_stream, spec, enabled=self.config.verifier_enabled
                )
                if verifier_result.valid:
                    goal_stream.entries.append(proposed)
                else:
                    verifier_rejections += 1
                    append_observation(
                        obs_stream,
                        make_rejection_notice(verifier_result, step, self.tokenizer),
                    )
                    trim(obs_stream)

            if parsed.mark_complete is not None:
                mark_complete(goal_stream, parsed.mark_complete, step)

            final_tool_call = None
            if parsed.tool_call is not None:
                final_tool_call = parsed.tool_call
                # Handle WebArena stop action: stop [answer] → final_answer
                stop_match = re.match(r"^stop\s*\[?(.*?)\]?\s*$", parsed.tool_call.strip(), re.DOTALL)
                if stop_match and parsed.final_answer is None:
                    answer = stop_match.group(1).strip()
                    return AgentResult(
                        answer=answer,
                        steps_used=step,
                        goal_stream_snapshot=goal_stream,
                        obs_stream_snapshot=obs_stream,
                        timed_out=False,
                        verifier_rejections=verifier_rejections,
                        final_tool_call=final_tool_call,
                    )
                output = self.tool_executor(parsed.tool_call)
                append_observation(
                    obs_stream,
                    ObservationEntry(
                        step=step,
                        role="tool_output",
                        content=output,
                        tokens=len(self.tokenizer.encode(output)),
                    ),
                )
                trim(obs_stream)

            self._log_step(
                log_path,
                step,
                goal_stream,
                obs_stream,
                "tool_call" if parsed.tool_call else "llm_response",
                parsed.tool_call,
                verifier_called,
                verifier_result,
            )

            if parsed.final_answer is not None:
                return AgentResult(
                    answer=parsed.final_answer,
                    steps_used=step,
                    goal_stream_snapshot=goal_stream,
                    obs_stream_snapshot=obs_stream,
                    timed_out=False,
                    verifier_rejections=verifier_rejections,
                    final_tool_call=final_tool_call,
                )

        return AgentResult(
            answer=None,
            steps_used=step,
            goal_stream_snapshot=goal_stream,
            obs_stream_snapshot=obs_stream,
            timed_out=True,
            verifier_rejections=verifier_rejections,
        )

    @staticmethod
    def default_tool_executor(tool_call: str) -> str:
        """Return deterministic local tool output for tests and dry runs."""
        return f"Executed tool call: {tool_call}"

    def _log_path(self, task_id: str) -> Path | None:
        if self.config.log_dir is None:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.config.log_dir / f"local_dual_stream_{self.config.max_steps}_{task_id}_{timestamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _log_step(
        self,
        path: Path | None,
        step: int,
        goal_stream: GoalStream,
        obs_stream: ObservationStream,
        action: str,
        tool: str | None,
        verifier_called: bool,
        verifier_result: VerifierResult | None,
    ) -> None:
        """Write one JSONL runtime record. [INV-4, INV-6]"""
        if path is None:
            return
        goal_tokens = goal_stream.token_count(self.tokenizer)
        record = {
            "step": step,
            "goal_stream_depth": max((entry.depth for entry in goal_stream.entries), default=0),
            "goal_stream_tokens": goal_tokens,
            "obs_stream_tokens": obs_stream.tokens_used,
            "total_context_tokens": goal_tokens + obs_stream.tokens_used,
            "action": action,
            "tool": tool,
            "verifier_called": verifier_called,
            "verifier_result": None
            if verifier_result is None
            else {
                "valid": verifier_result.valid,
                "failed_check": verifier_result.failed_check,
                "reason": verifier_result.reason,
            },
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
