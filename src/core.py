from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
import uuid


@dataclass
class TaskSpec:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    key_terms: list[str] = field(default_factory=list)


@dataclass
class GoalEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    parent_id: str | None = None
    status: Literal["active", "completed", "failed"] = "active"
    depth: int = 0
    step_created: int = 0
    step_resolved: int | None = None


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        """Return token ids for text."""


@dataclass
class GoalStream:
    entries: list[GoalEntry] = field(default_factory=list)
    max_depth: int = 8
    spec: TaskSpec | None = None

    def active_stack(self) -> list[GoalEntry]:
        """Return active root-to-leaf chain. [INV-6: entries are never removed.]"""
        active = [entry for entry in self.entries if entry.status == "active"]
        return sorted(active, key=lambda entry: entry.depth)

    def token_count(self, tokenizer: Tokenizer) -> int:
        """Return rendered goal token count. [INV-6: counts full append-only history.]"""
        rendered = "\n".join(
            f"{entry.depth}:{entry.status}:{entry.text}" for entry in self.entries
        )
        return len(tokenizer.encode(rendered))


@dataclass
class ObservationEntry:
    step: int
    role: Literal["tool_output", "environment", "rejection_notice"]
    content: str
    tokens: int


@dataclass
class ObservationStream:
    window: deque[ObservationEntry] = field(default_factory=deque)
    budget: int = 2048
    tokens_used: int = 0


@dataclass
class VerifierResult:
    valid: bool
    failed_check: str | None = None
    reason: str | None = None


@dataclass
class AgentConfig:
    model: str = "openai/gpt-5.5"
    obs_budget: int = 2048
    goal_max_depth: int = 8
    max_steps: int = 200
    seed: int = 42
    backend: Literal["openai"] = "openai"
    count_rejections_as_steps: bool = True
    verifier_enabled: bool = True
    log_dir: Path | None = None


@dataclass
class ParsedResponse:
    goal_update: str | None = None
    tool_call: str | None = None
    final_answer: str | None = None
    mark_complete: str | None = None


@dataclass
class AgentResult:
    answer: str | None
    steps_used: int
    goal_stream_snapshot: GoalStream
    obs_stream_snapshot: ObservationStream
    timed_out: bool = False
    verifier_rejections: int = 0
    final_tool_call: str | None = None


@dataclass
class BenchmarkTask:
    id: str
    instruction: str
    key_terms: list[str]
    expected_answer: str
    expected_tool_call: str | None = None
    category: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


ToolExecutor = Callable[[str], str]


class Backend(Protocol):
    def complete(self, context: str, model: str) -> str:
        """Return one model completion for the assembled context."""
