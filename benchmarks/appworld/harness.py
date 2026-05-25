from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from benchmarks.base import BenchmarkHarness
from agent import extract_key_terms
from core import AgentResult, BenchmarkTask, TaskSpec


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name and raw_value.strip():
            return raw_value.strip().strip('"').strip("'")
    return None


@dataclass
class AppWorldSession:
    task_id: str
    experiment_name: str
    world: Any

    def execute(self, code: str) -> str:
        """Execute one AppWorld code block as a tool call."""
        return str(self.world.execute(_strip_code_fences(code)))

    def completed(self) -> bool:
        """Return whether the AppWorld supervisor marked completion."""
        return bool(self.world.task_completed())

    def evaluate(self) -> float:
        """Run AppWorld state-based evaluator and return binary success."""
        tracker = self.world.evaluate()
        return 1.0 if tracker.success else 0.0

    def close(self) -> None:
        """Close the AppWorld task world."""
        self.world.close()


class AppWorldHarness(BenchmarkHarness):
    def __init__(self, dataset_name: str | None = None, root: str | None = None) -> None:
        self.dataset_name = dataset_name or _env_value("APPWORLD_DATASET") or "test_challenge"
        self.root = root or _env_value("APPWORLD_ROOT")

    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Load AppWorld task ids from the configured dataset."""
        self._configure_root()
        from appworld import AppWorld, load_task_ids

        tasks: list[BenchmarkTask] = []
        for task_id in load_task_ids(self.dataset_name)[:n]:
            with AppWorld(task_id=task_id, experiment_name="dual_stream_load") as world:
                instruction = str(world.task.instruction)
                metadata = dict(world.task.ground_truth.metadata)
                app_names = ",".join(sorted(world.task.app_descriptions.keys()))
                api_summary = _api_summary(world.task.app_descriptions, world.task.api_docs)
            tasks.append(
                BenchmarkTask(
                    id=task_id,
                    instruction=instruction,
                    key_terms=extract_key_terms(f"{instruction} {app_names}"),
                    expected_answer="",
                    category=app_names,
                    metadata={key: str(value) for key, value in metadata.items()}
                    | {
                        "dataset": self.dataset_name,
                        "apps": app_names,
                        "api_summary": api_summary,
                    },
                )
            )
        return tasks

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        """Extract TaskSpec from AppWorld instruction and app names."""
        text = f"Apps: {task.category}\n\nTask: {task.instruction}"
        terms = task.key_terms or extract_key_terms(text)
        return TaskSpec(id=task.id, text=text, key_terms=terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Score from an AppWorld evaluation result stored in task metadata."""
        return float(task.metadata.get("last_score", "0.0"))

    def open_session(self, task: BenchmarkTask, experiment_name: str) -> AppWorldSession:
        """Open an AppWorld task world for an agent run."""
        self._configure_root()
        from appworld import AppWorld

        world = AppWorld(task_id=task.id, experiment_name=experiment_name)
        return AppWorldSession(task_id=task.id, experiment_name=experiment_name, world=world)

    def setup(self) -> None:
        """Verify that AppWorld is installed and task data is available."""
        self._configure_root()
        from appworld import load_task_ids

        if not load_task_ids(self.dataset_name):
            raise RuntimeError(f"No AppWorld tasks found for dataset {self.dataset_name}")

    def teardown(self) -> None:
        """No persistent AppWorld resources are held by the harness."""
        return None

    def _configure_root(self) -> None:
        if self.root:
            os.environ["APPWORLD_ROOT"] = self.root


def _api_summary(app_descriptions: dict[str, str], api_docs: Any) -> str:
    lines = ["Available AppWorld APIs:"]
    for app_name in sorted(app_descriptions):
        app_doc = getattr(api_docs, app_name)
        api_names = [name for name in dir(app_doc) if not name.startswith("_")]
        lines.append(f"- {app_name}: {app_descriptions[app_name]}")
        lines.append(f"  APIs: {', '.join(api_names)}")
    return "\n".join(lines)


def _strip_code_fences(code: str) -> str:
    text = code.strip()
    match = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text
