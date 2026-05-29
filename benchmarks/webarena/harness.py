from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkHarness
from src.core import AgentResult, BenchmarkTask, TaskSpec

# URL placeholders → resolved at runtime from .env / environment
_SITE_MAP: dict[str, str] = {
    "__SHOPPING__": lambda: _env("SHOPPING", "http://localhost:7770"),
    "__SHOPPING_ADMIN__": lambda: _env("SHOPPING_ADMIN", "http://localhost:7780/admin"),
    "__REDDIT__": lambda: _env("REDDIT", "http://localhost:9999"),
    "__GITLAB__": lambda: _env("GITLAB", "http://localhost:8023"),
    "__MAP__": lambda: _env("MAP", "http://localhost:3000"),
    "__WIKIPEDIA__": lambda: _env("WIKIPEDIA", "http://localhost:8888"),
    "__HOMEPAGE__": lambda: _env("HOMEPAGE", "http://localhost"),
}

# Sites available on this instance (skip map/wikipedia — not pulled)
_SUPPORTED_SITES = {"shopping", "shopping_admin", "reddit", "gitlab"}


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    return default


def _load_webarena_env() -> None:
    """Set required environment variables for WebArena evaluator import."""
    os.environ.setdefault("SHOPPING", _env("SHOPPING", "http://localhost:7770"))
    os.environ.setdefault("SHOPPING_ADMIN", _env("SHOPPING_ADMIN", "http://localhost:7780/admin"))
    os.environ.setdefault("REDDIT", _env("REDDIT", "http://localhost:9999"))
    os.environ.setdefault("GITLAB", _env("GITLAB", "http://localhost:8023"))
    os.environ.setdefault("MAP", _env("MAP", "http://localhost:3000"))
    os.environ.setdefault("WIKIPEDIA", _env("WIKIPEDIA", "http://localhost:8888"))
    os.environ.setdefault("HOMEPAGE", _env("HOMEPAGE", "http://localhost"))


def _resolve_url(url: str) -> str:
    for placeholder, resolver in _SITE_MAP.items():
        if placeholder in url:
            url = url.replace(placeholder, resolver())
    return url


def _task_to_benchmark(raw: dict[str, Any], webarena_root: Path) -> BenchmarkTask | None:
    """Convert a raw WebArena task dict to a BenchmarkTask, or None if unsupported."""
    sites = set(raw.get("sites", []))
    if not sites.issubset(_SUPPORTED_SITES):
        return None
    task_id = raw["task_id"]
    config_file = webarena_root / "config_files" / f"{task_id}.json"
    if not config_file.exists():
        return None
    intent = raw.get("intent", "")
    terms = [w for w in intent.lower().split() if len(w) > 2][:8]
    if not terms:
        terms = sorted(sites)
    return BenchmarkTask(
        id=str(task_id),
        instruction=intent,
        key_terms=terms,
        expected_answer="",
        category=",".join(sorted(sites)),
        metadata={
            "config_file": str(config_file),
            "start_url": _resolve_url(raw.get("start_url", "")),
            "sites": ",".join(sorted(sites)),
            "eval_types": ",".join(raw.get("eval", {}).get("eval_types", [])),
        },
    )


class WebArenaHarness(BenchmarkHarness):
    def __init__(self, webarena_path: str | None = None) -> None:
        self.webarena_path = Path(webarena_path or _env("WEBARENA_PATH", ""))
        self._tasks: list[BenchmarkTask] = []

    def setup(self) -> None:
        """Set env vars and verify WebArena path is accessible."""
        _load_webarena_env()
        if self.webarena_path.exists():
            sys.path.insert(0, str(self.webarena_path))

    def teardown(self) -> None:
        return None

    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Load supported WebArena tasks ordered from easiest to hardest."""
        if self._tasks:
            return self._tasks if n is None else self._tasks[:n]

        raw_path = self.webarena_path / "config_files" / "test.raw.json"
        if not raw_path.exists():
            return self._fixture_tasks(n)

        raw_tasks = json.loads(raw_path.read_text())
        tasks = []
        for raw in raw_tasks:
            task = _task_to_benchmark(raw, self.webarena_path)
            if task is not None:
                tasks.append(task)

        def _difficulty_key(t: BenchmarkTask) -> int:
            """Lower = easier. url_match shopping first, string_match admin last."""
            et = t.metadata.get("eval_types", "")
            sites = t.metadata.get("sites", "")
            if "url_match" in et and "shopping_admin" not in sites and "gitlab" not in sites:
                return 0  # easiest: url_match shopping/reddit
            if "string_match" in et and "shopping" in sites and "shopping_admin" not in sites:
                return 1  # string_match shopping
            if "url_match" in et:
                return 2  # url_match other
            if "shopping_admin" not in sites and "gitlab" not in sites:
                return 3  # other sites
            return 4  # admin/gitlab (hardest)

        tasks.sort(key=_difficulty_key)
        self._tasks = tasks
        return tasks if n is None else tasks[:n]

    def _fixture_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        """Deterministic fixture fallback when WebArena is not set up."""
        tasks = [
            BenchmarkTask(
                f"webarena-fixture-{i}",
                f"navigate shop site to target state {i}",
                ["shop", "target", "state"],
                f"target reached {i}",
                category="navigation",
            )
            for i in range(1, 51)
        ]
        return tasks if n is None else tasks[:n]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        """Extract TaskSpec from WebArena task. Use category as key_term fallback."""
        terms = task.key_terms or ([task.category] if task.category else [])
        return TaskSpec(id=task.id, text=task.instruction, key_terms=terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Score task using official WebArena evaluator when live, fixture otherwise."""
        # Fixture tasks: check answer string or observation
        if task.id.startswith("webarena-fixture-"):
            return _fixture_score(task, result)
        # Live tasks: check stored last_score from open_session
        return float(task.metadata.get("last_score", "0.0"))

    def open_session(self, task: BenchmarkTask, experiment_name: str) -> "WebArenaSession":
        """Open a live browser session for a WebArena task."""
        _load_webarena_env()
        if self.webarena_path.exists():
            sys.path.insert(0, str(self.webarena_path))
        return WebArenaSession(task, self.webarena_path)

    def make_tool_executor(self, task: BenchmarkTask):
        """Return a tool executor for fixture tasks."""
        if task.id.startswith("webarena-fixture-"):
            return _fixture_executor(task)
        return None  # live tasks use open_session


def _fixture_score(task: BenchmarkTask, result: AgentResult) -> float:
    expected = task.expected_answer.lower()
    if result.answer and expected in result.answer.lower():
        return 1.0
    for entry in result.obs_stream_snapshot.window:
        if entry.role == "tool_output" and expected in entry.content.lower():
            return 1.0
    return 0.0


def _fixture_executor(task: BenchmarkTask):
    expected = task.expected_answer
    call_count = [0]

    def executor(tool_call: str) -> str:
        call_count[0] += 1
        if (
            call_count[0] >= 3
            or "navigate" in tool_call.lower()
            or "target" in tool_call.lower()
        ):
            return f"Navigation successful. Result: {expected}"
        return f"Executed: {tool_call[:200]}"

    return executor


class WebArenaSession:
    """Live browser session wrapping ScriptBrowserEnv."""

    def __init__(self, task: BenchmarkTask, webarena_root: Path) -> None:
        self.task = task
        self.webarena_root = webarena_root
        self._env: Any = None
        self._trajectory: list = []
        self._obs: str = ""
        self._score: float = 0.0
        self._config_file = task.metadata.get("config_file", "")
        self._setup()

    def _setup(self) -> None:
        try:
            import os
            # Must run from webarena directory so relative .auth/ paths resolve
            if str(self.webarena_root) not in str(os.getcwd()):
                os.chdir(str(self.webarena_root))
            from browser_env import ScriptBrowserEnv, StateInfo, create_stop_action
            env = ScriptBrowserEnv(
                headless=True,
                slow_mo=0,
                observation_type="accessibility_tree",
                current_viewport_only=True,
                viewport_size={"width": 1280, "height": 720},
            )
            obs, info = env.reset(options={"config_file": self._config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            self._trajectory.append(state_info)
            self._obs = obs.get("text", "")
            self._env = env
        except Exception as exc:
            self._obs = f"[BROWSER INIT ERROR] {exc}"

    def execute(self, tool_call: str) -> str:
        """Execute a WebArena action string and return the new accessibility tree."""
        if self._env is None:
            return self._obs
        try:
            from browser_env import StateInfo, create_id_based_action, create_stop_action
            import re as _re
            tool_call = tool_call.strip()
            # Normalise common model output variants
            tool_call = _re.sub(r'^action\s+', '', tool_call)          # "action goto" → "goto"
            # Normalise goto URL into goto[URL] format WebArena expects
            # Handle: goto URL, goto [URL], goto[URL]
            goto_m = _re.match(r'^goto\s*\[?\s*(https?://[^\]\s]+)\s*\]?', tool_call)
            if goto_m:
                tool_call = f'goto[{goto_m.group(1)}]'
            navigate_m = _re.match(r'^navigate\s+(https?://\S+)', tool_call)
            if navigate_m:
                tool_call = f'goto[{navigate_m.group(1)}]'
            if not tool_call or tool_call.lower() in ("stop", "done", "complete"):
                action = create_stop_action("")
            else:
                action = create_id_based_action(tool_call)
            self._trajectory.append(action)
            obs, _, terminated, _, info = self._env.step(action)
            state_info: StateInfo = {"observation": obs, "info": info}
            self._trajectory.append(state_info)
            self._obs = obs.get("text", "")
            if terminated:
                return self._obs + "\n[PAGE TERMINATED]"
            return self._obs
        except Exception as exc:
            return f"[ACTION ERROR] {exc}\n{self._obs}"

    def evaluate(self, answer: str | None = None) -> float:
        """Run the official WebArena evaluator and return 0.0 or 1.0."""
        if self._env is None:
            return 0.0
        try:
            # Ensure env vars are set before importing evaluator
            _load_webarena_env()
            from browser_env import create_stop_action, StateInfo
            from evaluation_harness.evaluators import evaluator_router
            # Create stop action with the agent's answer (required for string_match eval)
            stop = create_stop_action(answer or "")
            self._trajectory.append(stop)
            evaluator = evaluator_router(self._config_file)
            score = evaluator(
                trajectory=self._trajectory,
                config_file=self._config_file,
                page=self._env.page,
                client=self._env.get_page_client(self._env.page),
            )
            self._score = float(score)
            self.task.metadata["last_score"] = str(self._score)
            return self._score
        except Exception as exc:
            self.task.metadata["last_score"] = "0.0"
            return 0.0

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
