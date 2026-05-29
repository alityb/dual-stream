from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from benchmarks.appworld.harness import AppWorldHarness
from benchmarks.base import BenchmarkHarness
from benchmarks.tau_bench.harness import TauBenchHarness
from benchmarks.webarena.harness import WebArenaHarness
from src.agent import DualStreamAgent, WhitespaceTokenizer
from src.core import AgentConfig, AgentResult, Backend, BenchmarkTask, TaskSpec
from experiments import config
from experiments.baselines.flat_sink_recent import FlatSinkRecentAgent
from experiments.baselines.flat_sliding_window import FlatSlidingWindowAgent

BackendFactory = Callable[[], Backend]


def clean_results(results_dir: Path) -> None:
    """Remove generated JSON result files before reproduction. [INV-4]"""
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.glob("*.json"):
        path.unlink()


def harness_for(benchmark: str) -> BenchmarkHarness:
    """Instantiate the configured real benchmark harness."""
    if benchmark == "appworld":
        return AppWorldHarness()
    if benchmark == "webarena":
        return WebArenaHarness()
    if benchmark == "tau_bench":
        return TauBenchHarness()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def agent_for(
    condition: str,
    agent_config: AgentConfig,
    backend: Backend,
    tool_executor: Callable[[str], str] | None,
):
    """Construct the correct agent for one condition. [INV-3]"""
    if condition == "dual_stream":
        return DualStreamAgent(agent_config, backend=backend, tool_executor=tool_executor)
    if condition == "verifier_off":
        disabled = AgentConfig(
            model=agent_config.model,
            obs_budget=agent_config.obs_budget,
            goal_max_depth=agent_config.goal_max_depth,
            max_steps=agent_config.max_steps,
            seed=agent_config.seed,
            backend=agent_config.backend,
            count_rejections_as_steps=agent_config.count_rejections_as_steps,
            verifier_enabled=False,
            log_dir=agent_config.log_dir,
        )
        return DualStreamAgent(disabled, backend=backend, tool_executor=tool_executor)
    if condition == "flat_sliding_window":
        return FlatSlidingWindowAgent(agent_config, backend=backend, tool_executor=tool_executor)
    if condition == "flat_sink_recent":
        return FlatSinkRecentAgent(agent_config, backend=backend, tool_executor=tool_executor)
    raise ValueError(f"Unknown condition: {condition}")


def run_task(
    harness: BenchmarkHarness,
    benchmark: str,
    condition: str,
    task: BenchmarkTask,
    step_budget: int,
    backend_factory: BackendFactory | None = None,
) -> tuple[AgentResult, float]:
    """Run one real task and score it with its harness."""
    agent_config = AgentConfig(
        model=config.MODEL,
        obs_budget=config.OBS_BUDGETS[benchmark],
        max_steps=step_budget,
        seed=config.SEED,
        verifier_enabled=condition != "verifier_off",
    )
    backend = backend_factory() if backend_factory is not None else _default_backend()
    spec = harness.extract_task_spec(task)
    experiment_name = f"dual_stream_{condition}_{step_budget}"
    session = None
    tool_executor = None
    if hasattr(harness, "make_tool_executor"):
        tool_executor = harness.make_tool_executor(task)
    if hasattr(harness, "open_session"):
        session = getattr(harness, "open_session")(task, experiment_name)
        tool_executor = session.execute
    agent = agent_for(condition, agent_config, backend, tool_executor)
    try:
        result = agent.run(spec.text, task_spec=spec)
        if session is not None:
            # Pass agent's final answer so string_match evaluation works correctly
            task.metadata["last_score"] = str(session.evaluate(answer=result.answer))
        score = harness.score(task, result)
        return result, score
    finally:
        if session is not None:
            session.close()


def _default_backend() -> Backend:
    from backends.openai import OpenAIBackend

    return OpenAIBackend()


def build_result(
    benchmark: str,
    condition: str,
    step_budget: int,
    tasks: int,
    backend_factory: BackendFactory | None = None,
) -> dict:
    """Run tasks and build one result JSON object matching the schema. [INV-4]"""
    harness = harness_for(benchmark)
    harness.setup()
    tokenizer = WhitespaceTokenizer()
    rows = []
    results: list[AgentResult] = []
    try:
        loaded_tasks = harness.load_tasks(tasks)
        for task in loaded_tasks:
            result, score = run_task(
                harness,
                benchmark,
                condition,
                task,
                step_budget,
                backend_factory=backend_factory,
            )
            results.append(result)
            rows.append(
                {
                    "task_id": task.id,
                    "passed": bool(score == 1.0),
                    "steps_used": result.steps_used,
                    "goal_stream_depth_max": max(
                        (entry.depth for entry in result.goal_stream_snapshot.entries), default=0
                    ),
                    "verifier_rejections": result.verifier_rejections,
                }
            )
    finally:
        harness.teardown()

    n_tasks = len(rows)
    if n_tasks == 0:
        raise RuntimeError(f"No tasks loaded for benchmark {benchmark}")
    goal_tokens = [result.goal_stream_snapshot.token_count(tokenizer) for result in results]
    obs_tokens = [result.obs_stream_snapshot.tokens_used for result in results]
    total_tokens = [goal + obs for goal, obs in zip(goal_tokens, obs_tokens)]
    mean_total = sum(total_tokens) / n_tasks
    mean_goal = sum(goal_tokens) / n_tasks
    total_steps = sum(row["steps_used"] for row in rows)
    total_rejections = sum(row["verifier_rejections"] for row in rows)
    return {
        "benchmark": benchmark,
        "condition": condition,
        "step_budget": step_budget,
        "model": config.MODEL,
        "seed": config.SEED,
        "obs_budget": config.OBS_BUDGETS[benchmark],
        "n_tasks": n_tasks,
        "completion_rate": round(sum(1 for row in rows if row["passed"]) / n_tasks, 3),
        "mean_steps_used": round(total_steps / n_tasks, 2),
        "mean_goal_stream_tokens": round(mean_goal, 2),
        "mean_total_context_tokens": round(mean_total, 2),
        "goal_stream_overhead_pct": round((mean_goal / mean_total * 100) if mean_total else 0.0, 2),
        "verifier_rejection_rate": round(
            (total_rejections / total_steps)
            if condition == "dual_stream" and total_steps
            else 0.0,
            3,
        ),
        "tasks": rows,
    }


def write_sweep(
    tasks: int,
    results_dir: Path,
    backend_factory: BackendFactory | None = None,
    benchmarks: list[str] | None = None,
    conditions: list[str] | None = None,
    step_budgets: list[int] | None = None,
) -> list[Path]:
    """Write all condition, budget, and benchmark result files. [INV-4]"""
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    selected_benchmarks = benchmarks or config.BENCHMARKS
    selected_conditions = conditions or config.CONDITIONS
    selected_step_budgets = step_budgets or config.STEP_BUDGETS
    for benchmark in selected_benchmarks:
        for condition in selected_conditions:
            for step_budget in selected_step_budgets:
                print(
                    f"[sweep] start benchmark={benchmark} condition={condition} N={step_budget} tasks={tasks}",
                    flush=True,
                )
                result = build_result(
                    benchmark,
                    condition,
                    step_budget,
                    tasks,
                    backend_factory=backend_factory,
                )
                path = results_dir / f"{benchmark}_{condition}_{step_budget}.json"
                path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(f"[sweep] wrote {path}", flush=True)
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=config.TASKS_PER_BENCHMARK)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--conditions", nargs="+", choices=config.CONDITIONS)
    parser.add_argument("--benchmarks", nargs="+", choices=config.BENCHMARKS)
    parser.add_argument("--step-budgets", nargs="+", type=int, choices=config.STEP_BUDGETS)
    parser.add_argument("--clean-results-only", action="store_true")
    args = parser.parse_args()
    if args.clean_results_only:
        clean_results(args.results_dir)
        return
    write_sweep(
        args.tasks,
        args.results_dir,
        benchmarks=args.benchmarks,
        conditions=args.conditions,
        step_budgets=args.step_budgets,
    )


if __name__ == "__main__":
    main()
