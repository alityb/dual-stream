from __future__ import annotations

import argparse

from benchmarks.webarena.harness import WebArenaHarness


def validate_known_passing() -> int:
    """Validate WebArena harness integration: browser opens, evaluator runs."""
    harness = WebArenaHarness()
    harness.setup()
    tasks = harness.load_tasks(5)
    if not tasks:
        print("No tasks loaded")
        return 1

    webarena_live = harness.webarena_path.exists() and not tasks[0].id.startswith(
        "webarena-fixture-"
    )
    if webarena_live:
        # Live validation: confirm browser opens and evaluator runs for 5 tasks
        passed = 0
        for task in tasks[:5]:
            print(f"  Task {task.id}: {task.instruction[:60]}...")
            try:
                session = harness.open_session(task, "validation")
                env_ok = session._env is not None
                obs_ok = bool(session._obs and len(session._obs) > 10)
                # Run evaluator to confirm it executes without crashing
                # (score will be 0 since we gave no actions — that's expected)
                score = session.evaluate()
                session.close()
                if env_ok and obs_ok:
                    passed += 1
                    print(f"    browser=ok obs=ok evaluator_ran=ok score={score:.1f}")
                else:
                    print(f"    FAIL env={env_ok} obs={obs_ok}")
            except Exception as exc:
                print(f"    EXCEPTION: {exc}")
        print(f"{passed}/5 harness integration checks passed")
        return 0 if passed == 5 else 1
    else:
        # Fixture validation
        from src.core import AgentResult, GoalStream, ObservationStream
        scores = []
        for task in tasks:
            result = AgentResult(
                task.expected_answer, 1, GoalStream(), ObservationStream(),
                final_tool_call=task.expected_tool_call,
            )
            scores.append(harness.score(task, result))
        passed = sum(1 for s in scores if s == 1.0)
        print(f"{passed}/5 fixture tasks scored correctly")
        return 0 if passed == 5 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-known-passing", action="store_true")
    args = parser.parse_args()
    if args.validate_known_passing:
        raise SystemExit(validate_known_passing())


if __name__ == "__main__":
    main()
