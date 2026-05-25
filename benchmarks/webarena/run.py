from __future__ import annotations

import argparse

from benchmarks.webarena.harness import WebArenaHarness
from dual_stream.types import AgentResult, GoalStream, ObservationStream


def validate_known_passing() -> int:
    harness = WebArenaHarness()
    harness.setup()
    scores = []
    for task in harness.load_tasks(5):
        result = AgentResult(task.expected_answer, 1, GoalStream(), ObservationStream())
        scores.append(harness.score(task, result))
    harness.teardown()
    return 0 if scores == [1.0] * 5 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-known-passing", action="store_true")
    args = parser.parse_args()
    if args.validate_known_passing:
        raise SystemExit(validate_known_passing())


if __name__ == "__main__":
    main()
