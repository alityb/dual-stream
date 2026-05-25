from __future__ import annotations

import argparse

from benchmarks.appworld.harness import AppWorldHarness


def validate_known_passing() -> int:
    harness = AppWorldHarness(dataset_name="test_challenge")
    harness.setup()
    tasks = harness.load_tasks(5)
    return 0 if len(tasks) == 5 and all(task.instruction for task in tasks) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-known-passing", action="store_true")
    args = parser.parse_args()
    if args.validate_known_passing:
        raise SystemExit(validate_known_passing())


if __name__ == "__main__":
    main()
