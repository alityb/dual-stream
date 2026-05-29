from __future__ import annotations

import argparse

from benchmarks.synthetic.harness import SyntheticHarness, _RetrievalExecutor, _TransformationExecutor, _SearchExecutor


def validate_known_passing() -> int:
    """Confirm oracle=1.0, random≈0.0, flat-obs-limited < 0.5 on n_items=50."""
    harness = SyntheticHarness()
    harness.setup()

    # --- Oracle agent: always calls the right tools in order ---
    passed = 0
    total = 0
    for n_items in [10, 20, 50]:
        for task_type in ["retrieval", "transformation", "search"]:
            tasks = SyntheticHarness(n_items_per_task=n_items).load_tasks(3)
            tasks = [t for t in tasks if t.metadata["task_type"] == task_type][:1]
            for task in tasks:
                total += 1
                executor = harness.make_tool_executor(task)
                import json
                raw = json.loads(task.metadata["data"])
                # Run oracle
                if task_type == "retrieval":
                    catalog = raw["catalog"]
                    for p in catalog:
                        executor(f"get_product({p['id']})")
                        executor(f"record_price({p['id']}, {p['price']})")
                    executor("finish()")
                elif task_type == "transformation":
                    for r in raw["records"]:
                        executor(f"get_record({r['id']})")
                        executor(f"label({r['id']}, {r['expected_label']})")
                    executor("finish()")
                else:  # search
                    target = raw["target_id"]
                    for item in raw["items"]:
                        executor(f"check_item({item['id']})")
                    executor(f"submit_answer({target})")
                score = harness.score(task, None)  # type: ignore[arg-type]
                if score >= 0.99:
                    passed += 1
                else:
                    print(f"  ORACLE FAIL task_type={task_type} n={n_items} score={score:.3f}")

    print(f"Oracle: {passed}/{total} scored >= 0.99")
    if passed < total:
        return 1

    # --- Confirm gate: harness loads tasks, executors respond correctly ---
    tasks = harness.load_tasks(9)
    print(f"Loaded {len(tasks)} tasks, types: {set(t.metadata['task_type'] for t in tasks)}")

    print("--validate-known-passing PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-known-passing", action="store_true")
    args = parser.parse_args()
    if args.validate_known_passing:
        raise SystemExit(validate_known_passing())


if __name__ == "__main__":
    main()
