from __future__ import annotations

import json

from benchmarks.synthetic.harness import (
    SyntheticHarness,
    _RetrievalExecutor,
    _TransformationExecutor,
    _SearchExecutor,
    _make_catalog,
    _make_records,
    _make_database,
)
from src.core import GoalStream, ObservationStream, AgentResult


def _dummy_result() -> AgentResult:
    return AgentResult(
        answer=None, steps_used=1,
        goal_stream_snapshot=GoalStream(),
        obs_stream_snapshot=ObservationStream(),
    )


def test_retrieval_oracle_scores_1() -> None:
    catalog = _make_catalog(5, 42)
    ex = _RetrievalExecutor(catalog, 5)
    for p in catalog:
        ex(f"get_product({p['id']})")
        ex(f"record_price({p['id']}, {p['price']})")
    ex("finish()")
    assert ex.score() == 1.0


def test_retrieval_partial_scores_fraction() -> None:
    catalog = _make_catalog(10, 42)
    ex = _RetrievalExecutor(catalog, 10)
    # Record only first 5 correctly
    for p in catalog[:5]:
        ex(f"record_price({p['id']}, {p['price']})")
    assert 0.4 < ex.score() < 0.6


def test_transformation_oracle_scores_1() -> None:
    records = _make_records(5, 42)
    ex = _TransformationExecutor(records, 5)
    for r in records:
        ex(f"get_record({r['id']})")
        ex(f"label({r['id']}, {r['expected_label']})")
    ex("finish()")
    assert ex.score() == 1.0


def test_transformation_wrong_labels_score_0() -> None:
    records = _make_records(5, 42)
    ex = _TransformationExecutor(records, 5)
    for r in records:
        ex(f"label({r['id']}, LOW)")  # always LOW regardless
    wrong = sum(1 for r in records if r["expected_label"] != "LOW")
    expected_score = (5 - wrong) / 5
    assert abs(ex.score() - expected_score) < 0.01


def test_search_oracle_scores_1() -> None:
    items, target_idx = _make_database(10, 42)
    ex = _SearchExecutor(items, items[target_idx]["id"], 10)
    for item in items:
        ex(f"check_item({item['id']})")
    ex(f"submit_answer({items[target_idx]['id']})")
    assert ex.score() == 1.0


def test_search_wrong_answer_scores_0_no_check() -> None:
    items, target_idx = _make_database(10, 42)
    target_id = items[target_idx]["id"]
    wrong_id = next(item["id"] for item in items if item["id"] != target_id)
    # Checked nothing, submitted wrong answer
    ex = _SearchExecutor(items, target_id, 10)
    ex(f"submit_answer({wrong_id})")
    assert ex.score() == 0.0

    # Checked the right item but submitted wrong answer
    ex2 = _SearchExecutor(items, target_id, 10)
    ex2(f"check_item({target_id})")
    # No answer submitted — checked it though — partial credit
    assert ex2.score() == 0.3

    # Submitted wrong answer after checking correct item — still 0
    ex3 = _SearchExecutor(items, target_id, 10)
    ex3(f"check_item({target_id})")
    ex3(f"submit_answer({wrong_id})")
    assert ex3.score() == 0.0


def test_harness_load_tasks_returns_all_types() -> None:
    harness = SyntheticHarness()
    tasks = harness.load_tasks(9)
    assert len(tasks) == 9
    types = {t.metadata["task_type"] for t in tasks}
    assert types == {"retrieval", "transformation", "search"}


def test_harness_make_tool_executor_is_callable() -> None:
    harness = SyntheticHarness(n_items_per_task=5)
    tasks = harness.load_tasks(3)
    for task in tasks:
        ex = harness.make_tool_executor(task)
        result = ex("finish()")
        assert isinstance(result, str)


def test_harness_score_reads_last_score_from_metadata() -> None:
    harness = SyntheticHarness(n_items_per_task=5)
    tasks = harness.load_tasks(1)
    task = tasks[0]
    task.metadata["last_score"] = "0.75"
    assert harness.score(task, _dummy_result()) == 0.75


def test_validate_known_passing_gate() -> None:
    """Oracle should score 1.0, harness loads tasks correctly."""
    harness = SyntheticHarness(n_items_per_task=5)
    tasks = harness.load_tasks(6)
    for task in tasks:
        executor = harness.make_tool_executor(task)
        raw = json.loads(task.metadata["data"])
        task_type = task.metadata["task_type"]
        if task_type == "retrieval":
            for p in raw["catalog"]:
                executor(f"get_product({p['id']})")
                executor(f"record_price({p['id']}, {p['price']})")
            executor("finish()")
        elif task_type == "transformation":
            for r in raw["records"]:
                executor(f"get_record({r['id']})")
                executor(f"label({r['id']}, {r['expected_label']})")
            executor("finish()")
        else:
            target = raw["target_id"]
            for item in raw["items"]:
                executor(f"check_item({item['id']})")
            executor(f"submit_answer({target})")
        score = harness.score(task, _dummy_result())
        assert score >= 0.99, f"Oracle failed task_type={task_type} score={score}"
