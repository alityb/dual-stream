from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from benchmarks.base import BenchmarkHarness
from src.core import AgentResult, BenchmarkTask, TaskSpec


# ---------------------------------------------------------------------------
# Task data generators — all deterministic, seeded by task_id
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_catalog(n_items: int, seed: int) -> list[dict[str, Any]]:
    rng = _rng(seed)
    categories = ["Electronics", "Books", "Clothing", "Food", "Tools"]
    products = []
    for i in range(1, n_items + 1):
        price = round(rng.uniform(1.99, 499.99), 2)
        products.append({
            "id": i,
            "name": f"Product-{i:03d}-{hashlib.md5(f'{seed}{i}'.encode()).hexdigest()[:4].upper()}",
            "price": price,
            "category": rng.choice(categories),
        })
    return products


def _make_records(n_items: int, seed: int) -> list[dict[str, Any]]:
    rng = _rng(seed)
    records = []
    for i in range(1, n_items + 1):
        value = rng.randint(1, 999)
        records.append({
            "id": i,
            "value": value,
            # Rule: if value > 500 → "HIGH", 200-500 → "MED", else → "LOW"
            "expected_label": "HIGH" if value > 500 else ("MED" if value >= 200 else "LOW"),
        })
    return records


def _make_database(n_items: int, seed: int) -> list[dict[str, Any]]:
    rng = _rng(seed)
    items = []
    # Constraints: color in {red, blue}, size > 20, weight < 10, material=metal, year >= 2020
    colors = ["red", "blue", "green", "yellow"]
    materials = ["metal", "plastic", "wood", "fabric"]
    for i in range(1, n_items + 1):
        items.append({
            "id": i,
            "color": rng.choice(colors),
            "size": rng.randint(5, 50),
            "weight": round(rng.uniform(0.5, 20.0), 1),
            "material": rng.choice(materials),
            "year": rng.randint(2015, 2024),
        })
    # Ensure exactly one item satisfies all constraints (for solvability)
    target_idx = rng.randint(0, n_items - 1)
    items[target_idx].update({
        "color": "red",
        "size": rng.randint(21, 50),
        "weight": round(rng.uniform(0.5, 9.9), 1),
        "material": "metal",
        "year": rng.randint(2020, 2024),
    })
    return items, target_idx


TASK_TYPES = ["retrieval", "transformation", "search"]


def _task_instruction(task_type: str, n_items: int, seed: int) -> tuple[str, dict]:
    if task_type == "retrieval":
        catalog = _make_catalog(n_items, seed)
        instruction = (
            f"Collect the price of every product in this catalog of {n_items} products. "
            f"For each product, call get_product(id) to retrieve its price, then call "
            f"record_price(id, price) to record it. "
            f"You must record all {n_items} products. "
            f"Call finish() when all prices are recorded."
        )
        metadata = {"catalog": catalog}
    elif task_type == "transformation":
        records = _make_records(n_items, seed)
        instruction = (
            f"Process all {n_items} records. For each record, call get_record(id) to "
            f"retrieve its value, then classify it: if value > 500 call label(id, 'HIGH'), "
            f"if 200-500 call label(id, 'MED'), else call label(id, 'LOW'). "
            f"Call finish() when all {n_items} records are labeled."
        )
        metadata = {"records": records}
    else:  # search
        items, target_idx = _make_database(n_items, seed)
        instruction = (
            f"Search this database of {n_items} items. Find the ONE item that satisfies "
            f"ALL of these constraints: color=red, size>20, weight<10, material=metal, "
            f"year>=2020. "
            f"For each candidate, call check_item(id) to inspect it. "
            f"When you find the matching item, call submit_answer(id)."
        )
        metadata = {"items": items, "target_id": items[target_idx]["id"]}
    return instruction, metadata


# ---------------------------------------------------------------------------
# Deterministic tool executor — state machine per task
# ---------------------------------------------------------------------------

class _RetrievalExecutor:
    """Tracks which product prices have been retrieved and recorded."""

    def __init__(self, catalog: list[dict], n_items: int) -> None:
        self._catalog = {p["id"]: p for p in catalog}
        self._n_items = n_items
        self._retrieved: set[int] = set()
        self._recorded: dict[int, float] = {}
        self._finished = False

    def __call__(self, tool_call: str) -> str:
        tool_call = tool_call.strip()
        m = re.match(r"get_product\((\d+)\)", tool_call)
        if m:
            pid = int(m.group(1))
            self._retrieved.add(pid)
            if pid not in self._catalog:
                return f"ERROR: product {pid} not found"
            p = self._catalog[pid]
            return f"Product {pid}: name={p['name']}, price={p['price']}, category={p['category']}"
        m = re.match(r"record_price\((\d+),\s*([0-9.]+)\)", tool_call)
        if m:
            pid, price = int(m.group(1)), float(m.group(2))
            if pid not in self._catalog:
                return f"ERROR: product {pid} not found"
            self._recorded[pid] = price
            return f"Recorded price {price} for product {pid}. ({len(self._recorded)}/{self._n_items} done)"
        if tool_call.startswith("finish"):
            self._finished = True
            return f"Task finished. Recorded {len(self._recorded)}/{self._n_items} prices."
        return f"Unknown command: {tool_call}. Available: get_product(id), record_price(id, price), finish()"

    def score(self) -> float:
        if not self._catalog:
            return 0.0
        correct = 0
        for pid, p in self._catalog.items():
            if pid in self._recorded and abs(self._recorded[pid] - p["price"]) < 0.01:
                correct += 1
        return correct / len(self._catalog)


class _TransformationExecutor:
    """Tracks which records have been retrieved and correctly labeled."""

    def __init__(self, records: list[dict], n_items: int) -> None:
        self._records = {r["id"]: r for r in records}
        self._n_items = n_items
        self._retrieved: set[int] = set()
        self._labels: dict[int, str] = {}
        self._finished = False

    def __call__(self, tool_call: str) -> str:
        tool_call = tool_call.strip()
        m = re.match(r"get_record\((\d+)\)", tool_call)
        if m:
            rid = int(m.group(1))
            self._retrieved.add(rid)
            if rid not in self._records:
                return f"ERROR: record {rid} not found"
            return f"Record {rid}: value={self._records[rid]['value']}"
        m = re.match(r"label\((\d+),\s*['\"]?(HIGH|MED|LOW)['\"]?\)", tool_call)
        if m:
            rid, label = int(m.group(1)), m.group(2)
            self._labels[rid] = label
            return f"Labeled record {rid} as {label}. ({len(self._labels)}/{self._n_items} done)"
        if tool_call.startswith("finish"):
            self._finished = True
            return f"Task finished. Labeled {len(self._labels)}/{self._n_items} records."
        return f"Unknown: {tool_call}. Available: get_record(id), label(id, LEVEL), finish()"

    def score(self) -> float:
        if not self._records:
            return 0.0
        correct = sum(
            1 for rid, r in self._records.items()
            if self._labels.get(rid) == r["expected_label"]
        )
        return correct / len(self._records)


class _SearchExecutor:
    """Tracks constraint filtering progress and final answer."""

    def __init__(self, items: list[dict], target_id: int, n_items: int) -> None:
        self._items = {item["id"]: item for item in items}
        self._target_id = target_id
        self._n_items = n_items
        self._checked: set[int] = set()
        self._answer: int | None = None

    def __call__(self, tool_call: str) -> str:
        tool_call = tool_call.strip()
        m = re.match(r"check_item\((\d+)\)", tool_call)
        if m:
            iid = int(m.group(1))
            self._checked.add(iid)
            if iid not in self._items:
                return f"ERROR: item {iid} not found"
            item = self._items[iid]
            return (
                f"Item {iid}: color={item['color']}, size={item['size']}, "
                f"weight={item['weight']}, material={item['material']}, year={item['year']}"
            )
        m = re.match(r"submit_answer\((\d+)\)", tool_call)
        if m:
            self._answer = int(m.group(1))
            correct = self._answer == self._target_id
            return f"Answer submitted: item {self._answer}. {'CORRECT!' if correct else 'WRONG.'}"
        return f"Unknown: {tool_call}. Available: check_item(id), submit_answer(id)"

    def score(self) -> float:
        if self._answer == self._target_id:
            return 1.0
        if self._answer is not None:
            return 0.0  # submitted a wrong answer — no partial credit
        # No answer submitted: partial credit only if target was inspected
        if self._target_id in self._checked:
            return 0.3
        return 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class SyntheticHarness(BenchmarkHarness):
    """Deterministic synthetic long-horizon benchmark. No network, no Docker."""

    def __init__(self, n_items_per_task: int | None = None) -> None:
        # Default n_items progression: mix of sizes to stress-test horizon lengths
        self._n_items_per_task = n_items_per_task
        self._tasks: list[BenchmarkTask] = []

    def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    def load_tasks(self, n: int | None = None) -> list[BenchmarkTask]:
        if self._tasks:
            return self._tasks if n is None else self._tasks[:n]

        count = n if n is not None else 100
        tasks = []
        # Distribute across task types and n_item sizes
        sizes = [10, 20, 50, 100]
        for i in range(count):
            task_type = TASK_TYPES[i % len(TASK_TYPES)]
            # Cycle through sizes, weighted toward larger (harder) tasks
            size_idx = (i // len(TASK_TYPES)) % len(sizes)
            if self._n_items_per_task is not None:
                n_items = self._n_items_per_task
            else:
                n_items = sizes[size_idx]
            seed = 42 + i
            instruction, raw_meta = _task_instruction(task_type, n_items, seed)
            # Store metadata as JSON strings (BenchmarkTask.metadata is dict[str,str])
            metadata = {
                "task_type": task_type,
                "n_items": str(n_items),
                "seed": str(seed),
                "data": json.dumps(raw_meta),
            }
            key_terms = extract_key_terms_synthetic(instruction)
            tasks.append(BenchmarkTask(
                id=f"synthetic-{task_type}-{n_items}-{i:04d}",
                instruction=instruction,
                key_terms=key_terms,
                expected_answer="",
                category=task_type,
                metadata=metadata,
            ))
        self._tasks = tasks
        return tasks if n is None else tasks[:n]

    def extract_task_spec(self, task: BenchmarkTask) -> TaskSpec:
        terms = task.key_terms or ([task.category] if task.category else [])
        return TaskSpec(id=task.id, text=task.instruction, key_terms=terms)

    def score(self, task: BenchmarkTask, result: AgentResult) -> float:
        """Continuous partial-credit score from the executor stored in metadata."""
        score_str = task.metadata.get("last_score", "0.0")
        try:
            return float(score_str)
        except ValueError:
            return 0.0

    def make_tool_executor(self, task: BenchmarkTask):
        """Return a stateful deterministic executor for this task."""
        task_type = task.metadata["task_type"]
        n_items = int(task.metadata["n_items"])
        seed = int(task.metadata["seed"])
        raw_meta = json.loads(task.metadata["data"])

        if task_type == "retrieval":
            executor = _RetrievalExecutor(raw_meta["catalog"], n_items)
        elif task_type == "transformation":
            executor = _TransformationExecutor(raw_meta["records"], n_items)
        else:
            executor = _SearchExecutor(raw_meta["items"], raw_meta["target_id"], n_items)

        def wrapped_executor(tool_call: str) -> str:
            result = executor(tool_call)
            # Update score in task metadata after each call so scorer can read it
            task.metadata["last_score"] = str(executor.score())
            return result

        return wrapped_executor


def extract_key_terms_synthetic(text: str) -> list[str]:
    """Extract simple key terms without external dependencies."""
    stop = {"a", "all", "an", "and", "are", "as", "at", "be", "by", "call",
            "do", "each", "for", "from", "if", "in", "is", "it", "of", "on",
            "or", "the", "then", "this", "to", "when", "with", "you", "your"}
    words = re.findall(r"[A-Za-z]+", text.lower())
    terms = [w for w in words if len(w) > 3 and w not in stop]
    # deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for w in terms:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:8]
