from __future__ import annotations

from analysis.utils import bootstrap_ci, load_results
from experiments.run_sweep import write_sweep
from tests.test_run_sweep import FakeBackend, FakeHarness


def test_bootstrap_ci_returns_min_max() -> None:
    assert bootstrap_ci([0.2, 0.5, 0.3]) == (0.2, 0.5)


def test_load_results_reads_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("experiments.run_sweep.harness_for", lambda benchmark: FakeHarness())
    write_sweep(10, tmp_path, backend_factory=lambda: FakeBackend())
    assert len(load_results(tmp_path)) == 32
