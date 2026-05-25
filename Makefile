.PHONY: test check quickstart figures reproduce clean-results

test:
	python3.11 -m pytest

check:
	python3.11 -m ruff check .
	python3.11 -m black --check .

quickstart:
	python3.11 -m quickstart

reproduce:
	python3.11 -m benchmarks.webarena.run --validate-known-passing
	python3.11 -m benchmarks.tau_bench.run --validate-known-passing
	python3.11 -m experiments.run_sweep --tasks 50
	$(MAKE) figures
	python3.11 -m analysis.verify_paper_numbers
