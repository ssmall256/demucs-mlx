all: lint test

lint:
	ruff check demucs_mlx
	pyright

format:
	ruff format demucs_mlx

test:
	python tests/test_metal_kernels.py

bench:
	python tests/bench_metal_kernels.py

dist:
	python -m build

clean:
	rm -rf dist build *.egg-info

.PHONY: lint format test bench dist clean
