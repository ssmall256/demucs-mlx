all: lint test

lint:
	ruff check demucs_mlx
	pyright

format:
	ruff format demucs_mlx

test:
	python tests/test_metal_kernels.py
	python tests/test_apply_model_chunk_seed.py
	python tests/test_model_converter_optional_mlx_weights.py
	python tests/test_apply_model_overlap_add.py

bench:
	python tests/bench_metal_kernels.py
	python tests/bench_overlap_add.py

dist:
	python -m build

clean:
	rm -rf dist build *.egg-info

.PHONY: lint format test bench dist clean
