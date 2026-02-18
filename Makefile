all: linter tests

linter:
	ruff check demucs_mlx
	pyright

format:
	ruff format demucs_mlx

tests: test_train test_eval

test_train: tests/musdb
	_DORA_TEST_PATH=/tmp/demucs python3 -m dora run --clear \
		dset.musdb=./tests/musdb dset.segment=4 dset.shift=2 epochs=2 model=demucs \
		demucs_mlx.depth=2 demucs_mlx.channels=4 test.sdr=false misc.num_workers=0 test.workers=0 \
		test.shifts=0

test_eval:
	@test -n "$(TEST_AUDIO)" || (echo "Set TEST_AUDIO=/path/to/audio.wav to run test_eval" && exit 1)
	python3 -m demucs_mlx -n demucs_unittest $(TEST_AUDIO)
	python3 -m demucs_mlx -n demucs_unittest --two-stems=vocals $(TEST_AUDIO)
	python3 -m demucs_mlx -n demucs_unittest --mp3 $(TEST_AUDIO)
	python3 -m demucs_mlx -n demucs_unittest --flac --int24 $(TEST_AUDIO)
	python3 -m demucs_mlx -n demucs_unittest --int24 --clip-mode clamp $(TEST_AUDIO)
	python3 -m demucs_mlx -n demucs_unittest --segment 8 $(TEST_AUDIO)
	python3 -m demucs_mlx --list-models

tests/musdb:
	test -e tests || mkdir tests
	python3 -c 'import musdb; musdb.DB("tests/tmp", download=True)'
	musdbconvert tests/tmp tests/musdb

dist:
	python3 -m build

clean:
	rm -r dist build *.egg-info

.PHONY: linter format dist test_train test_eval
