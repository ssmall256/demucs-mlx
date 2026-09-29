"""CPU-only validation for opt-in Neural Engine argument errors."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from demucs_mlx.ane import WaveformConv
from demucs_mlx.api import Separator
from demucs_mlx.separate import main


def raises(call, expected):
    try:
        call()
    except (ValueError, FileNotFoundError, SystemExit) as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected an error containing {expected!r}")


def main_test():
    raises(lambda: Separator(model="htdemucs_ft", ane_time_encoder=True), "default htdemucs")
    raises(lambda: Separator(segment=6.0, ane_time_encoder=True), "7.8-second")
    raises(lambda: Separator(split=False, ane_time_encoder=True), "split=True")
    raises(lambda: Separator(batch_size=3, ane_time_encoder=True), "batch_size 1 or 2")
    raises(
        lambda: main(["track.wav", "--ane-time-encoder", "-n", "htdemucs_ft"]),
        "default htdemucs",
    )
    raises(
        lambda: main(["track.wav", "--ane-time-encoder", "--batch-size", "3"]),
        "--batch-size 1 or 2",
    )
    with TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.mlmodelc"
        with patch("demucs_mlx.ane.compiled_path", return_value=missing):
            raises(WaveformConv, "Converted waveform convolution missing")
    print("test_ane_errors.py: OK")


if __name__ == "__main__":
    main_test()
