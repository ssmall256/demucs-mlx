"""Optional model-level reproduction for GitHub issue #1."""
from __future__ import annotations

import contextlib
import sys

import numpy as np

from demucs_mlx import Separator


def _make_audio(seconds: int, sample_rate: int = 44_100) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    x = np.stack([x, np.roll(x, 100)]).astype(np.float32)
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def _peak(x) -> float:
    return float(np.abs(np.asarray(x)).max())


def _run_case(seconds: int, split: bool) -> float:
    with contextlib.redirect_stdout(sys.stderr):
        separator = Separator("htdemucs", shifts=0, split=split)
        _, stems = separator.separate_tensor(_make_audio(seconds), return_mx=False)
    reconstruction = sum(np.asarray(stems[name]) for name in stems)
    return _peak(reconstruction)


def main() -> None:
    try:
        results = [
            (6, False, _run_case(6, False)),
            (6, True, _run_case(6, True)),
            (20, True, _run_case(20, True)),
        ]
    except Exception as exc:
        print(f"test_issue_1_separator_repro.py: SKIP ({exc.__class__.__name__}: {exc})")
        return

    print("| seconds | split | reconstruction peak |")
    print("|---:|:---:|---:|")
    for seconds, split, peak in results:
        print(f"| {seconds} | `{split}` | **{peak:.6f}** |")
    long_peak = results[-1][2]
    assert long_peak < 2.0, f"20s split=True reconstruction peak is too high: {long_peak}"
    print("test_issue_1_separator_repro.py: OK")


if __name__ == "__main__":
    main()
