"""Regression checks for split-mode overlap-add accumulation."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from demucs_mlx.apply_mlx import apply_model


class _IdentityModel:
    samplerate = 44_100
    segment = 7.8
    sources = ["mix"]

    def valid_length(self, length: int) -> int:
        return int(length)

    def __call__(self, x: mx.array) -> mx.array:
        return x[:, None, :, :]


def _make_audio(seconds: int) -> np.ndarray:
    sr = _IdentityModel.samplerate
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    x = np.stack([x, np.roll(x, 100)]).astype(np.float32)
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def _assert_split_reconstructs(seconds: int, overlap: float, batch_size: int) -> None:
    wav = _make_audio(seconds)
    mix = mx.array(wav[None, ...])
    out = apply_model(
        _IdentityModel(),
        mix,
        shifts=0,
        split=True,
        overlap=overlap,
        batch_size=batch_size,
    )
    mx.eval(out)
    reconstructed = np.asarray(out)[0, 0]
    max_error = float(np.max(np.abs(reconstructed - wav)))
    output_peak = float(np.abs(reconstructed).max())
    assert max_error <= 1e-4, (
        f"{seconds}s overlap={overlap} batch_size={batch_size}: "
        f"max reconstruction error {max_error}"
    )
    assert output_peak <= 0.91, (
        f"{seconds}s overlap={overlap} batch_size={batch_size}: output peak {output_peak}"
    )


def main() -> None:
    for seconds in (20, 60):
        for overlap in (0.0, 0.25, 0.5):
            for batch_size in (1, 8):
                _assert_split_reconstructs(seconds, overlap, batch_size)
    print("test_apply_model_overlap_add.py: OK")


if __name__ == "__main__":
    main()
