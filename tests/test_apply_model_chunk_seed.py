"""Regression checks for apply_model chunk handling and seeded shifts."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from demucs_mlx.apply_mlx import TensorChunk, apply_model


class _TinyModel:
    samplerate = 100
    segment = 0.3
    sources = ["vocals", "drums", "bass", "other"]

    def valid_length(self, length: int) -> int:
        return int(length)

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: [batch, channels, time]
        return mx.stack([x * float(i + 1) for i in range(len(self.sources))], axis=1)


def _assert_chunk_shift_length() -> None:
    model = _TinyModel()
    mix = mx.arange(1 * 2 * 100, dtype=mx.float32).reshape(1, 2, 100) / 1000.0
    chunk = TensorChunk(mix, offset=15, length=70)
    out = apply_model(model, chunk, shifts=2, split=False, seed=0)
    mx.eval(out)
    assert int(out.shape[-1]) == 70


def _assert_chunk_split_length() -> None:
    model = _TinyModel()
    mix = mx.arange(1 * 2 * 120, dtype=mx.float32).reshape(1, 2, 120) / 1000.0
    chunk = TensorChunk(mix, offset=20, length=80)
    out = apply_model(
        model,
        chunk,
        shifts=0,
        split=True,
        overlap=0.25,
        segment=0.3,
        batch_size=2,
    )
    mx.eval(out)
    assert int(out.shape[-1]) == 80


def _assert_seed_reproducible() -> None:
    model = _TinyModel()
    mix = mx.arange(1 * 2 * 96, dtype=mx.float32).reshape(1, 2, 96) / 1000.0
    out1 = apply_model(model, mix, shifts=3, split=False, seed=123)
    out2 = apply_model(model, mix, shifts=3, split=False, seed=123)
    mx.eval(out1, out2)
    np1 = np.asarray(out1)
    np2 = np.asarray(out2)
    assert np.allclose(np1, np2, atol=1e-7)


def _assert_seed_none_executes() -> None:
    model = _TinyModel()
    mix = mx.arange(1 * 2 * 64, dtype=mx.float32).reshape(1, 2, 64) / 1000.0
    out = apply_model(model, mix, shifts=2, split=False, seed=None)
    mx.eval(out)
    assert int(out.shape[-1]) == 64


if __name__ == "__main__":
    _assert_chunk_shift_length()
    _assert_chunk_split_length()
    _assert_seed_reproducible()
    _assert_seed_none_executes()
    print("test_apply_model_chunk_seed.py: OK")
