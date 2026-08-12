"""Regression check for MLX thread-local streams during audio prefetch."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as mac
import numpy as np

from demucs_mlx.separate import _iter_prefetched_audio


class _FakeModel:
    samplerate = 44_100
    audio_channels = 2


def main() -> None:
    t = mx.arange(4_410, dtype=mx.float32) / _FakeModel.samplerate
    mono = 0.25 * mx.sin(2 * np.pi * 440 * t)
    audio = mx.stack([mono, mono], axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
        path = Path(file.name)
    try:
        mac.save(str(path), audio, _FakeModel.samplerate, encoding="float32")

        direct_items = list(_iter_prefetched_audio([str(path)], _FakeModel(), prefetch=0))
        prefetched_items = list(_iter_prefetched_audio([str(path)], _FakeModel(), prefetch=2))
        assert direct_items[0][0] == path
        assert prefetched_items[0][0] == path

        direct = direct_items[0][1]
        prefetched = prefetched_items[0][1]
        mx.eval(direct, prefetched)
        np.testing.assert_allclose(np.asarray(prefetched), np.asarray(direct), atol=0, rtol=0)
    finally:
        path.unlink(missing_ok=True)

    print("test_prefetch_thread_stream.py: OK")


if __name__ == "__main__":
    main()
