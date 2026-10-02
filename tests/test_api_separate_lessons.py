"""Tests for demucs_mlx.api.Separator.separate with unified inputs and async writing."""
from pathlib import Path
import tempfile
import mlx.core as mx
import numpy as np
import pytest

from demucs_mlx.api import Separator


def test_separator_separate_in_memory_tensor():
    sep = Separator("htdemucs", split=False)
    # Generate 1s stereo audio
    sr = sep.samplerate
    t = np.arange(sr, dtype=np.float32) / sr
    sine = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_np = np.stack([sine, sine], axis=0)

    # 1. Test numpy array input
    mix_np, stems_np = sep.separate(audio_np)
    assert isinstance(stems_np, dict)
    assert set(stems_np.keys()) == set(sep.model.sources)
    for stem_name, stem_arr in stems_np.items():
        assert isinstance(stem_arr, np.ndarray)
        assert stem_arr.shape == (2, sr)

    # 2. Test mlx array input with return_mx=True
    audio_mx = mx.array(audio_np)
    mix_mx, stems_mx = sep.separate(audio_mx, return_mx=True)
    assert isinstance(stems_mx, dict)
    for stem_name, stem_arr in stems_mx.items():
        assert isinstance(stem_arr, mx.array)
        assert stem_arr.shape == (2, sr)


def test_separator_separate_async_disk_writing():
    sep = Separator("htdemucs", split=False)
    sr = sep.samplerate
    t = np.arange(sr, dtype=np.float32) / sr
    sine = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_np = np.stack([sine, sine], axis=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save via async writing
        saved = sep.separate(
            audio_np,
            output_dir=tmpdir,
            async_write=True,
            filename_format="{stem}.wav",
        )
        assert isinstance(saved, dict)
        assert set(saved.keys()) == set(sep.model.sources)
        for stem_name, stem_path in saved.items():
            assert Path(stem_path).exists()
            assert Path(stem_path).stat().st_size > 0


def test_eval_flush_interval_parity(monkeypatch):
    monkeypatch.setenv("DEMUCS_MLX_EVAL_FLUSH_INTERVAL", "1")
    sep = Separator("htdemucs", split=True, segment=2.0)
    sr = sep.samplerate
    t = np.arange(sr * 4, dtype=np.float32) / sr
    sine = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_np = np.stack([sine, sine], axis=0)

    mix, stems = sep.separate(audio_np)
    assert isinstance(stems, dict)
    for stem_arr in stems.values():
        assert stem_arr.shape[-1] == sr * 4


def test_separator_separate_pure_mlx_async_writing():
    sep = Separator("htdemucs", split=False)
    sr = sep.samplerate
    audio_mx = mx.random.normal((2, sr))
    mx.eval(audio_mx)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = sep.separate(
            audio_mx,
            output_dir=tmpdir,
            async_write=True,
            filename_format="{stem}.wav",
        )
        assert isinstance(saved, dict)
        assert set(saved.keys()) == set(sep.model.sources)
        for stem_name, stem_path in saved.items():
            assert Path(stem_path).exists()
            assert Path(stem_path).stat().st_size > 0

