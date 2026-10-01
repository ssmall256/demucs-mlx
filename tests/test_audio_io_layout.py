"""Tests for audio load/save layout ergonomics and consistency."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from demucs_mlx.audio import load_audio, save_audio


def test_save_load_mlx_stereo_roundtrip():
    """Verify that (2, frames) MLX array roundtrips directly without transposition."""
    sr = 44100
    frames = 4410
    t = mx.arange(frames, dtype=mx.float32) / sr
    sine1 = 0.5 * mx.sin(2 * np.pi * 440.0 * t)
    sine2 = 0.3 * mx.sin(2 * np.pi * 880.0 * t)
    audio = mx.stack([sine1, sine2], axis=0)  # (channels=2, frames)
    mx.eval(audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        save_audio(audio, path, samplerate=sr, as_float=True)
        loaded, loaded_sr = load_audio(path, sr=sr)
        mx.eval(loaded)

        assert loaded_sr == sr
        assert loaded.shape == (2, frames)
        max_diff = float(mx.max(mx.abs(loaded - audio)))
        assert max_diff < 1e-4
    finally:
        path.unlink(missing_ok=True)


def test_save_load_mlx_mono_roundtrip():
    """Verify that (1, frames) MLX array roundtrips directly."""
    sr = 44100
    frames = 4410
    t = mx.arange(frames, dtype=mx.float32) / sr
    audio = mx.reshape(0.5 * mx.sin(2 * np.pi * 440.0 * t), (1, frames))
    mx.eval(audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        save_audio(audio, path, samplerate=sr, as_float=True)
        loaded, loaded_sr = load_audio(path, sr=sr)
        mx.eval(loaded)

        assert loaded_sr == sr
        assert loaded.shape == (1, frames)
        max_diff = float(mx.max(mx.abs(loaded - audio)))
        assert max_diff < 1e-4
    finally:
        path.unlink(missing_ok=True)


def test_save_load_numpy_stereo_roundtrip():
    """Verify that (2, frames) NumPy array roundtrips directly without transposition."""
    sr = 44100
    frames = 4410
    t = np.arange(frames, dtype=np.float32) / sr
    sine1 = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sine2 = 0.3 * np.sin(2 * np.pi * 880.0 * t)
    audio_np = np.stack([sine1, sine2], axis=0)  # (channels=2, frames)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        save_audio(audio_np, path, samplerate=sr, as_float=True)
        loaded, loaded_sr = load_audio(path, sr=sr)
        mx.eval(loaded)

        assert loaded_sr == sr
        assert loaded.shape == (2, frames)
        max_diff = float(mx.max(mx.abs(loaded - mx.array(audio_np))))
        assert max_diff < 1e-4
    finally:
        path.unlink(missing_ok=True)


def test_save_load_1d_roundtrip():
    """Verify that 1D audio array is saved as mono and loaded as (1, frames)."""
    sr = 44100
    frames = 4410
    t = mx.arange(frames, dtype=mx.float32) / sr
    audio = 0.5 * mx.sin(2 * np.pi * 440.0 * t)
    mx.eval(audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        save_audio(audio, path, samplerate=sr, as_float=True)
        loaded, loaded_sr = load_audio(path, sr=sr)
        mx.eval(loaded)

        assert loaded_sr == sr
        assert loaded.shape == (1, frames)
        max_diff = float(mx.max(mx.abs(loaded[0] - audio)))
        assert max_diff < 1e-4
    finally:
        path.unlink(missing_ok=True)


def test_separate_audio_file_e2e():
    """Verify Separator.separate_audio_file runs correctly with the new layout."""
    from demucs_mlx import Separator

    sr = 44100
    frames = 44100  # 1 second
    t = mx.arange(frames, dtype=mx.float32) / sr
    sine1 = 0.5 * mx.sin(2 * np.pi * 440.0 * t)
    sine2 = 0.3 * mx.sin(2 * np.pi * 880.0 * t)
    audio = mx.stack([sine1, sine2], axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)

    try:
        save_audio(audio, path, samplerate=sr, as_float=True)
        separator = Separator("htdemucs", shifts=0, split=False)
        _, stems = separator.separate_audio_file(path, return_mx=True)
        assert len(stems) == 4
        for name, stem in stems.items():
            assert stem.shape == (2, frames)
    finally:
        path.unlink(missing_ok=True)


def test_separate_cli_e2e():
    """Verify separate.main CLI separates a file and writes output stems."""
    from demucs_mlx.separate import main

    sr = 44100
    frames = 44100  # 1 second
    t = mx.arange(frames, dtype=mx.float32) / sr
    sine1 = 0.5 * mx.sin(2 * np.pi * 440.0 * t)
    sine2 = 0.3 * mx.sin(2 * np.pi * 880.0 * t)
    audio = mx.stack([sine1, sine2], axis=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_in.wav"
        out_dir = Path(tmpdir) / "out"
        save_audio(audio, input_path, samplerate=sr, as_float=True)

        cmd = [
            "-n", "htdemucs",
            "--shifts", "0",
            "--no-split",
            "-o", str(out_dir),
            str(input_path),
        ]
        ret = main(cmd)
        assert ret == 0

        # Model creates out_dir / "test_in" / {stem}.wav
        track_out = out_dir / "test_in"
        assert track_out.exists()
        for stem_name in ["drums", "bass", "other", "vocals"]:
            stem_path = track_out / f"{stem_name}.wav"
            assert stem_path.exists()
            stem_audio, stem_sr = load_audio(stem_path, sr=sr)
            assert stem_sr == sr
            assert stem_audio.shape == (2, frames)


