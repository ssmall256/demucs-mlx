"""Tests for audio resampling via mlx-audio-io in load paths."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as mac
import numpy as np


def _make_sine_wav(
    sr: int,
    duration: float = 1.0,
    freq: float = 440.0,
    channels: int = 2,
) -> mx.array:
    """Generate a sine wave as (frames, channels) MLX array."""
    t = mx.arange(int(sr * duration), dtype=mx.float32) / sr
    mono = 0.5 * mx.sin(2 * np.pi * freq * t)
    if channels == 1:
        return mono[:, None]
    return mx.stack([mono] * channels, axis=1)


def _assert_resample_basic() -> None:
    """mac.resample produces correct output shape and preserves layout."""
    sr_in, sr_out = 48000, 44100
    audio = _make_sine_wav(sr_in, duration=0.5, channels=2)
    resampled = mac.resample(audio, sr_in, sr_out)
    expected_frames = int(round(audio.shape[0] * sr_out / sr_in))
    assert resampled.shape[1] == audio.shape[1], "channels must be preserved"
    assert abs(resampled.shape[0] - expected_frames) <= 1, (
        f"expected ~{expected_frames} frames, got {resampled.shape[0]}"
    )


def _assert_resample_soxr_vhq() -> None:
    """soxr_vhq quality produces numerically different (higher quality) results than 'fastest'."""
    if not mac.supports_soxr():
        print("test_resample.py: SKIP soxr_vhq (mlx-audio-io built without libsoxr)")
        return

    sr_in, sr_out = 48000, 44100
    audio = _make_sine_wav(sr_in, duration=0.5, channels=1)
    r_fast = mac.resample(audio, sr_in, sr_out, quality="fastest")
    r_vhq = mac.resample(audio, sr_in, sr_out, quality="soxr_vhq")
    assert r_fast.shape == r_vhq.shape
    diff = float(mx.max(mx.abs(r_fast - r_vhq)))
    assert diff > 1e-6, f"soxr_vhq and fastest should differ, but max_diff={diff}"


def _assert_resample_identity() -> None:
    """Resampling to the same rate is a no-op (or near-identity)."""
    sr = 44100
    audio = _make_sine_wav(sr, duration=0.25, channels=2)
    resampled = mac.resample(audio, sr, sr)
    assert resampled.shape[0] == audio.shape[0]
    diff = float(mx.max(mx.abs(resampled - audio)))
    assert diff < 1e-4, f"same-rate resample should be near-identity, max_diff={diff}"


def _assert_load_audio_resamples() -> None:
    """_load_audio resamples when file SR != model SR."""
    from demucs_mlx.separate import _load_audio

    class _FakeModel:
        samplerate = 44100
        audio_channels = 2

    sr_in = 48000
    audio = _make_sine_wav(sr_in, duration=0.5, channels=2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        mac.save(str(tmp), audio, sr_in, encoding="float32")
        wav = _load_audio(tmp, _FakeModel())
        assert isinstance(wav, mx.array), f"expected mx.array, got {type(wav)}"
        assert wav.shape[0] == 2, f"expected 2 channels, got {wav.shape[0]}"
        expected_frames = int(round(audio.shape[0] * 44100 / sr_in))
        assert abs(wav.shape[1] - expected_frames) <= 1, (
            f"expected ~{expected_frames} frames, got {wav.shape[1]}"
        )
    finally:
        tmp.unlink(missing_ok=True)


def _assert_load_audio_no_resample() -> None:
    """_load_audio skips resampling when file SR matches model SR."""
    from demucs_mlx.separate import _load_audio

    class _FakeModel:
        samplerate = 44100
        audio_channels = 2

    sr = 44100
    audio = _make_sine_wav(sr, duration=0.25, channels=2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        mac.save(str(tmp), audio, sr, encoding="float32")
        wav = _load_audio(tmp, _FakeModel())
        assert isinstance(wav, mx.array), f"expected mx.array, got {type(wav)}"
        assert wav.shape[0] == 2
        assert wav.shape[1] == audio.shape[0]
    finally:
        tmp.unlink(missing_ok=True)


def _assert_load_audio_channel_remix() -> None:
    """_load_audio handles mono -> stereo and stereo -> mono."""
    from demucs_mlx.separate import _load_audio

    sr = 44100

    # mono file, stereo model
    class _StereoModel:
        samplerate = 44100
        audio_channels = 2

    audio_mono = _make_sine_wav(sr, duration=0.25, channels=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        mac.save(str(tmp), audio_mono, sr, encoding="float32")
        wav = _load_audio(tmp, _StereoModel())
        assert wav.shape[0] == 2, f"expected 2 channels, got {wav.shape[0]}"
    finally:
        tmp.unlink(missing_ok=True)

    # stereo file, mono model
    class _MonoModel:
        samplerate = 44100
        audio_channels = 1

    audio_stereo = _make_sine_wav(sr, duration=0.25, channels=2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        mac.save(str(tmp), audio_stereo, sr, encoding="float32")
        wav = _load_audio(tmp, _MonoModel())
        assert wav.shape[0] == 1, f"expected 1 channel, got {wav.shape[0]}"
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    _assert_resample_basic()
    _assert_resample_soxr_vhq()
    _assert_resample_identity()
    _assert_load_audio_resamples()
    _assert_load_audio_no_resample()
    _assert_load_audio_channel_remix()
    print("test_resample.py: OK")
