import typing as tp
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_audio(path, *, sr: int, layout: str = "channels_first", dtype: str = "float32"):
    """Load through mlx-audio-io with a useful error for a broken native binding."""
    import mlx_audio_io as mac

    try:
        return mac.load(str(path), sr=sr, layout=layout, dtype=dtype)
    except TypeError as exc:
        if "Unable to convert function return value to a Python type" not in str(exc):
            raise
        try:
            from mlx_audio_io._native_loader import load_build_info

            build = load_build_info()
            pairing = (
                f" (built for MLX {build.get('build_mlx_version')} with "
                f"nanobind {build.get('build_nanobind_version')})"
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            pairing = ""
        raise RuntimeError(
            "mlx-audio-io could not return an MLX array"
            f"{pairing}. Rebuild it with the nanobind version used by the "
            "installed MLX runtime."
        ) from exc


def prevent_clip(wav, mode='rescale'):
    """Prevent clipping in torch tensors."""
    import torch
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _prevent_clip_numpy(wav: np.ndarray, mode: str):
    """Prevent clipping in numpy arrays."""
    if mode is None or mode == 'none':
        return wav
    if not np.issubdtype(wav.dtype, np.floating):
        raise AssertionError("too late for clipping")
    if mode == 'rescale':
        max_val = float(np.max(np.abs(wav)))
        scale = max(1.01 * max_val, 1.0)
        wav = wav / scale
    elif mode == 'clamp':
        wav = np.clip(wav, -0.99, 0.99)
    elif mode == 'tanh':
        wav = np.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _prevent_clip_mlx(wav: mx.array, mode: str):
    """Prevent clipping using MLX ops (keeps data on GPU)."""
    if mode is None or mode == 'none':
        return wav
    if mode == 'rescale':
        max_val = mx.max(mx.abs(wav))
        scale = mx.maximum(1.01 * max_val, 1.0)
        wav = wav / scale
    elif mode == 'clamp':
        wav = mx.clip(wav, -0.99, 0.99)
    elif mode == 'tanh':
        wav = mx.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav,
               path: tp.Union[str, Path],
               samplerate: int,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False):
    """
    Save audio file using mlx_audio_io.
    Supports np.ndarray, mlx.core.array, and torch.Tensor.
    """
    import mlx_audio_io as mac
    path = Path(path)

    # Determine encoding
    if as_float:
        encoding = "float32"
    else:
        encoding = "pcm16" if bits_per_sample == 16 else "float32"

    # --- MLX HANDLING (Optimized) ---
    if isinstance(wav, mx.array):
        wav_mx = _prevent_clip_mlx(wav, mode=clip)
        mac.save(str(path), wav_mx, samplerate, encoding=encoding, clip=(clip != 'none'))
    # --- NUMPY HANDLING ---
    elif isinstance(wav, np.ndarray):
        wav_np = wav
        if np.issubdtype(wav_np.dtype, np.floating):
            wav_np = _prevent_clip_numpy(wav_np, mode=clip)
        mac.save(str(path), wav_np, samplerate, encoding=encoding, clip=False)
    # --- TORCH HANDLING (lazy import) ---
    else:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = prevent_clip(wav, mode=clip)
            wav_np = wav.detach().cpu().numpy()
            mac.save(str(path), wav_np, samplerate, encoding=encoding, clip=False)
        else:
            raise TypeError(f"Unsupported audio type: {type(wav)}")
