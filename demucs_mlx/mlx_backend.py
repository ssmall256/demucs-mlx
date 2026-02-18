from __future__ import annotations

import typing as tp

MIN_MLX_VERSION = "0.30.3"

def _get_mlx_version() -> tp.Optional[str]:
    try:
        import mlx  # type: ignore
    except ImportError:
        return None
    return getattr(mlx, "__version__", None)

def ensure_mlx_version() -> None:
    version = _get_mlx_version()
    if version is None:
        return
    if version < MIN_MLX_VERSION:
        print(f"Warning: MLX version {version} is older than recommended {MIN_MLX_VERSION}.")

def _require_mlx():
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("mlx is required for the MLX backend.") from exc
    return mx

def is_mx_array(x: tp.Any) -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return isinstance(x, mx.array)
    except ImportError:
        return False

def resample_mx(
    x,
    orig_freq: int,
    new_freq: int,
    quality: str = "high",
):
    """
    Resample MLX array using mlx-audio-io's native resampling.

    Args:
        x: Input MLX array with shape (channels, frames) or (batch, channels, frames)
        orig_freq: Original sample rate
        new_freq: Target sample rate
        quality: Resampling quality ("fastest", "low", "medium", "high", "best", "default")

    Returns:
        Resampled MLX array with same shape layout
    """
    import tempfile
    from pathlib import Path

    import mlx_audio_io as mac
    import numpy as np

    mx = _require_mlx()

    if orig_freq == new_freq:
        return x

    original_shape = x.shape

    # Handle batch dimension if present
    if len(original_shape) == 3:
        batch_size = original_shape[0]
        outputs = []
        for i in range(batch_size):
            resampled = resample_mx(x[i], orig_freq, new_freq, quality)
            outputs.append(resampled)
        return mx.stack(outputs, axis=0)
    elif len(original_shape) == 2:
        pass
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {original_shape}")

    # Convert to numpy and transpose to (frames, channels) for mlx-audio-io
    x_np = np.array(x, copy=False).T

    # Use mlx-audio-io via temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    try:
        mac.save(temp_path, x_np, orig_freq, encoding="float32", clip=False)
        resampled, _ = mac.load(temp_path, sr=new_freq, resample_quality=quality)
        result = mx.array(resampled).T
        return result
    finally:
        Path(temp_path).unlink(missing_ok=True)
