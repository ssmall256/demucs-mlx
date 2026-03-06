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
    import mlx_audio_io as mac

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

    # x is (channels, frames); mac.resample expects (frames, channels)
    resampled = mac.resample(mx.transpose(x, (1, 0)), orig_freq, new_freq, quality=quality)
    return mx.transpose(resampled, (1, 0))
