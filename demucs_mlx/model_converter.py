"""Convert PyTorch models to MLX models."""

import logging
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)


def _mlx_weights_cache_dir() -> tp.Optional[tp.Callable[[str], Path]]:
    """Return the optional mlx-weights cache-directory integration."""
    import importlib

    try:
        mlx_weights = importlib.import_module("mlx_weights")
    except ModuleNotFoundError:
        return None

    cache_dir = getattr(mlx_weights, "cache_dir", None)
    if not callable(cache_dir):
        logger.warning("Ignoring installed mlx-weights package with an unsupported API.")
        return None
    return tp.cast(tp.Callable[[str], Path], cache_dir)


def get_mlx_cache_dir() -> Path:
    """Get or create the MLX model cache directory."""
    cache_dir_provider = _mlx_weights_cache_dir()
    if cache_dir_provider is not None:
        cache_dir = cache_dir_provider
        return Path(cache_dir("demucs-mlx"))

    cache_dir = Path.home() / ".cache" / "demucs-mlx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_mlx_model(name: str, repo: tp.Optional[Path] = None):
    """
    Get an MLX model, loading from cache or converting from PyTorch if needed.
    """
    from .mlx_convert import convert_htdemucs_weights, load_mlx_model

    cache_dir = get_mlx_cache_dir()

    # NOTE: load_mlx_model should handle caching, but we wrap it
    # to ensure we don't accidentally trigger conversion logic on every run.
    try:
        # auto_convert=False ensures we fail fast if not found,
        # allowing us to handle the conversion step explicitly below.
        model = load_mlx_model(name, cache_dir=str(cache_dir), auto_convert=False, verbose=False)
        return model
    except FileNotFoundError:
        # If we are here, the model is missing.
        logger.info("Cache miss for '%s'. Converting from the official registry...", name)
        convert_htdemucs_weights(
            name,
            output_dir=str(cache_dir),
            verify=False,
            verbose=True,
        )

        # Load the newly converted model
        logger.info("Loading converted model...")
        return load_mlx_model(
            name,
            cache_dir=str(cache_dir),
            auto_convert=False,
            verbose=True,
        )
