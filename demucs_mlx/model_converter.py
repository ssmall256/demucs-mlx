"""Convert PyTorch models to MLX models."""

import logging
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)


def _mlx_weights_api() -> tp.Optional[tuple[tp.Callable[[str], Path], tp.Callable[..., tp.Any]]]:
    """Return the optional mlx-weights integration when it is installed."""
    import importlib

    try:
        mlx_weights = importlib.import_module("mlx_weights")
    except ModuleNotFoundError:
        return None

    cache_dir = getattr(mlx_weights, "cache_dir", None)
    resolve_converted_model = getattr(mlx_weights, "resolve_converted_model", None)
    if not callable(cache_dir) or not callable(resolve_converted_model):
        logger.warning("Ignoring installed mlx-weights package with an unsupported API.")
        return None
    return tp.cast(tp.Callable[[str], Path], cache_dir), resolve_converted_model


def get_mlx_cache_dir() -> Path:
    """Get or create the MLX model cache directory."""
    mlx_weights = _mlx_weights_api()
    if mlx_weights is not None:
        cache_dir, _ = mlx_weights
        return Path(cache_dir("demucs-mlx"))

    cache_dir = Path.home() / '.cache' / 'demucs-mlx'
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
        mlx_weights = _mlx_weights_api()
        if mlx_weights is not None:
            _, resolve_converted_model = mlx_weights
            logger.info("Cache miss for '%s'. Converting through mlx-weights...", name)
            resolve_converted_model(f"demucs/{name}", convert_if_missing=True)
        else:
            logger.info("Cache miss for '%s'. Converting from PyTorch...", name)

            # This step might take a few seconds but only happens once.
            convert_htdemucs_weights(
                name,
                output_dir=str(cache_dir),
                verify=False,
                verbose=True
            )
        
        # Load the newly converted model
        logger.info("Loading converted model...")
        model = load_mlx_model(name, cache_dir=str(cache_dir), auto_convert=False, verbose=True)
        return model
