"""Convert PyTorch models to MLX models."""

import logging
import os
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
    """Get or create the MLX model cache directory.

    ``DEMUCS_MLX_CACHE_DIR`` wins, then the optional mlx-weights shared cache,
    then ``~/.cache/demucs-mlx``. Other tools may read that directory: it holds
    ``<model>.safetensors`` and ``<model>_config.json`` in cache format 1, the
    same files published at https://huggingface.co/ssmall256/demucs-mlx.
    """
    override = os.environ.get("DEMUCS_MLX_CACHE_DIR", "").strip()
    if override:
        cache_dir = Path(override).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    cache_dir_provider = _mlx_weights_cache_dir()
    if cache_dir_provider is not None:
        cache_dir = cache_dir_provider
        return Path(cache_dir("demucs-mlx"))

    cache_dir = Path.home() / ".cache" / "demucs-mlx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_mlx_model(name: str, repo: tp.Optional[Path] = None):
    """
    Get an MLX model from the cache, filling the cache first if needed.

    A missing or unusable cache is filled by downloading the published weights
    (verified against digests shipped in this package) and, if that is disabled
    or fails, by converting the official Demucs checkpoint locally.
    """
    from . import hub
    from .mlx_convert import SafeCacheError, convert_htdemucs_weights, load_mlx_model

    cache_dir = get_mlx_cache_dir()

    try:
        # auto_convert=False fails fast on a miss so the steps below stay explicit.
        return load_mlx_model(name, cache_dir=str(cache_dir), auto_convert=False, verbose=False)
    except FileNotFoundError:
        logger.info("Cache miss for '%s'.", name)
    except SafeCacheError as exc:
        # A cache that exists but cannot be trusted. Every cache written before
        # 1.4.6 lacks the fields the hardened loader requires, so without this
        # branch upgrading turned a self-healing cache miss into a hard failure
        # for every existing user. Replacing it is also the right response to a
        # digest mismatch: discard the suspect files and fetch verified ones.
        logger.info("Unusable cache for '%s' (%s). Replacing it...", name, exc)

    download_problem = "downloading is disabled"
    if name not in hub.PUBLISHED:
        download_problem = "no published weights for this model"
    elif hub.downloads_enabled():
        try:
            hub.download_model(name, cache_dir)
            return load_mlx_model(
                name, cache_dir=str(cache_dir), auto_convert=False, verbose=False
            )
        except hub.DownloadError as exc:
            download_problem = str(exc)
            logger.warning("%s. Converting from the official Demucs checkpoint instead.", exc)

    try:
        convert_htdemucs_weights(
            name,
            output_dir=str(cache_dir),
            verify=True,
            verbose=True,
        )
    except ImportError as exc:
        raise ImportError(
            f"No MLX weights for '{name}' in {cache_dir}. "
            f"Download failed or was skipped ({download_problem}), and local conversion "
            "needs the conversion extra: pip install 'demucs-mlx[convert]'. "
            f"You can also fetch the files yourself from https://huggingface.co/{hub.REPO_ID}."
        ) from exc

    logger.info("Loading converted model...")
    return load_mlx_model(
        name,
        cache_dir=str(cache_dir),
        auto_convert=False,
        verbose=True,
    )
