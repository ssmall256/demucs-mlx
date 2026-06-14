"""Regression checks for optional mlx-weights integration."""

from __future__ import annotations

import importlib
import tempfile
import types
from pathlib import Path
from unittest import mock

from demucs_mlx.model_converter import get_mlx_cache_dir

_REAL_IMPORT = importlib.import_module


def _import_without_mlx_weights(name: str, package: str | None = None):
    if name == "mlx_weights":
        raise ModuleNotFoundError(name)
    return _REAL_IMPORT(name, package)


def test_falls_back_without_mlx_weights() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        with (
            mock.patch("importlib.import_module", side_effect=_import_without_mlx_weights),
            mock.patch("pathlib.Path.home", return_value=home),
        ):
            cache_dir = get_mlx_cache_dir()

        assert cache_dir == home / ".cache" / "demucs-mlx"
        assert cache_dir.exists()


def test_uses_mlx_weights_when_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def fake_import(name: str, package: str | None = None):
            if name != "mlx_weights":
                return _REAL_IMPORT(name, package)
            return types.SimpleNamespace(
                cache_dir=lambda project: root / ".cache" / "mlx-weights" / project,
                resolve_converted_model=lambda *args, **kwargs: None,
            )

        with mock.patch("importlib.import_module", side_effect=fake_import):
            cache_dir = get_mlx_cache_dir()

        assert cache_dir == root / ".cache" / "mlx-weights" / "demucs-mlx"


if __name__ == "__main__":
    test_falls_back_without_mlx_weights()
    test_uses_mlx_weights_when_available()
    print("test_model_converter_optional_mlx_weights.py: OK")
