"""Regression checks for optional mlx-weights integration."""

from __future__ import annotations

import importlib
import tempfile
import types
from pathlib import Path
from unittest import mock

from demucs_mlx.model_converter import get_mlx_cache_dir, get_mlx_model

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

        external_converter = mock.Mock()

        def fake_import(name: str, package: str | None = None):
            if name != "mlx_weights":
                return _REAL_IMPORT(name, package)
            return types.SimpleNamespace(
                cache_dir=lambda project: root / ".cache" / "mlx-weights" / project,
                resolve_converted_model=external_converter,
            )

        with mock.patch("importlib.import_module", side_effect=fake_import):
            cache_dir = get_mlx_cache_dir()

        assert cache_dir == root / ".cache" / "mlx-weights" / "demucs-mlx"
        external_converter.assert_not_called()


def test_cache_miss_uses_internal_safe_converter() -> None:
    sentinel = object()
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        with (
            mock.patch(
                "demucs_mlx.model_converter.get_mlx_cache_dir",
                return_value=cache,
            ),
            mock.patch(
                "demucs_mlx.mlx_convert.load_mlx_model",
                side_effect=[FileNotFoundError("missing"), sentinel],
            ) as loader,
            mock.patch("demucs_mlx.mlx_convert.convert_htdemucs_weights") as converter,
        ):
            model = get_mlx_model("htdemucs")

    assert model is sentinel
    converter.assert_called_once_with(
        "htdemucs",
        output_dir=str(cache),
        verify=False,
        verbose=True,
    )
    assert loader.call_count == 2


if __name__ == "__main__":
    test_falls_back_without_mlx_weights()
    test_uses_mlx_weights_when_available()
    test_cache_miss_uses_internal_safe_converter()
    print("test_model_converter_optional_mlx_weights.py: OK")
