"""An unusable cache must regenerate, not raise.

1.4.6 hardened the safetensors cache: the loader now requires fields that no
cache written before it contains. `get_mlx_model` only caught
`FileNotFoundError`, so a cache that existed but could not be validated raised
`SafeCacheError` straight out to the caller -- turning a self-healing cache
miss into a hard failure for every user upgrading past 1.4.6.
"""

from __future__ import annotations

import pytest

from demucs_mlx import model_converter
from demucs_mlx.mlx_convert import SafeCacheError


@pytest.fixture
def converting(monkeypatch, tmp_path):
    """Record whether conversion ran, and what the second load was asked for."""
    calls = {"converted": 0, "loads": 0}

    monkeypatch.setattr(model_converter, "get_mlx_cache_dir", lambda: tmp_path)

    def fake_convert(*args, **kwargs):
        calls["converted"] += 1

    def make_loader(first_error):
        def fake_load(name, **kwargs):
            calls["loads"] += 1
            if calls["loads"] == 1 and first_error is not None:
                raise first_error
            return f"model:{name}"

        return fake_load

    def install(first_error):
        monkeypatch.setattr(
            model_converter,
            "_imports",
            None,
            raising=False,
        )
        import demucs_mlx.mlx_convert as mlx_convert

        monkeypatch.setattr(mlx_convert, "convert_htdemucs_weights", fake_convert)
        monkeypatch.setattr(mlx_convert, "load_mlx_model", make_loader(first_error))
        return calls

    return install


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("no cache"),
        SafeCacheError("Demucs cache config is missing fields: ['format_version']"),
        SafeCacheError("Safetensors digest mismatch for htdemucs.safetensors"),
    ],
    ids=["missing", "stale-schema", "digest-mismatch"],
)
def test_unusable_cache_is_regenerated(converting, error):
    calls = converting(error)

    result = model_converter.get_mlx_model("htdemucs")

    assert result == "model:htdemucs"
    assert calls["converted"] == 1, "conversion should have been triggered"
    assert calls["loads"] == 2, "the regenerated cache should be loaded"


def test_a_usable_cache_does_not_reconvert(converting):
    calls = converting(None)

    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 0
    assert calls["loads"] == 1
