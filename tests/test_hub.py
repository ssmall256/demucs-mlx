"""Published-weight download: verification, failure handling and load order."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from demucs_mlx import hub, model_converter


def _publish(directory: Path, name: str, weights: bytes, config: bytes) -> hub.PublishedModel:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.safetensors").write_bytes(weights)
    (directory / f"{name}_config.json").write_bytes(config)
    return hub.PublishedModel(
        hub.PublishedFile(len(weights), hashlib.sha256(weights).hexdigest()),
        hub.PublishedFile(len(config), hashlib.sha256(config).hexdigest()),
    )


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A local 'hub' holding one tiny model, with downloads enabled."""
    source = tmp_path / "hub"
    published = _publish(source, "htdemucs", b"w" * 5000, b'{"k": 1}')
    monkeypatch.setattr(hub, "PUBLISHED", {"htdemucs": published})
    monkeypatch.setenv("DEMUCS_MLX_HUB_URL", source.as_uri())
    monkeypatch.delenv("DEMUCS_MLX_NO_DOWNLOAD", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return source


def test_published_table_is_well_formed():
    from demucs_mlx.mlx_registry import MLX_MODEL_REGISTRY

    assert set(hub.PUBLISHED) == set(MLX_MODEL_REGISTRY)
    for model in hub.PUBLISHED.values():
        for item in model:
            assert item.size > 0
            assert len(item.sha256) == 64 and int(item.sha256, 16) >= 0


def test_download_writes_verified_files(served, tmp_path):
    cache = tmp_path / "cache"
    weights = hub.download_model("htdemucs", cache, progress=False)
    assert weights.read_bytes() == b"w" * 5000
    assert (cache / "htdemucs_config.json").read_bytes() == b'{"k": 1}'
    assert sorted(p.name for p in cache.iterdir()) == [
        "htdemucs.safetensors",
        "htdemucs_config.json",
    ]
    assert hub.verify_directory(cache) == []


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: (d / "htdemucs.safetensors").write_bytes(b"x" * 5000),
        lambda d: (d / "htdemucs.safetensors").write_bytes(b"w" * 4999),
        lambda d: (d / "htdemucs.safetensors").write_bytes(b"w" * 9000),
        lambda d: (d / "htdemucs_config.json").write_bytes(b'{"k": 2}'),
        lambda d: (d / "htdemucs.safetensors").unlink(),
    ],
    ids=["wrong-bytes", "truncated", "oversized", "wrong-config", "missing"],
)
def test_bad_download_leaves_cache_untouched(served, tmp_path, tamper):
    tamper(served)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "htdemucs.safetensors").write_bytes(b"existing")
    with pytest.raises(hub.DownloadError):
        hub.download_model("htdemucs", cache, progress=False)
    assert [p.name for p in cache.iterdir()] == ["htdemucs.safetensors"]
    assert (cache / "htdemucs.safetensors").read_bytes() == b"existing"


def test_unpublished_model_is_refused(served, tmp_path):
    with pytest.raises(hub.DownloadError, match="No published"):
        hub.download_model("mdx", tmp_path / "cache", progress=False)


@pytest.mark.parametrize("variable", ["DEMUCS_MLX_NO_DOWNLOAD", "HF_HUB_OFFLINE"])
def test_offline_switches(monkeypatch, variable):
    monkeypatch.delenv("DEMUCS_MLX_NO_DOWNLOAD", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert hub.downloads_enabled()
    monkeypatch.setenv(variable, "1")
    assert not hub.downloads_enabled()


def test_base_url(monkeypatch):
    monkeypatch.delenv("DEMUCS_MLX_HUB_URL", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert hub.base_url() == "https://huggingface.co/ssmall256/demucs-mlx/resolve/main"
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/")
    assert hub.base_url() == "https://mirror.example/ssmall256/demucs-mlx/resolve/main"
    monkeypatch.setenv("DEMUCS_MLX_HUB_URL", "https://files.example/weights/")
    assert hub.base_url() == "https://files.example/weights"


def test_cache_dir_override(tmp_path, monkeypatch):
    target = tmp_path / "shared" / "demucs"
    monkeypatch.setenv("DEMUCS_MLX_CACHE_DIR", str(target))
    assert model_converter.get_mlx_cache_dir() == target
    assert target.is_dir()


def _patch_loading(monkeypatch, tmp_path, *, convert):
    """Make get_mlx_model use a temp cache, a fake loader and a fake converter."""
    import demucs_mlx.mlx_convert as mlx_convert

    cache = tmp_path / "cache"
    calls = {"converted": 0}

    def load(name, cache_dir, **_):
        if not (Path(cache_dir) / f"{name}.safetensors").exists():
            raise FileNotFoundError(name)
        return f"model:{name}"

    def fake_convert(name, output_dir, **_):
        calls["converted"] += 1
        convert(name, Path(output_dir))

    monkeypatch.setattr(model_converter, "get_mlx_cache_dir", lambda: cache)
    monkeypatch.setattr(mlx_convert, "load_mlx_model", load)
    monkeypatch.setattr(mlx_convert, "convert_htdemucs_weights", fake_convert)
    return cache, calls


def _write_weights(name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.safetensors").write_bytes(b"converted")


def test_cache_miss_downloads_before_converting(served, tmp_path, monkeypatch):
    cache, calls = _patch_loading(monkeypatch, tmp_path, convert=_write_weights)
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 0
    assert (cache / "htdemucs.safetensors").read_bytes() == b"w" * 5000


def test_failed_download_falls_back_to_conversion(served, tmp_path, monkeypatch):
    (served / "htdemucs.safetensors").write_bytes(b"x" * 5000)
    cache, calls = _patch_loading(monkeypatch, tmp_path, convert=_write_weights)
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 1
    assert (cache / "htdemucs.safetensors").read_bytes() == b"converted"


def test_offline_goes_straight_to_conversion(served, tmp_path, monkeypatch):
    monkeypatch.setenv("DEMUCS_MLX_NO_DOWNLOAD", "1")
    cache, calls = _patch_loading(monkeypatch, tmp_path, convert=_write_weights)
    assert model_converter.get_mlx_model("htdemucs") == "model:htdemucs"
    assert calls["converted"] == 1


def test_no_download_and_no_converter_explains_both_options(served, tmp_path, monkeypatch):
    def missing_extra(name, directory):
        raise ImportError("torch")

    (served / "htdemucs.safetensors").unlink()
    _patch_loading(monkeypatch, tmp_path, convert=missing_extra)
    with pytest.raises(ImportError) as caught:
        model_converter.get_mlx_model("htdemucs")
    message = str(caught.value)
    assert "demucs-mlx[convert]" in message and "huggingface.co/ssmall256/demucs-mlx" in message
