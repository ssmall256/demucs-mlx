"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _no_network_downloads(monkeypatch):
    """Tests never reach the network or a user-configured cache directory."""
    monkeypatch.setenv("DEMUCS_MLX_NO_DOWNLOAD", "1")
    monkeypatch.delenv("DEMUCS_MLX_CACHE_DIR", raising=False)
