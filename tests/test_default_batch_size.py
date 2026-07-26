"""Checks for the memory-scaled default batch size."""
from __future__ import annotations

from demucs_mlx.apply_mlx import default_batch_size


def _assert_ladder() -> None:
    # 32 GB machines report ~25 GiB recommended working sets: measured
    # optimum is 2 (batch 8 thrashes, ~5x slower end to end).
    assert default_batch_size(25 << 30) == 2
    # 64 GB class
    assert default_batch_size(48 << 30) == 4
    # 96-128 GB class keeps the previous default of 8.
    assert default_batch_size(96 << 30) == 8
    assert default_batch_size(200 << 30) == 8
    # Degenerate/unknown sizes stay conservative.
    assert default_batch_size(0) == 2


def _assert_device_detection() -> None:
    resolved = default_batch_size()
    assert isinstance(resolved, int)
    assert resolved >= 1


def _assert_separator_resolution() -> None:
    from demucs_mlx.api import Separator

    auto = Separator()
    assert auto.batch_size >= 1

    explicit = Separator(batch_size=3)
    assert explicit.batch_size == 3

    try:
        Separator(batch_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("batch_size=0 must raise")


def test_default_batch_size() -> None:
    _assert_ladder()
    _assert_device_detection()
    _assert_separator_resolution()
