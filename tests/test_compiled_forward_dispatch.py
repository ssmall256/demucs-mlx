"""Dispatch and cache behavior for the optional whole-model forward graph."""

import gc

import mlx.core as mx
import pytest

from demucs_mlx import apply_mlx


@pytest.fixture(autouse=True)
def clear_graph_cache():
    apply_mlx._COMPILED_FORWARDS.clear()
    yield
    apply_mlx._COMPILED_FORWARDS.clear()


class _Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return 2 * x + 1


def test_default_stays_eager(monkeypatch):
    monkeypatch.delenv("DEMUCS_MLX_COMPILE_FORWARD", raising=False)
    model = _Model()
    x = mx.ones((2, 2, 64))
    for _ in range(4):
        mx.eval(apply_mlx._forward(model, x))
    assert model.calls == 4
    assert apply_mlx._COMPILED_FORWARDS == {}


def test_opt_in_compiles_after_first_shape_call(monkeypatch):
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_FORWARD", "1")
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "0")
    model = _Model()
    x = mx.ones((2, 2, 64))
    for _ in range(4):
        mx.eval(apply_mlx._forward(model, x))
    assert model.calls == 2  # One eager call and one trace.
    assert len(apply_mlx._COMPILED_FORWARDS[id(model)][1]) == 1


def test_ane_keeps_its_path_when_compile_is_requested(monkeypatch):
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_FORWARD", "1")
    x = mx.ones((2, 2, 64))
    ane = _Model()
    ane._ane_time_conv = object()
    for _ in range(2):
        mx.eval(apply_mlx._forward(ane, x))
    assert ane.calls == 2
    assert id(ane) not in apply_mlx._COMPILED_FORWARDS


def test_explicit_opt_out_and_cache_cleanup(monkeypatch):
    x = mx.ones((2, 2, 64))
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_FORWARD", "0")
    model = _Model()
    mx.eval(apply_mlx._forward(model, x))
    assert apply_mlx._COMPILED_FORWARDS == {}

    monkeypatch.setenv("DEMUCS_MLX_COMPILE_FORWARD", "1")
    mx.eval(apply_mlx._forward(model, x))
    key = id(model)
    assert key in apply_mlx._COMPILED_FORWARDS
    del model
    gc.collect()
    assert key not in apply_mlx._COMPILED_FORWARDS
