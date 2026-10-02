"""Compiled DConv inference preserves values and its parameter tree."""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from demucs_mlx.mlx_demucs import DConv


def test_compiled_dconv_matches_eager_and_tracks_weights(monkeypatch):
    mx.random.seed(481)
    block = DConv(8, compress=2, depth=2).eval()
    value = mx.random.normal((2, 8, 128))
    mx.eval(value)
    parameter_names = [name for name, _ in tree_flatten(block.parameters())]

    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "0")
    eager = np.asarray(block(value))
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "1")
    compiled = np.asarray(block(value))
    np.testing.assert_array_equal(compiled, eager)
    assert [name for name, _ in tree_flatten(block.parameters())] == parameter_names

    conv = block.layers[0].layers[0].conv
    conv.weight = conv.weight * 0.5
    block.layers[0].layers[6].scale = mx.ones((8,))
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "0")
    eager_updated = np.asarray(block(value))
    assert np.max(np.abs(eager_updated - eager)) > 1e-5
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "1")
    compiled_updated = np.asarray(block(value))
    np.testing.assert_allclose(compiled_updated, eager_updated, rtol=1e-6, atol=1e-6)


def test_dconv_compile_auto_suppression(monkeypatch):
    from demucs_mlx.mlx_utils import is_dconv_compile_enabled, outer_compile_context

    monkeypatch.delenv("DEMUCS_MLX_COMPILE_DCONV", raising=False)

    # 1. Default (no outer compile): DConv compile should be enabled
    assert is_dconv_compile_enabled() is True

    # 2. Inside outer compile context: DConv compile should be automatically suppressed
    with outer_compile_context(True):
        assert is_dconv_compile_enabled() is False

    # 3. Outside context again: back to True
    assert is_dconv_compile_enabled() is True

    # 4. Explicit user override to 1 inside outer compile context
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "1")
    with outer_compile_context(True):
        assert is_dconv_compile_enabled() is True

    # 5. Explicit user override to 0 outside outer compile context
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "0")
    assert is_dconv_compile_enabled() is False
    with outer_compile_context(True):
        assert is_dconv_compile_enabled() is False

