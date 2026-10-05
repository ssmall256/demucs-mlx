"""The split placement preserves arithmetic and compiled live parameters."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from demucs_mlx.mlx_demucs import DConv, _dconv_block_forward_nlc
from demucs_mlx.mlx_layers import _conv_channels_last


def original(block, x):
    n, length, _ = x.shape
    norm1, norm2 = block.layers[1], block.layers[4]
    h = block.layers[0].conv(x)
    h = mx.fast.layer_norm(h.reshape(n, 1, -1), None, None, norm1.eps).reshape(h.shape)
    h = h * norm1.weight + norm1.bias
    o = _conv_channels_last(block.layers[3].conv, nn.gelu(h))
    o = mx.fast.layer_norm(o.reshape(n, 1, -1), None, None, norm2.eps).reshape(o.shape)
    if norm2.affine:
        o = o * norm2.weight + norm2.bias
    a, b = mx.split(o, 2, axis=-1)
    return a * mx.sigmoid(b) * block.layers[6].scale


@pytest.mark.parametrize("depth", [2, -2])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("strided", [False, True])
def test_affine_gate_matches_original(depth, affine, strided):
    mx.random.seed(481)
    model = DConv(8, compress=2, depth=depth).eval()
    for layer in model.layers:
        for index in [1, 4]:
            norm = layer.layers[index]
            norm.weight = mx.linspace(-0.7, 1.3, norm.weight.size)
            norm.bias = mx.linspace(-0.3, 0.4, norm.bias.size)
        layer.layers[4].affine = affine
        layer.layers[6].scale = mx.linspace(-0.5, 0.8, 8)
    x = mx.random.normal((2, 130, 8))
    if strided:
        x = x[:, 1::2, :]
    for block in model.layers:
        reference = original(block, x)
        actual = _dconv_block_forward_nlc(block, x)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        old = mx.compile(lambda value: value + original(block, value))
        new = mx.compile(lambda value: value + _dconv_block_forward_nlc(block, value))
        np.testing.assert_array_equal(np.asarray(new(x)), np.asarray(old(x)))


def test_affine_parameter_replacement_invalidates_chain(monkeypatch):
    mx.random.seed(481)
    model = DConv(8, compress=2).eval()
    x = mx.random.normal((2, 8, 65))
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "1")
    previous = np.asarray(model(x))
    graph = model._compiled_layers
    norm = model.layers[0].layers[4]
    norm.weight = norm.weight * 0.75
    norm.bias = mx.linspace(-0.3, 0.3, 16)
    compiled = np.asarray(model(x))
    assert model._compiled_layers is not graph
    assert np.max(np.abs(previous - compiled)) > 1e-6
    monkeypatch.setenv("DEMUCS_MLX_COMPILE_DCONV", "0")
    np.testing.assert_array_equal(compiled, np.asarray(model(x)))


def test_float16_affine_keeps_original_path():
    model = DConv(8, compress=2).eval()
    block = model.layers[0]
    norm = block.layers[4]
    norm.weight = mx.linspace(-0.7, 1.3, norm.weight.size).astype(mx.float16)
    norm.bias = mx.linspace(-0.3, 0.4, norm.bias.size).astype(mx.float16)
    x = mx.random.normal((2, 65, 8))
    np.testing.assert_array_equal(
        np.asarray(_dconv_block_forward_nlc(block, x)), np.asarray(original(block, x))
    )
    old = mx.compile(lambda value: value + original(block, value))
    new = mx.compile(lambda value: value + _dconv_block_forward_nlc(block, value))
    np.testing.assert_array_equal(np.asarray(new(x)), np.asarray(old(x)))
