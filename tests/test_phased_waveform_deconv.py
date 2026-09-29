"""Check the phased waveform decoder against MLX transposed convolution."""

import mlx.core as mx
import numpy as np
import pytest

from demucs_mlx.mlx_layers import ConvTranspose1dNCL


def original(layer, x):
    return layer.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


@pytest.mark.parametrize("shape", [(1, 3, 1), (2, 3, 9), (2, 3, 1001)])
@pytest.mark.parametrize("bias", [False, True])
def test_phased_deconvolution_matches_original(shape, bias):
    rng = np.random.default_rng(481)
    layer = ConvTranspose1dNCL(3, 2, 8, stride=4, bias=bias)
    x = mx.array(rng.standard_normal(shape, dtype=np.float32))
    want = np.asarray(original(layer, x))
    got = np.asarray(layer(x))
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert layer._phased_cache is not None
    assert "_phased_cache" not in layer

    old_cache = layer._phased_cache
    layer.conv.weight = layer.conv.weight * 0.5
    want = np.asarray(original(layer, x))
    got = np.asarray(layer(x))
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    assert layer._phased_cache is not old_cache


def test_other_kernel_and_stride_use_original_path():
    layer = ConvTranspose1dNCL(3, 2, 4, stride=2)
    x = mx.ones((1, 3, 17))
    np.testing.assert_array_equal(np.asarray(layer(x)), np.asarray(original(layer, x)))
    assert layer._phased_cache is None


def test_half_precision_uses_original_path():
    layer = ConvTranspose1dNCL(3, 2, 8, stride=4)
    layer.conv.weight = layer.conv.weight.astype(mx.float16)
    layer.conv.bias = layer.conv.bias.astype(mx.float16)
    x = mx.ones((1, 3, 9), dtype=mx.float16)
    np.testing.assert_array_equal(np.asarray(layer(x)), np.asarray(original(layer, x)))
    assert layer._phased_cache is None
