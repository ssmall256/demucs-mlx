"""Check grouped fast LayerNorm against the former GroupNorm reduction."""

import mlx.core as mx
import numpy as np
import pytest

from demucs_mlx.mlx_demucs import GroupNorm
from demucs_mlx.mlx_layers import GroupNormNCHW, GroupNormNCL


def previous_group_norm(x, groups, eps, weight, bias):
    batch, channels = x.shape[:2]
    grouped = x.reshape(batch, groups, channels // groups, *x.shape[2:])
    axes = tuple(range(2, grouped.ndim))
    mean = grouped.mean(axis=axes, keepdims=True)
    variance = grouped.var(axis=axes, keepdims=True)
    normalized = ((grouped - mean) * mx.rsqrt(variance + eps)).reshape(x.shape)
    if weight is None:
        return normalized
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * weight.reshape(affine_shape) + bias.reshape(affine_shape)


@pytest.mark.parametrize(
    ("layer_type", "shape", "groups"),
    [
        (GroupNorm, (2, 12, 257), 3),
        (GroupNormNCL, (2, 12, 257), 3),
        (GroupNormNCHW, (2, 8, 13, 17), 4),
    ],
)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_group_norm_matches_previous_reduction(layer_type, shape, groups, affine, dtype):
    rng = np.random.default_rng(481)
    x = mx.array(rng.normal(size=shape).astype(np.float32), dtype=dtype)
    layer = layer_type(groups, shape[1], affine=affine)
    if affine:
        layer.weight = mx.array(rng.normal(loc=1, scale=0.2, size=shape[1]), dtype=dtype)
        layer.bias = mx.array(rng.normal(scale=0.2, size=shape[1]), dtype=dtype)
    want = np.asarray(previous_group_norm(x, groups, layer.eps, layer.weight, layer.bias))
    got = np.asarray(layer(x))
    tolerance = 2e-2 if dtype == mx.float16 else 3e-5
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("layer_type", [GroupNorm, GroupNormNCL, GroupNormNCHW])
def test_group_norm_rejects_indivisible_channels(layer_type):
    layer = layer_type(3, 8)
    x = mx.zeros((1, 8, 17) if layer_type != GroupNormNCHW else (1, 8, 3, 5))
    with pytest.raises(ValueError, match="not divisible"):
        layer(x)
