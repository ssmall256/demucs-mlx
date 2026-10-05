"""Pointwise GEMM preserves convolution semantics and parameter updates."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from demucs_mlx.mlx_layers import Conv1dNCL, Conv2dNCHW


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("case", ["projection", "grouped", "stride", "padding", "spatial", "half"])
def test_projection_matches_convolution_and_tracks_weights(frequency, bias, case):
    options = dict(bias=bias)
    kernel = 1
    if case == "grouped":
        options["groups"] = 2
    elif case == "stride":
        options["stride"] = (2, 1) if frequency else 2
    elif case == "padding":
        options["padding"] = (0, 1) if frequency else 1
    elif case == "spatial":
        kernel = (1, 3) if frequency else 3
    layer = (Conv2dNCHW if frequency else Conv1dNCL)(6, 12, kernel, **options)
    dtype = mx.float16 if case == "half" else mx.float32
    layer.conv.weight = (
        mx.cos(mx.arange(layer.conv.weight.size) * 0.07)
        .reshape(layer.conv.weight.shape)
        .astype(dtype)
    )
    if bias:
        layer.conv.bias = mx.sin(mx.arange(12) * 0.13).astype(dtype)
    # Channels-first input is a view of channels-last storage.
    axes = (0, 3, 1, 2) if frequency else (0, 2, 1)
    input_axes = (0, 2, 3, 1) if frequency else (0, 2, 1)
    x = (
        mx.sin(mx.arange(2 * 99 * 6) * 0.09)
        .reshape((2, 9, 11, 6) if frequency else (2, 99, 6))
        .astype(dtype)
        .transpose(axes)
    )
    names = [name for name, _ in tree_flatten(layer.parameters())]
    actual = np.asarray(layer(x))
    reference = np.asarray(layer.conv(x.transpose(input_axes)).transpose(axes))
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
    assert [name for name, _ in tree_flatten(layer.parameters())] == names
    layer.conv.weight = layer.conv.weight * 0.5
    updated = np.asarray(layer(x))
    reference = np.asarray(layer.conv(x.transpose(input_axes)).transpose(axes))
    np.testing.assert_allclose(updated, reference, rtol=2e-5, atol=2e-5)
    assert np.max(np.abs(updated - actual)) > 1e-4
