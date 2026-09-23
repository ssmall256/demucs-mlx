"""Fused GroupNorm+GELU must match the unfused path at the shapes htdemucs uses.

The test that existed fed a *constant* input, which makes the reduction
order-independent by construction -- so the threadgroup race below could not
show up, and the kernel shipped for seven months producing wrong output on the
frequency-branch DConv shapes.

The kernel reduces in three passes through one `shared_sums` array. Pass 1 ends
with every simdgroup reading `shared_sums[0]` as the mean; pass 2 has simdgroup
0 write that same slot. Without a barrier between them a fast simdgroup
clobbers the mean before a lagging one has loaded it. The window is widest when
pass 2's loop is *short*, i.e. at small `elems_per_group` -- which is exactly
the freq-branch shapes, not the large ones anyone would have worried about.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# Real htdemucs DConv shapes, taken from a module trace. num_groups is always 1
# there (hidden = channels // 8), and the first four are the frequency branch.
SHAPES = [
    (4096, 6, 336),
    (1024, 12, 336),
    (256, 24, 336),
    (64, 48, 336),
    (8, 48, 1344),
    (1, 6, 85995),
]


@pytest.fixture(scope="module")
def fused():
    os.environ["DEMUCS_MLX_USE_FUSED_GN_GLU"] = "1"
    from demucs_mlx import metal_kernels

    if not metal_kernels.HAS_METAL:
        pytest.skip("Metal unavailable")
    return metal_kernels.fused_groupnorm_gelu


def _unfused(x, weight, bias, num_groups, eps=1e-5):
    b, c, length = x.shape
    grouped = x.reshape(b, num_groups, c // num_groups, length)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    var = grouped.var(axis=(2, 3), keepdims=True)
    normed = ((grouped - mean) * mx.rsqrt(var + eps)).reshape(b, c, length)
    return nn.gelu(normed * weight.reshape(1, c, 1) + bias.reshape(1, c, 1))


def _inputs(shape, seed=0):
    b, c, length = shape
    mx.random.seed(seed)
    # Random, NOT constant: a constant input makes the summation
    # order-independent and hides exactly the bug this file exists for.
    x = mx.random.normal(shape)
    weight = mx.random.normal((c,)) + 1.0
    bias = mx.random.normal((c,)) * 0.1
    mx.eval(x, weight, bias)
    return x, weight, bias


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_the_unfused_path(fused, shape):
    x, weight, bias = _inputs(shape)
    got = np.array(fused(x, weight, bias, 1), dtype=np.float64)
    want = np.array(_unfused(x, weight, bias, 1), dtype=np.float64)
    rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
    # MSL has no erf in any namespace, so the kernel uses the Abramowitz &
    # Stegun polynomial, accurate to ~1.5e-7. Nothing should exceed that by
    # much; the race produced 1.6e-02.
    assert rel < 1e-5, f"{shape}: relative error {rel:.2e}"


@pytest.mark.parametrize("shape", SHAPES[:4], ids=lambda s: "x".join(map(str, s)))
def test_is_deterministic(fused, shape):
    """A race shows up as run-to-run variation before it shows up as error."""
    x, weight, bias = _inputs(shape)
    first = np.array(fused(x, weight, bias, 1))
    for _ in range(4):
        again = np.array(fused(x, weight, bias, 1))
        assert np.array_equal(first, again), f"{shape}: output varies between runs"
