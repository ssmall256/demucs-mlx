"""Decoder bias/split fusion preserves arithmetic, fallbacks and live weights."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from demucs_mlx.mlx_hdemucs import HDecLayer
from demucs_mlx.mlx_layers import GLUNCL


def layer(**kwargs):
    result = HDecLayer(8, 4, norm=False, dconv=False, gated_rewrite=True, **kwargs).eval()
    result.rewrite.conv.bias = mx.linspace(-0.4, 0.6, 16)
    return result


def original(model, x):
    if model._fused_norm1:
        return model.norm1(model.rewrite(x))
    return GLUNCL(axis=1)(model.norm1(model.rewrite(x)))


@pytest.mark.parametrize("strided", [False, True])
def test_rewrite_matches_eager_and_compiled(strided, monkeypatch):
    mx.random.seed(481)
    model = layer()
    x = mx.random.normal((2, 8, 15, 34))
    if strided:
        x = x[:, :, ::2, 1::2]
    expected = np.asarray(original(model, x))
    before = mx.compile(lambda v: original(model, v))
    compiled_expected = np.asarray(before(x))

    # A value-only check also passes if an eligibility guard silently falls
    # back. For eligible cases require bypassing the original wrapper.
    def forbidden_wrapper(*args, **kwargs):
        raise AssertionError("Eligible rewrite used the original wrapper")

    monkeypatch.setattr(type(model.rewrite), "__call__", forbidden_wrapper)
    np.testing.assert_array_equal(np.asarray(model._rewrite_glu(x)), expected)
    after = mx.compile(model._rewrite_glu)
    np.testing.assert_array_equal(np.asarray(after(x)), compiled_expected)


@pytest.mark.parametrize(
    "case", ["legacy", "normalized", "float16", "bias16", "context0", "time_only", "multiband"]
)
def test_incompatible_rewrite_keeps_original(case):
    model = layer(
        context=0 if case == "context0" else 1,
        freq=case != "time_only",
        context_freq=case != "multiband",
    )
    if case == "legacy":
        model._gated_rewrite = False
    if case == "normalized":
        from demucs_mlx.mlx_layers import GroupNormNCHW

        model.norm1 = GroupNormNCHW(2, 16)
    if case == "float16":
        model.rewrite.conv.weight = model.rewrite.conv.weight.astype(mx.float16)
        model.rewrite.conv.bias = model.rewrite.conv.bias.astype(mx.float16)
    if case == "bias16":
        model.rewrite.conv.bias = model.rewrite.conv.bias.astype(mx.float16)
    x = mx.random.normal((2, 8, 11) if case == "time_only" else (2, 8, 9, 11))
    if case == "float16":
        x = x.astype(mx.float16)
    np.testing.assert_array_equal(np.asarray(model._rewrite_glu(x)), np.asarray(original(model, x)))
    before, after = mx.compile(lambda v: original(model, v)), mx.compile(model._rewrite_glu)
    np.testing.assert_array_equal(np.asarray(after(x)), np.asarray(before(x)))


def test_live_parameter_replacement_and_names():
    model = layer()
    names = [key for key, _ in tree_flatten(model.parameters())]
    x = mx.random.normal((2, 8, 9, 11))
    previous = np.asarray(model._rewrite_glu(x))
    model.rewrite.conv.weight = model.rewrite.conv.weight * 0.75
    model.rewrite.conv.bias = mx.linspace(-0.6, 0.2, 16)
    actual = np.asarray(model._rewrite_glu(x))
    assert np.max(abs(previous - actual)) > 1e-4
    np.testing.assert_array_equal(actual, np.asarray(original(model, x)))
    assert names == [key for key, _ in tree_flatten(model.parameters())]


def test_only_standard_transformer_frequency_decoders_opt_in():
    from demucs_mlx.mlx_hdemucs import HDemucsMLX
    from demucs_mlx.mlx_htdemucs import HTDemucsMLX

    for model in [
        HTDemucsMLX(["a", "b"], channels=8, depth=4, segment=1),
        HDemucsMLX(["a", "b"], channels=8, depth=4, segment=1),
    ]:
        assert all(not dec._gated_rewrite for dec in model.tdecoder)
        assert all(dec._gated_rewrite for dec in model.decoder) == isinstance(model, HTDemucsMLX)
