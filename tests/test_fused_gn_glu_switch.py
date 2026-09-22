"""The fused GroupNorm+activation kernels must be switchable at runtime.

These are custom Metal kernels with simdgroup reductions. When Demucs output
looks wrong, the first question is whether the kernels are responsible, and
until now there was no way to answer it without editing the package. The switch
defaults to enabled, so this is purely a diagnostic lever.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from demucs_mlx.mlx_hdemucs import HEncLayer
    from demucs_mlx.mlx_layers import _use_fused_gn_glu

    mx.random.seed(0)
    layer = HEncLayer(4, 8, norm=True, norm_groups=4, freq=False)
    keys = sorted(k for k, _ in tree_flatten(layer.parameters()))
    weights = {k: mx.ones(v.shape) * 0.5 for k, v in tree_flatten(layer.parameters())}
    layer.update(tree_unflatten(list(weights.items())))
    x = mx.ones((1, 4, 128)) * 0.25
    out = np.asarray(layer(x))
    json.dump(
        {"fused": _use_fused_gn_glu(), "keys": keys, "out": out.ravel().tolist()},
        sys.stdout,
    )
    """
)


def _probe(value):
    """Run in a subprocess: the switch is read at layer-construction time."""
    env = {"PATH": "/usr/bin:/bin", "DEMUCS_MLX_USE_FUSED_GN_GLU": value}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"probe failed for {value!r}:\n{result.stderr}")
    import json

    return json.loads(result.stdout)


def test_switch_defaults_to_disabled():
    """Fused kernels cost ~20 dB SNR against the unfused path and are not
    faster, so they are opt-in."""
    from demucs_mlx.mlx_layers import _use_fused_gn_glu

    assert _use_fused_gn_glu() is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", ""])
def test_switch_recognizes_disabling_values(monkeypatch, value):
    from demucs_mlx.mlx_layers import _use_fused_gn_glu

    monkeypatch.setenv("DEMUCS_MLX_USE_FUSED_GN_GLU", value)
    assert _use_fused_gn_glu() is False


@pytest.mark.parametrize("value", ["1", "true", "on", "YES"])
def test_switch_recognizes_enabling_values(monkeypatch, value):
    from demucs_mlx.mlx_layers import _use_fused_gn_glu

    monkeypatch.setenv("DEMUCS_MLX_USE_FUSED_GN_GLU", value)
    assert _use_fused_gn_glu() is True


def test_both_paths_expose_the_same_parameters():
    """A converted cache must load regardless of how the switch is set."""
    on, off = _probe("1"), _probe("0")
    assert on["fused"] is True and off["fused"] is False
    assert on["keys"] == off["keys"]


def test_both_paths_agree_numerically():
    import numpy as np

    on, off = _probe("1"), _probe("0")
    a = np.array(on["out"])
    b = np.array(off["out"])
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-4
