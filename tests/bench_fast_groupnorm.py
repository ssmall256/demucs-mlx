"""Compare current GroupNorm with grouped fast.layer_norm.

Run through metalq. GroupNorm is LayerNorm over flattened channels-per-group
and spatial dimensions, followed by the same per-channel affine transform.
"""

import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.mlx_layers import GroupNormNCHW, GroupNormNCL


def previous_group_norm(x, layer):
    batch, channels = x.shape[:2]
    grouped = x.reshape(batch, layer.num_groups, channels // layer.num_groups, *x.shape[2:])
    axes = tuple(range(2, grouped.ndim))
    mean = mx.mean(grouped, axis=axes, keepdims=True)
    variance = mx.var(grouped, axis=axes, keepdims=True)
    normalized = ((grouped - mean) * mx.rsqrt(variance + layer.eps)).reshape(x.shape)
    if not layer.affine:
        return normalized
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * layer.weight.reshape(affine_shape) + layer.bias.reshape(affine_shape)


def measure(fn, x, count=20):
    times = []
    for _ in range(count):
        start = time.perf_counter()
        mx.eval(fn(x))
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


print("## Previous GroupNorm versus grouped fast.layer_norm", flush=True)
print("| Shape | Groups | Previous | Fast | Previous / fast | SNR | Peak error |", flush=True)
print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
for shape, groups in (
    ((2, 6, 85_995), 1),
    ((2, 12, 21_499), 1),
    ((2, 48, 85_995), 4),
    ((1024, 6, 336), 1),
    ((2, 48, 512, 336), 4),
):
    layer = GroupNormNCL(groups, shape[1]) if len(shape) == 3 else GroupNormNCHW(groups, shape[1])
    x = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(x)
    previous = previous_group_norm(x, layer)
    fast = layer(x)
    mx.eval(previous, fast)
    want = np.asarray(previous, dtype=np.float64)
    error = want - np.asarray(fast, dtype=np.float64)
    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
    for _ in range(3):
        mx.eval(previous_group_norm(x, layer), layer(x))
    previous_ms = measure(lambda value: previous_group_norm(value, layer), x)
    fast_ms = measure(layer, x)
    print(
        f"| `{shape}` | {groups} | {previous_ms:.3f} ms | **{fast_ms:.3f} ms** | "
        f"{previous_ms / fast_ms:.2f}x | {snr:.2f} dB | {np.max(np.abs(error)):.6g} |",
        flush=True,
    )
