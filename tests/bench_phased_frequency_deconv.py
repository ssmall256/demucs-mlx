"""Compare spectral decoder transposed convolution with four-phase Conv2d.

Run through metalq. For kernel height 8 and stride 4, each output phase is a
two-tap ordinary convolution over the input frequency axis.
"""

import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_layers import ConvTranspose2dNCHW


def phased(layer, x, weight):
    conv = layer.conv
    batch, channels, frequency, frames = x.shape
    nhwc = x.transpose(0, 2, 3, 1)
    padded = mx.pad(nhwc, [(0, 0), (1, 1), (0, 0), (0, 0)])
    phases = mx.conv2d(padded, weight)
    out_channels = conv.weight.shape[0]
    phases = phases.reshape(batch, frequency + 1, frames, 4, out_channels)
    joined = phases.transpose(0, 1, 3, 2, 4).reshape(
        batch, 4 * (frequency + 1), frames, out_channels
    )
    if "bias" in conv:
        joined = joined + conv.bias
    return joined.transpose(0, 3, 1, 2)


def phased_weight(conv):
    original = conv.weight
    phases = [
        mx.stack([original[:, phase + 4, 0, :], original[:, phase, 0, :]], axis=1)
        for phase in range(4)
    ]
    return mx.concatenate(phases, axis=0).reshape(-1, 2, 1, original.shape[-1])


def measure(fn, count=15):
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


captured = []
integrated = ConvTranspose2dNCHW.__call__


def original(layer, x):
    return layer.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


def capture(self, x):
    captured.append((self, x))
    return original(self, x)


ConvTranspose2dNCHW.__call__ = capture
try:
    rng = np.random.default_rng(481)
    data = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
    model = Separator(seed=481)._model.models[0]
    mx.eval(model(data))
finally:
    ConvTranspose2dNCHW.__call__ = integrated

print("## Phased spectral transposed convolution", flush=True)
print("| Input | Original | Phased | Original / phased | SNR | Peak error |", flush=True)
print("|---|---:|---:|---:|---:|---:|", flush=True)
for layer, x in captured:
    conv = layer.conv
    if conv.weight.shape[1:3] != (8, 1) or tuple(conv.stride) != (4, 1):
        continue
    weight = phased_weight(conv)
    mx.eval(weight)
    for _ in range(3):
        mx.eval(original(layer, x), phased(layer, x, weight))
    want = np.asarray(original(layer, x), dtype=np.float64)
    got = np.asarray(phased(layer, x, weight), dtype=np.float64)
    assert want.shape == got.shape
    error = want - got
    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
    standard_ms = measure(lambda: original(layer, x))
    phased_ms = measure(lambda: phased(layer, x, weight))
    print(
        f"| `{x.shape}` | {standard_ms:.3f} ms | **{phased_ms:.3f} ms** | "
        f"{standard_ms / phased_ms:.2f}x | {snr:.2f} dB | "
        f"{np.max(np.abs(error)):.6g} |",
        flush=True,
    )
