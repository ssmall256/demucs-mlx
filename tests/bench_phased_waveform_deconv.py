"""Compare waveform transposed convolutions with four-phase Conv1d.

Run through metalq. A stride-four, eight-tap transposed convolution is four
interleaved two-tap ordinary convolutions over zero-padded input.
"""

import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_layers import ConvTranspose1dNCL


def phased_weight(conv):
    source = conv.weight
    phases = [
        mx.stack([source[:, phase + 4, :], source[:, phase, :]], axis=1)
        for phase in range(4)
    ]
    return mx.concatenate(phases, axis=0)


def phased(layer, x, weight):
    conv = layer.conv
    batch, _, length = x.shape
    nlc = x.transpose(0, 2, 1)
    padded = mx.pad(nlc, [(0, 0), (1, 1), (0, 0)])
    phases = mx.conv1d(padded, weight)
    out_channels = conv.weight.shape[0]
    joined = phases.reshape(batch, length + 1, 4, out_channels)
    joined = joined.reshape(batch, 4 * (length + 1), out_channels)
    if "bias" in conv:
        joined = joined + conv.bias
    return joined.transpose(0, 2, 1)


def measure(fn, count=15):
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


captured = []
integrated = ConvTranspose1dNCL.__call__


def original(layer, x):
    return layer.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def capture(self, x):
    captured.append((self, x))
    return original(self, x)


ConvTranspose1dNCL.__call__ = capture
try:
    rng = np.random.default_rng(481)
    data = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
    model = Separator(seed=481)._model.models[0]
    mx.eval(model(data))
finally:
    ConvTranspose1dNCL.__call__ = integrated

print("## Phased waveform transposed convolution", flush=True)
print("| Input | Original | Phased | Original / phased | SNR | Peak error |", flush=True)
print("|---|---:|---:|---:|---:|---:|", flush=True)
for layer, x in captured:
    conv = layer.conv
    if conv.weight.shape[1] != 8 or conv.stride != 4 or conv.padding != 0:
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
