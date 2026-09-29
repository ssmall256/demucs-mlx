"""Measure the effect of local GroupNorm when tiling the waveform encoder."""

import mlx.core as mx
import numpy as np

from demucs_mlx.ane import LENGTH
from demucs_mlx.model_converter import get_mlx_model


def encode(model, x):
    values = []
    for layer in model.tencoder:
        x = layer(x)
        values.append(x)
    mx.eval(*values)
    return values


model = get_mlx_model("htdemucs").models[0]
model.eval()
rng = np.random.default_rng(42)
mix = mx.array(rng.standard_normal((2, 2, LENGTH), dtype=np.float32) * 0.1)
x = (mix - mx.mean(mix, axis=(1, 2), keepdims=True)) / (
    1e-5 + mx.std(mix, axis=(1, 2), keepdims=True)
)
reference = encode(model, x)
center, halo = 8192, 4096
padded = mx.pad(x, [(0, 0), (0, 0), (halo, center + halo)])
chunks = [[] for _ in range(4)]
for start in range(0, LENGTH, center):
    tile = padded[:, :, start : start + center + 2 * halo]
    outputs = encode(model, tile)
    for stage, output in enumerate(outputs):
        stride = 4 ** (stage + 1)
        chunks[stage].append(output[..., halo // stride : (halo + center) // stride])

print("## Waveform encoder tile fidelity")
print(f"**Tile:** `{center + 2 * halo}` samples; **center:** `{center}`; **halo:** `{halo}`")
print("| Stage | SNR | Peak error |")
print("|---|---:|---:|")
for stage, want in enumerate(reference):
    got = mx.concatenate(chunks[stage], axis=-1)[..., : want.shape[-1]]
    mx.eval(got)
    want_np = np.asarray(want, dtype=np.float64)
    error = want_np - np.asarray(got, dtype=np.float64)
    snr = 10 * np.log10(np.sum(want_np * want_np) / np.sum(error * error))
    print(f"| {stage + 1} | {snr:.2f} dB | {np.max(np.abs(error)):.6g} |", flush=True)
