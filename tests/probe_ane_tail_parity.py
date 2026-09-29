"""Compare the full-length ANE waveform tail against MLX.

Run with metalq submit -w after `python -m demucs_mlx.ane convert-tail`.
"""

import mlx.core as mx
import numpy as np
import torch

from demucs_mlx.ane import LENGTH, WaveformTail
from demucs_mlx.model_converter import get_mlx_model
from demucs_mlx.secure_demucs import get_restricted_demucs_model

rng = np.random.default_rng(481)
mix = mx.array(rng.standard_normal((2, 2, LENGTH), dtype=np.float32) * 0.1)
model = get_mlx_model("htdemucs").models[0]
model.eval()
x = (mix - mx.mean(mix, axis=(1, 2), keepdims=True)) / (
    1e-5 + mx.std(mix, axis=(1, 2), keepdims=True)
)
stage0 = model.tencoder[0](x)
reference = []
for layer in model.tencoder[1:]:
    stage0 = layer(stage0)
    reference.append(stage0)
mx.eval(*reference)
initial = model.tencoder[0](x)
mx.eval(initial)
data = np.asarray(initial)
torch_model = get_restricted_demucs_model("htdemucs").model.models[0].eval()
torch_reference = []
with torch.no_grad():
    torch_x = torch.from_numpy(data.astype(np.float32))
    for layer in torch_model.tencoder[1:]:
        torch_x = layer(torch_x)
        torch_reference.append(torch_x.numpy())
    torch_model.tencoder[1:].half()
    half_reference = []
    half_x = torch.from_numpy(data.astype(np.float16))
    for layer in torch_model.tencoder[1:]:
        half_x = layer(half_x)
        half_reference.append(half_x.float().numpy())

worker = WaveformTail()
try:
    output = worker.submit(data).result()
    tail_one = worker.submit(data[:1]).result()
finally:
    worker.close()
assert not worker._worker.is_alive()
assert len(output) == len(reference) == 3
print("## Full-length ANE waveform tail parity")
print(f"**Placement:** `{worker.placement}`")
print(
    "| Stage | Shape | Torch vs MLX | Torch FP16 vs FP32 | ANE vs Torch | "
    "ANE vs MLX | Peak error | Batch-one parity |"
)
print("|---:|---|---:|---:|---:|---:|---:|---|")
for stage, (want_array, torch_array, half_array, got_array, one_array) in enumerate(
    zip(reference, torch_reference, half_reference, output, tail_one), 2
):
    want = np.asarray(want_array, dtype=np.float64)
    torch_want = torch_array.astype(np.float64)
    half_want = half_array.astype(np.float64)
    got = got_array.astype(np.float64)
    assert want.shape == got.shape
    np.testing.assert_array_equal(one_array, got_array[:1])
    def snr(reference, candidate):
        error = reference - candidate
        return 10 * np.log10(np.sum(reference * reference) / max(np.sum(error * error), 1e-30))

    print(
        f"| {stage} | `{got.shape}` | {snr(want, torch_want):.2f} dB | "
        f"{snr(torch_want, half_want):.2f} dB | {snr(torch_want, got):.2f} dB | "
        f"{snr(want, got):.2f} dB | "
        f"{np.max(np.abs(want - got)):.6g} | exact |"
    )
