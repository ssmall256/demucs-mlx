"""Compare the compiled Core ML encoder with the default MLX encoder.

Run through metalq as shown in docs/ane-prototype.md.
"""

import mlx.core as mx
import numpy as np

from demucs_mlx.ane import LENGTH, WaveformEncoder
from demucs_mlx.model_converter import get_mlx_model


def main():
    rng = np.random.default_rng(481)
    mix = (rng.standard_normal((2, 2, LENGTH), dtype=np.float32) * 0.1).astype(np.float32)
    model = get_mlx_model("htdemucs").models[0]
    model.eval()
    x = mx.array(mix)
    x = (x - mx.mean(x, axis=(1, 2), keepdims=True)) / (
        1e-5 + mx.std(x, axis=(1, 2), keepdims=True)
    )
    reference = []
    for layer in model.tencoder:
        x = layer(x)
        reference.append(x)
    mx.eval(*reference)
    reference = [np.asarray(value) for value in reference]

    backend = WaveformEncoder()
    try:
        try:
            backend.submit(np.empty((2, 2, 1), dtype=np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid waveform length to be rejected")
        converted = backend.submit(mix).result()
        final_one = backend.submit(mix[:1]).result()
    finally:
        backend.close()
    if backend._worker.is_alive():
        raise AssertionError("Core ML worker remained alive after close")

    print("## Core ML waveform encoder parity")
    print(f"**Compute plan:** `{backend.placement}`")
    print("| Stage | SNR | Peak error | Mean error |")
    print("|---|---:|---:|---:|")
    for index, (want, got, one) in enumerate(zip(reference, converted, final_one)):
        if got.shape != want.shape or one.shape != want[:1].shape:
            raise AssertionError(f"Unexpected output shape at stage {index}")
        want = want.astype(np.float64)
        got = got.astype(np.float64)
        error = want - got
        snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
        print(
            f"| {index} | **{snr:.2f} dB** | {np.max(np.abs(error)):.6g} "
            f"| {np.mean(np.abs(error)):.6g} |"
        )
        np.testing.assert_allclose(one, converted[index][:1], rtol=0, atol=0)
    print("> Batch-one padding preserves output values.")


if __name__ == "__main__":
    main()
