"""Compare the ANE convolution with the default MLX convolution.

Run through metalq as shown in docs/ane-prototype.md.
"""

import mlx.core as mx
import numpy as np

from demucs_mlx.ane import LENGTH, WaveformConv
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
    reference = model.tencoder[0].conv(x)
    mx.eval(reference)
    reference = np.asarray(reference)

    backend = WaveformConv()
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

    print("## Core ML waveform convolution parity")
    print(f"**Compute plan:** `{backend.placement}`")
    if converted.shape != reference.shape or final_one.shape != reference[:1].shape:
        raise AssertionError("Unexpected convolution output shape")
    want = reference.astype(np.float64)
    got = converted.astype(np.float64)
    error = want - got
    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
    print(f"**SNR:** {snr:.2f} dB; **peak error:** {np.max(np.abs(error)):.6g}")
    np.testing.assert_allclose(final_one, converted[:1], rtol=0, atol=0)
    print("> Batch-one padding preserves output values.")


if __name__ == "__main__":
    main()
