"""Compare the 21- and seven-tile Core ML convolution assets.

Run through metalq after converting both variants. The inputs and worker are
reused, so the timing isolates Core ML execution from loading and conversion.
"""

import time

import mlx.core as mx
import numpy as np

from demucs_mlx.ane import LENGTH, WaveformConv
from demucs_mlx.model_converter import get_mlx_model


def quality(reference, candidate):
    want = reference.astype(np.float64)
    error = want - candidate.astype(np.float64)
    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
    return snr, np.max(np.abs(error))


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

workers = {"21 tiles": WaveformConv(4095), "7 tiles": WaveformConv(12285)}
try:
    outputs = {}
    print("## ANE tile fidelity", flush=True)
    print("| Variant | SNR vs MLX | Peak error | Batch-one parity |", flush=True)
    print("|---|---:|---:|---|", flush=True)
    for name, worker in workers.items():
        output = worker.submit(mix).result()
        one = worker.submit(mix[:1]).result()
        np.testing.assert_array_equal(one, output[:1])
        outputs[name] = output
        snr, peak = quality(reference, output)
        print(f"| {name} | {snr:.2f} dB | {peak:.6g} | exact |", flush=True)
    snr, peak = quality(outputs["21 tiles"], outputs["7 tiles"])
    print(f"> Seven versus 21 tiles: **{snr:.2f} dB**, peak `{peak:.6g}`.", flush=True)

    for worker in workers.values():
        for _ in range(3):
            worker.submit(mix).result()

    print("## Alternating Core ML prediction timing", flush=True)
    print("| Pair | Variant | Predictions | Mean wall | Mean execution |", flush=True)
    print("|---:|---|---:|---:|---:|", flush=True)
    for pair in (1, 2, 3):
        for name, worker in workers.items():
            before = worker.busy_seconds
            start = time.perf_counter()
            for _ in range(12):
                worker.submit(mix).result()
            wall = time.perf_counter() - start
            execution = worker.busy_seconds - before
            print(
                f"| {pair} | {name} | 12 | **{wall / 12 * 1000:.2f} ms** | "
                f"{execution / 12 * 1000:.2f} ms |",
                flush=True,
            )
finally:
    for worker in workers.values():
        worker.close()
