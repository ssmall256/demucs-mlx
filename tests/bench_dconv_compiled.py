"""Test compiling the expensive DConv block at real HTDemucs shapes.

Run through ``metalq submit -w`` to serialize Metal measurements.
"""

import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def run(function, value):
    start = time.perf_counter()
    result = function(value)
    mx.eval(result)
    return (time.perf_counter() - start) * 1000, result


model = Separator(seed=481).model.models[0]
cases = (
    ("frequency", model.encoder[0].dconv.layers[0], (1024, 48, 336)),
    ("waveform", model.tencoder[0].dconv.layers[0], (2, 48, 85_995)),
)

print("## DConv block compilation at HTDemucs shapes", flush=True)
print(
    "| Block | Eager | Compiled | Eager / compiled | SNR | Peak error | Cold compile |", flush=True
)
print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
for name, block, shape in cases:
    mx.random.seed(481)
    value = mx.random.normal(shape) * 0.1
    mx.eval(value)
    compiled = mx.compile(lambda x: block(x))
    cold, _ = run(compiled, value)
    for _ in range(2):
        run(block, value)
        run(compiled, value)
    elapsed = {"eager": [], "compiled": []}
    outputs = {}
    for pair in range(6):
        order = ("eager", "compiled") if pair % 2 == 0 else ("compiled", "eager")
        for path in order:
            function = block if path == "eager" else compiled
            duration, result = run(function, value)
            elapsed[path].append(duration)
            outputs[path] = np.asarray(result)
    eager = float(np.median(elapsed["eager"]))
    fast = float(np.median(elapsed["compiled"]))
    reference = outputs["eager"].astype(np.float64)
    difference = reference - outputs["compiled"].astype(np.float64)
    mse = np.sum(difference * difference)
    snr = float("inf") if mse == 0 else 10 * np.log10(np.sum(reference * reference) / mse)
    print(
        f"| {name} | {eager:.3f} ms | {fast:.3f} ms | {eager / fast:.2f}x | "
        f"{snr:.1f} dB | {np.max(np.abs(difference)):.3g} | {cold:.1f} ms |",
        flush=True,
    )
