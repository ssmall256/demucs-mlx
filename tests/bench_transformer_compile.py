"""Compare eager and compiled cross-transformer inference.

Run through metalq. Captures realistic transformer inputs from one model pass,
then alternates calls with exactly the same evaluated input arrays.
"""

import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


class Capture:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.inputs = None

    def __call__(self, x, xt):
        self.inputs = (x, xt)
        return self.wrapped(x, xt)


rng = np.random.default_rng(481)
data = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
separator = Separator(seed=481)
model = separator._model.models[0]
original = model.crosstransformer
capture = Capture(original)
model.crosstransformer = capture
mx.eval(model(data))
model.crosstransformer = original
assert capture.inputs is not None, "the cross-transformer was not called"
x, xt = capture.inputs
mx.eval(x, xt)

traces = []


def forward(x, xt):
    traces.append((x.shape, xt.shape))
    return original(x, xt)


compiled = mx.compile(forward)
eager = original


def call(fn):
    output = fn(x, xt)
    mx.eval(*output)
    return output


eager_result = call(eager)
compiled_result = call(compiled)
print("## Cross-transformer compile probe", flush=True)
print(f"**Input shapes:** `{x.shape}`, `{xt.shape}`", flush=True)
print(f"**Python traces after first compiled call:** `{len(traces)}`", flush=True)
for index, (want, got) in enumerate(zip(eager_result, compiled_result)):
    error = np.asarray(want, dtype=np.float64) - np.asarray(got, dtype=np.float64)
    peak = np.max(np.abs(error))
    print(f"**Output {index} peak error:** `{peak:.6g}`", flush=True)

for _ in range(2):
    call(eager)
    call(compiled)

times = {"eager": [], "compiled": []}
print("| Pair | Eager | Compiled | Eager / compiled |", flush=True)
print("|---:|---:|---:|---:|", flush=True)
for pair in (1, 2, 3):
    results = {}
    for name, fn in (("eager", eager), ("compiled", compiled)):
        start = time.perf_counter()
        for _ in range(8):
            call(fn)
        elapsed = (time.perf_counter() - start) / 8
        times[name].append(elapsed)
        results[name] = elapsed
    print(
        f"| {pair} | {results['eager'] * 1000:.2f} ms | "
        f"**{results['compiled'] * 1000:.2f} ms** | "
        f"{results['eager'] / results['compiled']:.2f}x |",
        flush=True,
    )
print(f"**Python traces after all calls:** `{len(traces)}`", flush=True)
print(
    f"> Median: {statistics.median(times['eager']) * 1000:.2f} ms eager, "
    f"{statistics.median(times['compiled']) * 1000:.2f} ms compiled.",
    flush=True,
)
