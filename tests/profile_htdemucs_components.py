"""Coarse synchronized profile of one batch-two HTDemucs segment.

Run through metalq. Forced per-component evaluation changes scheduling, so
these are bottleneck hints rather than additive production timings.
"""

import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def evaluate(value):
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            evaluate(item)


class Timed:
    def __init__(self, name, wrapped, totals):
        self.name = name
        self.wrapped = wrapped
        self.totals = totals

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        output = self.wrapped(*args, **kwargs)
        evaluate(output)
        self.totals[self.name] += time.perf_counter() - start
        return output


rng = np.random.default_rng(481)
data = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
mx.eval(data)
separator = Separator(seed=481)
model = separator._model.models[0]

for _ in range(2):
    mx.eval(model(data))
baseline = []
for _ in range(3):
    start = time.perf_counter()
    mx.eval(model(data))
    baseline.append(time.perf_counter() - start)

totals = defaultdict(float)
for name in ("_spec", "_magnitude", "_mask", "_ispec"):
    setattr(model, name, Timed(name, getattr(model, name), totals))
for group in ("encoder", "tencoder", "decoder", "tdecoder"):
    layers = getattr(model, group)
    for index, layer in enumerate(layers):
        layers[index] = Timed(f"{group}.{index}", layer, totals)
if model.crosstransformer:
    model.crosstransformer = Timed("crosstransformer", model.crosstransformer, totals)

for _ in range(2):
    mx.eval(model(data))
totals.clear()
instrumented = []
for _ in range(3):
    start = time.perf_counter()
    mx.eval(model(data))
    instrumented.append(time.perf_counter() - start)

print("## Synchronized HTDemucs component profile", flush=True)
print("**Input:** batch 2, 343,980 samples per channel", flush=True)
print(f"**Uninstrumented median:** {np.median(baseline) * 1000:.2f} ms", flush=True)
print(f"**Instrumented median:** {np.median(instrumented) * 1000:.2f} ms", flush=True)
print("| Component | Mean per call |", flush=True)
print("|---|---:|", flush=True)
for name, total in sorted(totals.items(), key=lambda item: -item[1]):
    print(f"| `{name}` | {total / 3 * 1000:.2f} ms |", flush=True)
