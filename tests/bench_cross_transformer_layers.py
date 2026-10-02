"""Compare eager and per-layer compiled HTDemucs cross-transformer paths.

Run through ``metalq submit -w`` with one model and evaluated real inputs.
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


class CompiledLayer:
    def __init__(self, layer):
        self.function = mx.compile(lambda *inputs: layer(*inputs))

    def __call__(self, *inputs):
        return self.function(*inputs)


model = Separator(seed=481).model.models[0]
transformer = model.crosstransformer
capture = Capture(transformer)
model.crosstransformer = capture
rng = np.random.default_rng(481)
segment = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
mx.eval(model(segment))
model.crosstransformer = transformer
assert capture.inputs is not None, "the cross-transformer was not called"
x, xt = capture.inputs
mx.eval(x, xt)

original = (transformer.layers, transformer.layers_t)
compiled = tuple([CompiledLayer(layer) for layer in layers] for layers in original)


def select(path):
    for source, eager, fast in zip(("layers", "layers_t"), original, compiled):
        selected = [
            fast[index]
            if path == "all"
            or (path == "self" and index % 2 == transformer.classic_parity)
            or (path == "cross" and index % 2 != transformer.classic_parity)
            else layer
            for index, layer in enumerate(eager)
        ]
        setattr(transformer, source, selected)


def run(path):
    select(path)
    start = time.perf_counter()
    output = transformer(x, xt)
    mx.eval(*output)
    return (time.perf_counter() - start) * 1000, output


print("## Per-layer transformer compilation", flush=True)
print(f"**Inputs:** `{x.shape}` frequency, `{xt.shape}` waveform", flush=True)
print("| Path | Warmed median | Eager / path | Output peak error |", flush=True)
print("|---|---:|---:|---:|", flush=True)

try:
    _, reference = run("eager")
    for path in ("self", "cross", "all"):
        cold, output = run(path)
        peak = max(
            np.max(np.abs(np.asarray(want) - np.asarray(got)))
            for want, got in zip(reference, output)
        )
        print(f"> Cold {path} first call: **{cold:.2f} ms**, peak error `{peak:.3g}`", flush=True)
    for path in ("eager", "self", "cross", "all"):
        run(path)
    times = {path: [] for path in ("eager", "self", "cross", "all")}
    for pair in range(6):
        order = tuple(times) if pair % 2 == 0 else tuple(reversed(tuple(times)))
        for path in order:
            duration, _ = run(path)
            times[path].append(duration)
    eager = statistics.median(times["eager"])
    for path in times:
        value = statistics.median(times[path])
        _, output = run(path)
        peak = max(
            np.max(np.abs(np.asarray(want) - np.asarray(got)))
            for want, got in zip(reference, output)
        )
        print(f"| {path} | {value:.2f} ms | {eager / value:.2f}x | {peak:.3g} |", flush=True)
finally:
    transformer.layers, transformer.layers_t = original
