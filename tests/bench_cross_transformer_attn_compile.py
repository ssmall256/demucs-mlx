"""Compare eager and individually compiled transformer attention modules.

Run through ``metalq submit -w`` with fixed, evaluated real inputs.
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


class CompiledAttention:
    def __init__(self, attention):
        self.function = mx.compile(lambda q, k, v, mask=None: attention(q, k, v, mask=mask))

    def __call__(self, q, k, v, mask=None):
        return self.function(q, k, v, mask)


model = Separator(seed=481).model.models[0]
transformer = model.crosstransformer
capture = Capture(transformer)
model.crosstransformer = capture
rng = np.random.default_rng(481)
segment = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
mx.eval(model(segment))
model.crosstransformer = transformer
x, xt = capture.inputs
mx.eval(x, xt)

attention = []
for branch in (transformer.layers, transformer.layers_t):
    for layer in branch:
        member = "attn" if hasattr(layer, "attn") else "cross_attn"
        original = getattr(layer, member)
        attention.append((layer, member, original, CompiledAttention(original)))


def select(compiled):
    for layer, member, original, fast in attention:
        setattr(layer, member, fast if compiled else original)


def run(compiled):
    select(compiled)
    start = time.perf_counter()
    output = transformer(x, xt)
    mx.eval(*output)
    return (time.perf_counter() - start) * 1000, output


print("## Individual transformer attention compilation", flush=True)
print(f"**Inputs:** `{x.shape}` frequency, `{xt.shape}` waveform", flush=True)
try:
    _, reference = run(False)
    cold, compiled_output = run(True)
    peak = max(
        np.max(np.abs(np.asarray(want) - np.asarray(got)))
        for want, got in zip(reference, compiled_output)
    )
    print(f"**Cold compiled call:** {cold:.2f} ms", flush=True)
    print(f"**Output peak error:** {peak:.3g}", flush=True)
    for compiled in (False, True):
        run(compiled)
    times = {False: [], True: []}
    for pair in range(8):
        for compiled in (False, True) if pair % 2 == 0 else (True, False):
            duration, _ = run(compiled)
            times[compiled].append(duration)
    eager = statistics.median(times[False])
    fast = statistics.median(times[True])
    print("| Eager | Compiled attention | Eager / compiled |", flush=True)
    print("|---:|---:|---:|", flush=True)
    print(f"| {eager:.2f} ms | {fast:.2f} ms | {eager / fast:.2f}x |", flush=True)
finally:
    select(False)
