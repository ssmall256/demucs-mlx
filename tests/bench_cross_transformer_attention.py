"""Measure HTDemucs attention projections and MLX fused-kernel selection.

Run through ``metalq submit -w``. Inputs are captured from a real segment.
"""

import math
import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


class Capture:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.inputs = None

    def __call__(self, *inputs, **kwargs):
        self.inputs = inputs
        return self.wrapped(*inputs, **kwargs)


def timed(function, repeats=7):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        mx.eval(*(value if isinstance(value, tuple) else (value,)))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


model = Separator(seed=481).model.models[0]
transformer = model.crosstransformer
cases = {}
for name, layer in (
    ("frequency self", transformer.layers[0]),
    ("waveform self", transformer.layers_t[0]),
    ("frequency cross", transformer.layers[1]),
    ("waveform cross", transformer.layers_t[1]),
):
    member = "attn" if "self" in name else "cross_attn"
    original = getattr(layer, member)
    capture = Capture(original)
    setattr(layer, member, capture)
    cases[name] = (layer, member, original, capture)

rng = np.random.default_rng(481)
segment = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
mx.eval(model(segment))

print("## Cross-transformer attention at real segment shapes", flush=True)
print(
    "| Attention | Queries | Keys | QKV projections | SDPA default | "
    "SDPA forced | Contiguous QKV + SDPA | FP16 QKV + SDPA | FP16 SNR | Peak error |",
    flush=True,
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
for name, (layer, member, attention, capture) in cases.items():
    setattr(layer, member, attention)
    queries, keys, values = capture.inputs[:3]
    mx.eval(queries, keys, values)
    heads = attention.num_heads

    def project():
        q = attention.query_proj(queries)
        k = attention.key_proj(keys)
        v = attention.value_proj(values)
        return q, k, v

    projection_ms = timed(project)
    q, k, v = project()
    mx.eval(q, k, v)
    q = mx.unflatten(q, -1, (heads, -1)).transpose(0, 2, 1, 3)
    k = mx.unflatten(k, -1, (heads, -1)).transpose(0, 2, 1, 3)
    v = mx.unflatten(v, -1, (heads, -1)).transpose(0, 2, 1, 3)
    mx.eval(q, k, v)
    scale = math.sqrt(1 / q.shape[-1])

    def sdpa(force):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, force_fused=force)

    def contiguous_sdpa():
        return mx.fast.scaled_dot_product_attention(
            mx.contiguous(q), mx.contiguous(k), mx.contiguous(v), scale=scale
        )

    def half_sdpa():
        return mx.fast.scaled_dot_product_attention(
            q.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            scale=scale,
        ).astype(mx.float32)

    try:
        default = sdpa(False)
        forced = sdpa(True)
        contiguous = contiguous_sdpa()
        half = half_sdpa()
        mx.eval(default, forced, contiguous, half)
        np.testing.assert_array_equal(np.asarray(default), np.asarray(forced))
        np.testing.assert_array_equal(np.asarray(default), np.asarray(contiguous))
        timed(lambda: sdpa(False), repeats=2)
        timed(lambda: sdpa(True), repeats=2)
        times = {False: [], True: []}
        for pair in range(6):
            for force in (False, True) if pair % 2 == 0 else (True, False):
                times[force].append(timed(lambda force=force: sdpa(force), repeats=1))
        default_ms = statistics.median(times[False])
        forced_ms = statistics.median(times[True])
        forced_text = f"{forced_ms:.3f} ms"
    except RuntimeError as exc:
        default_ms = timed(lambda: sdpa(False))
        forced_text = f"unsupported: {type(exc).__name__}"

    timed(contiguous_sdpa, repeats=2)
    contiguous_ms = timed(contiguous_sdpa)
    timed(half_sdpa, repeats=2)
    half_ms = timed(half_sdpa)
    half_error = np.asarray(default, dtype=np.float64) - np.asarray(half, dtype=np.float64)
    half_snr = 10 * np.log10(
        np.sum(np.asarray(default, dtype=np.float64) ** 2)
        / max(np.sum(half_error * half_error), 1e-30)
    )
    half_peak = np.max(np.abs(half_error))

    print(
        f"| {name} | {q.shape[-2]} | {k.shape[-2]} | {projection_ms:.3f} ms | "
        f"{default_ms:.3f} ms | {forced_text} | {contiguous_ms:.3f} ms | "
        f"{half_ms:.3f} ms | {half_snr:.1f} dB | {half_peak:.3g} |",
        flush=True,
    )
