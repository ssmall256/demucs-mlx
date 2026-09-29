"""Test FP16 attention scores inside an otherwise FP32 HTDemucs model.

Run through ``metalq submit -w``. This is a diagnostic, not a runtime option.
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


original = nn.MultiHeadAttention.__call__


def half_attention(self, queries, keys, values, mask=None):
    queries = self.query_proj(queries)
    keys = self.key_proj(keys)
    values = self.value_proj(values)
    heads = self.num_heads
    queries = mx.unflatten(queries, -1, (heads, -1)).transpose(0, 2, 1, 3)
    keys = mx.unflatten(keys, -1, (heads, -1)).transpose(0, 2, 1, 3)
    values = mx.unflatten(values, -1, (heads, -1)).transpose(0, 2, 1, 3)
    output = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float16),
        keys.astype(mx.float16),
        values.astype(mx.float16),
        scale=math.sqrt(1 / queries.shape[-1]),
        mask=mask,
    )
    output = output.astype(queries.dtype).transpose(0, 2, 1, 3).flatten(-2, -1)
    return self.out_proj(output)


def run(separator, path, audio):
    nn.MultiHeadAttention.__call__ = original if path == "fp32" else half_attention
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    return time.perf_counter() - start, stems


separator = Separator(seed=481)
print("## FP16 scores in otherwise FP32 HTDemucs attention", flush=True)
print("**Settings:** default shifts, overlap, split and batch size; seed 481", flush=True)
print(
    "| Input | Pair | FP32 | FP16 attention | FP32 / FP16 | Minimum stem SNR | Peak error |",
    flush=True,
)
print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)

try:
    audio = signal(30)
    run(separator, "fp32", audio)
    run(separator, "fp16", audio)
    for seconds in (30, 60):
        audio = signal(seconds)
        for pair, order in ((1, ("fp32", "fp16")), (2, ("fp16", "fp32"))):
            measured = {path: run(separator, path, audio) for path in order}
            reference_time, reference = measured["fp32"]
            candidate_time, candidate = measured["fp16"]
            snrs = []
            peaks = []
            for stem, want in reference.items():
                expected = want.astype(np.float64)
                error = expected - candidate[stem].astype(np.float64)
                snrs.append(
                    10 * np.log10(np.sum(expected * expected) / max(np.sum(error * error), 1e-30))
                )
                peaks.append(np.max(np.abs(error)))
            print(
                f"| {seconds}s | {pair} | {reference_time:.3f}s | "
                f"**{candidate_time:.3f}s** | {reference_time / candidate_time:.2f}x | "
                f"{min(snrs):.2f} dB | {max(peaks):.3g} |",
                flush=True,
            )
finally:
    nn.MultiHeadAttention.__call__ = original
