"""Measure all adopted GPU optimizations against the original path.

Run via ``metalq submit -w``. Both paths use the same loaded default model,
input, seed, and inference settings. Model load and file I/O are excluded.
"""

import os
import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_demucs import GroupNorm
from demucs_mlx.mlx_layers import (
    ConvTranspose1dNCL,
    ConvTranspose2dNCHW,
    GroupNormNCHW,
    GroupNormNCL,
)


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


def previous_group_norm(self, x):
    batch, channels = x.shape[:2]
    grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, *x.shape[2:])
    axes = tuple(range(2, grouped.ndim))
    mean = mx.mean(grouped, axis=axes, keepdims=True)
    variance = mx.var(grouped, axis=axes, keepdims=True)
    normalized = ((grouped - mean) * mx.rsqrt(variance + self.eps)).reshape(x.shape)
    if not self.affine:
        return normalized
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * self.weight.reshape(affine_shape) + self.bias.reshape(affine_shape)


def previous_1d(self, x):
    return self.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def previous_2d(self, x):
    return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


group_norm_classes = (GroupNorm, GroupNormNCL, GroupNormNCHW)
group_norm_calls = {cls: cls.__call__ for cls in group_norm_classes}
current_1d = ConvTranspose1dNCL.__call__
current_2d = ConvTranspose2dNCHW.__call__
prior_compile_setting = os.environ.get("DEMUCS_MLX_COMPILE_DCONV")


def select(path):
    final = path == "after"
    for cls in group_norm_classes:
        cls.__call__ = group_norm_calls[cls] if final else previous_group_norm
    ConvTranspose1dNCL.__call__ = current_1d if final else previous_1d
    ConvTranspose2dNCHW.__call__ = current_2d if final else previous_2d
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "1" if final else "0"


def run(separator, path, audio):
    select(path)
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    return time.perf_counter() - start, stems


def fidelity(before, after):
    snrs = []
    peaks = []
    for stem, reference in before.items():
        want = reference.astype(np.float64)
        error = want - after[stem].astype(np.float64)
        energy = np.sum(want * want)
        noise = np.sum(error * error)
        snrs.append(float("inf") if noise == 0 else 10 * np.log10(energy / noise))
        peaks.append(float(np.max(np.abs(error))))
    return min(snrs), max(peaks)


separator = Separator(seed=481)
print("## Combined HTDemucs GPU improvement", flush=True)
print(
    "**Settings:** default GPU path, one shift, 25% overlap, batch two, "
    "seed 481; loaded model and no file I/O.",
    flush=True,
)
print("**Before:** previous GroupNorm and transposed convolutions; eager DConv.", flush=True)
print("**After:** fast GroupNorm, phased decoders, compiled DConv.", flush=True)
print(
    "| Input | Pair | Before | After | Wall reduction | Audio throughput gain | "
    "Minimum stem SNR | Peak error |",
    flush=True,
)
print("|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
try:
    for seconds in (30, 60):
        audio = signal(seconds)
        run(separator, "before", audio)
        run(separator, "after", audio)
        pairs = []
        for index, order in enumerate(
            (("before", "after"), ("after", "before"), ("before", "after")),
            start=1,
        ):
            times = {}
            results = {}
            for path in order:
                times[path], results[path] = run(separator, path, audio)
            before = times["before"]
            after = times["after"]
            snr, peak = fidelity(results["before"], results["after"])
            pairs.append((before, after, snr, peak))
            print(
                f"| {seconds}s | {index} | {before:.3f}s | **{after:.3f}s** | "
                f"{100 * (before - after) / before:.1f}% | "
                f"{100 * (before / after - 1):.1f}% | {snr:.2f} dB | {peak:.3g} |",
                flush=True,
            )
        before_mean = statistics.mean(pair[0] for pair in pairs)
        after_mean = statistics.mean(pair[1] for pair in pairs)
        print(
            f"> **{seconds}s mean:** {before_mean:.3f}s → {after_mean:.3f}s; "
            f"**{100 * (before_mean - after_mean) / before_mean:.1f}% less wall time**, "
            f"**{100 * (before_mean / after_mean - 1):.1f}% more audio per second** "
            f"({before_mean / after_mean:.2f}×).",
            flush=True,
        )
finally:
    select("after")
    if prior_compile_setting is None:
        del os.environ["DEMUCS_MLX_COMPILE_DCONV"]
    else:
        os.environ["DEMUCS_MLX_COMPILE_DCONV"] = prior_compile_setting
