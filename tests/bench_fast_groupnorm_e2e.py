"""Alternate previous and fast GroupNorm in complete separation.

Run through metalq. The same model, inputs, and seed are reused; only the
three GroupNorm implementations are switched between measurements.
"""

import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_demucs import GroupNorm
from demucs_mlx.mlx_layers import GroupNormNCHW, GroupNormNCL


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


classes = (GroupNorm, GroupNormNCL, GroupNormNCHW)
originals = {cls: cls.__call__ for cls in classes}


def select_fast(enabled):
    for cls in classes:
        cls.__call__ = originals[cls] if enabled else previous_group_norm


separator = Separator(seed=481)
print("## Fast GroupNorm end-to-end probe", flush=True)
print("**Settings:** default inference, seed=481", flush=True)
print(
    "| Input | Pair | Path | Wall | Previous / fast | Minimum stem SNR | Peak error |",
    flush=True,
)
print("|---:|---:|---|---:|---:|---:|---:|", flush=True)
try:
    for seconds in (30, 60):
        audio = signal(seconds)
        for pair, order in (
            (1, ("previous", "fast")),
            (2, ("previous", "fast")),
            (3, ("fast", "previous")),
        ):
            results = {}
            times = {}
            for name in order:
                enabled = name == "fast"
                select_fast(enabled)
                start = time.perf_counter()
                _, stems = separator.separate_tensor(audio)
                elapsed = time.perf_counter() - start
                results[name] = stems
                times[name] = elapsed
            snrs = []
            peaks = []
            for stem, want_array in results["previous"].items():
                want = want_array.astype(np.float64)
                error = want - results["fast"][stem].astype(np.float64)
                snrs.append(
                    10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                )
                peaks.append(np.max(np.abs(error)))
            print(
                f"| {seconds}s | {pair} | previous | {times['previous']:.3f}s | — | — | — |",
                flush=True,
            )
            print(
                f"| {seconds}s | {pair} | fast | **{times['fast']:.3f}s** | "
                f"{times['previous'] / times['fast']:.2f}x | {min(snrs):.2f} dB | "
                f"{max(peaks):.6g} |",
                flush=True,
            )
finally:
    select_fast(True)
