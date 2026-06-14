"""Focused split-mode overlap-add benchmark.

Run via metalq for MLX/Metal timing:
    metalq submit -w -n "overlap-add bench" -- python tests/bench_overlap_add.py
"""
from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.apply_mlx import apply_model


class _IdentityModel:
    samplerate = 44_100
    segment = 7.8
    sources = ["mix"]

    def valid_length(self, length: int) -> int:
        return int(length)

    def __call__(self, x: mx.array) -> mx.array:
        return x[:, None, :, :]


def _make_audio(seconds: int) -> np.ndarray:
    sr = _IdentityModel.samplerate
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    x = np.stack([x, np.roll(x, 100)]).astype(np.float32)
    return (x / np.abs(x).max() * 0.9).astype(np.float32)


def _sync(out: mx.array) -> None:
    mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def _measure(seconds: int, batch_size: int, warmup: int, iters: int) -> dict[str, float]:
    wav = _make_audio(seconds)
    mix = mx.array(wav[None, ...])
    model = _IdentityModel()

    for _ in range(warmup):
        out = apply_model(model, mix, shifts=0, split=True, overlap=0.25, batch_size=batch_size)
        _sync(out)

    times = []
    last = None
    for _ in range(iters):
        start = time.perf_counter()
        out = apply_model(model, mix, shifts=0, split=True, overlap=0.25, batch_size=batch_size)
        _sync(out)
        times.append(time.perf_counter() - start)
        last = out

    assert last is not None
    reconstructed = np.asarray(last)[0, 0]
    median = statistics.median(times)
    return {
        "median": median,
        "rt_factor": seconds / median,
        "output_peak": float(np.abs(reconstructed).max()),
        "max_error": float(np.max(np.abs(reconstructed - wav))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark split-mode overlap-add accumulation")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    print("## Overlap-Add Benchmark")
    print(f"**Warmup:** `{args.warmup}`  **Iterations:** `{args.iters}`")
    print()
    print("| seconds | batch_size | median seconds | realtime | output peak | max error |")
    print("|---:|---:|---:|---:|---:|---:|")
    for seconds in (20, 60):
        for batch_size in (1, 8):
            row = _measure(seconds, batch_size, args.warmup, args.iters)
            print(
                f"| {seconds} | {batch_size} | **{row['median']:.6f}** | "
                f"**{row['rt_factor']:.2f}x** | **{row['output_peak']:.6f}** | "
                f"**{row['max_error']:.6g}** |"
            )


if __name__ == "__main__":
    main()
