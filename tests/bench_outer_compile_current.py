"""Compare whole-forward compilation with the current standalone GPU path.

Run through ``metalq submit -w``. Each variant owns a separate loaded model so
compile caches and DConv construction cannot cross-contaminate the comparison.
"""

import argparse
import os
import statistics
import time

import numpy as np

from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(917 + seconds)
    n = seconds * 44_100
    t = np.arange(n, dtype=np.float32) / 44_100
    return (0.04 * np.sin(2 * np.pi * 220 * t)[None, :] +
            0.01 * rng.standard_normal((2, n), dtype=np.float32)).astype(np.float32)


parser = argparse.ArgumentParser()
parser.add_argument("--long-first", action="store_true")
args = parser.parse_args()

variants = {
    "current": ("0", "1"),
    "outer-only": ("1", "0"),
    "outer+dconv": ("1", "1"),
}
models = {name: Separator(seed=481) for name in variants}


def run(name, wav):
    outer, dconv = variants[name]
    os.environ["DEMUCS_MLX_COMPILE_FORWARD"] = outer
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = dconv
    start = time.perf_counter()
    _, stems = models[name].separate_tensor(wav)
    return time.perf_counter() - start, stems


def fidelity(reference, candidate):
    snrs, peaks = [], []
    for stem in reference:
        want = reference[stem].astype(np.float64)
        error = want - candidate[stem].astype(np.float64)
        energy = np.sum(want * want)
        noise = np.sum(error * error)
        snrs.append(float("inf") if noise == 0 else 10 * np.log10(energy / noise))
        peaks.append(np.max(np.abs(error)))
    return min(snrs), max(peaks)


print("## Standalone whole-forward compile comparison", flush=True)
print("**Settings:** htdemucs, one shift, 25% overlap, batch two, seed 481", flush=True)
print(
    "| Input | Run | Current | Outer only | Outer + DConv | Min SNR vs current | Peak error |",
    flush=True,
)
print("|---:|---|---:|---:|---:|---:|---:|", flush=True)
try:
    for seconds in ((60, 30) if args.long_first else (30, 60)):
        wav = signal(seconds)
        rows = []
        for run_name, order in (
            ("cold", ("current", "outer-only", "outer+dconv")),
            ("warm A", ("outer+dconv", "current", "outer-only")),
            ("warm B", ("outer-only", "outer+dconv", "current")),
            ("warm C", ("current", "outer-only", "outer+dconv")),
            ("warm D", ("outer+dconv", "outer-only", "current")),
        ):
            measured = {name: run(name, wav) for name in order}
            times = {name: measured[name][0] for name in variants}
            snr, peak = fidelity(measured["current"][1], measured["outer-only"][1])
            snr2, peak2 = fidelity(measured["current"][1], measured["outer+dconv"][1])
            rows.append(times)
            print(f"| {seconds}s | {run_name} | {times['current']:.3f}s | "
                  f"{times['outer-only']:.3f}s | {times['outer+dconv']:.3f}s | "
                  f"{min(snr, snr2):.1f} dB | {max(peak, peak2):.3g} |", flush=True)
        for name in ("outer-only", "outer+dconv"):
            baseline = statistics.mean(row["current"] for row in rows[1:])
            candidate = statistics.mean(row[name] for row in rows[1:])
            print(f"> **{seconds}s {name}:** {100 * (baseline - candidate) / baseline:.1f}% "
                  "less warmed wall time than current.", flush=True)
finally:
    os.environ.pop("DEMUCS_MLX_COMPILE_FORWARD", None)
    os.environ.pop("DEMUCS_MLX_COMPILE_DCONV", None)
