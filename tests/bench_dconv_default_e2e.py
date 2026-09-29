"""Compare the shipped DConv inference path with its eager fallback.

Run through ``metalq submit -w`` to serialize Metal measurements.
"""

import argparse
import os
import time

import numpy as np
from mlx.utils import tree_flatten

from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


parser = argparse.ArgumentParser()
parser.add_argument("--cold-only", choices=("eager", "compiled"))
args = parser.parse_args()

separator = Separator(seed=481)
parameters_before = [name for name, _ in tree_flatten(separator.model.models[0].parameters())]


def run(path, audio):
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "1" if path == "compiled" else "0"
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    return time.perf_counter() - start, stems


print("## Default compiled DConv versus eager fallback", flush=True)
print("**Settings:** default shifts, overlap, split and batch size; seed 481", flush=True)
print("| Input | Pair | Eager | Compiled | Speedup | Minimum stem SNR | Peak error |", flush=True)
print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)

audio = signal(30)
if args.cold_only:
    cold, _ = run(args.cold_only, audio)
    print(f"> Fresh-process {args.cold_only} first pass: **{cold:.3f} s**", flush=True)
    raise SystemExit(0)

run("eager", audio)
run("compiled", audio)
for seconds in (30, 60):
    audio = signal(seconds)
    for pair, order in ((1, ("eager", "compiled")), (2, ("compiled", "eager"))):
        measured = {path: run(path, audio) for path in order}
        eager_time, eager_stems = measured["eager"]
        fast_time, fast_stems = measured["compiled"]
        snrs = []
        peaks = []
        for name, reference in eager_stems.items():
            want = reference.astype(np.float64)
            error = want - fast_stems[name].astype(np.float64)
            noise = np.sum(error * error)
            snrs.append(float("inf") if noise == 0 else 10 * np.log10(np.sum(want * want) / noise))
            peaks.append(float(np.max(np.abs(error))))
        print(
            f"| {seconds}s | {pair} | {eager_time:.3f}s | **{fast_time:.3f}s** | "
            f"**{eager_time / fast_time:.2f}x** | {min(snrs):.2f} dB | {max(peaks):.3g} |",
            flush=True,
        )

parameters_after = [name for name, _ in tree_flatten(separator.model.models[0].parameters())]
assert parameters_after == parameters_before
print("> ✅ Parameter tree is unchanged after graph compilation.", flush=True)
