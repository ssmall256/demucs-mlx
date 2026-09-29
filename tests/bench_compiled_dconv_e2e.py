"""Compare eager and compiled DConv blocks in complete separation.

Run through ``metalq submit -w``. Only the loaded benchmark model is rewired.
"""

import argparse
import os
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


class CompiledBlock:
    def __init__(self, block):
        self.function = mx.compile(lambda x: block(x))

    def __call__(self, value):
        return self.function(value)


parser = argparse.ArgumentParser()
parser.add_argument("--cold-only", choices=("eager", "hot", "all"))
args = parser.parse_args()
os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "0"

separator = Separator(seed=481)
model = separator.model.models[0]
blocks = []
for branch in ("encoder", "tencoder", "decoder", "tdecoder"):
    for index, layer in enumerate(getattr(model, branch)):
        dconv = getattr(layer, "dconv", None)
        if dconv is not None:
            original = dconv.layers
            compiled = [CompiledBlock(block) for block in original]
            blocks.append((f"{branch}.{index}", dconv, original, compiled))

hot_names = {"encoder.0", "tencoder.0", "decoder.3", "tdecoder.3"}


def select(path):
    for name, dconv, original, compiled in blocks:
        dconv.layers = (
            compiled if path == "all" or (path == "hot" and name in hot_names) else original
        )


def run(path, audio):
    select(path)
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    return time.perf_counter() - start, stems


print("## Compiled DConv blocks in complete HTDemucs separation", flush=True)
print("**Settings:** default shifts, overlap, split and batch size; seed 481", flush=True)
print("| Input | Pair | Path | Wall | Eager / path | Minimum stem SNR | Peak error |", flush=True)
print("|---:|---:|---|---:|---:|---:|---:|", flush=True)

if args.cold_only:
    cold, _ = run(args.cold_only, signal(30))
    print(f"> Fresh-process {args.cold_only} first pass: **{cold:.3f} s**", flush=True)
    raise SystemExit(0)

try:
    audio = signal(30)
    run("eager", audio)
    for path in ("hot", "all"):
        cold, _ = run(path, audio)
        print(f"> Cold {path} first pass: **{cold:.3f} s**", flush=True)

    for seconds in (30, 60):
        audio = signal(seconds)
        for pair, order in (
            (1, ("eager", "hot", "all")),
            (2, ("all", "hot", "eager")),
        ):
            times = {}
            outputs = {}
            for path in order:
                times[path], outputs[path] = run(path, audio)
            print(
                f"| {seconds}s | {pair} | eager | **{times['eager']:.3f}s** | — | — | — |",
                flush=True,
            )
            for path in ("hot", "all"):
                snrs = []
                peaks = []
                for stem, reference in outputs["eager"].items():
                    want = reference.astype(np.float64)
                    error = want - outputs[path][stem].astype(np.float64)
                    energy = np.sum(want * want)
                    noise = np.sum(error * error)
                    snrs.append(float("inf") if noise == 0 else 10 * np.log10(energy / noise))
                    peaks.append(float(np.max(np.abs(error))))
                print(
                    f"| {seconds}s | {pair} | {path} | **{times[path]:.3f}s** | "
                    f"{times['eager'] / times[path]:.2f}x | {min(snrs):.2f} dB | "
                    f"{max(peaks):.3g} |",
                    flush=True,
                )
finally:
    select("eager")
