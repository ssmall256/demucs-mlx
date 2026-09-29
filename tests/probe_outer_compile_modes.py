"""Check compiled-forward parity on the other Demucs inference modes.

Run through ``metalq submit -w``.
"""

import os
import time

import numpy as np

from demucs_mlx.api import Separator

rng = np.random.default_rng(935)
n = 30 * 44_100
wav = (0.01 * rng.standard_normal((2, n), dtype=np.float32)).astype(np.float32)

print("## Compiled forward model-mode parity", flush=True)
print("**Settings:** 30 s, one shift, batch two, seed 481", flush=True)
print("| Model | Eager + DConv | Outer compile | Min stem SNR | Peak error |", flush=True)
print("|---|---:|---:|---:|---:|", flush=True)
for name, stem in (("htdemucs_6s", None), ("htdemucs_ft", "bass"), ("hdemucs_mmi", None)):
    separator = Separator(model=name, stem=stem, seed=481)
    os.environ["DEMUCS_MLX_COMPILE_FORWARD"] = "0"
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "1"
    start = time.perf_counter()
    _, reference = separator.separate_tensor(wav)
    eager = time.perf_counter() - start

    os.environ["DEMUCS_MLX_COMPILE_FORWARD"] = "1"
    os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "0"
    start = time.perf_counter()
    _, candidate = separator.separate_tensor(wav)
    compiled = time.perf_counter() - start
    snrs, peaks = [], []
    for source in reference:
        want = reference[source].astype(np.float64)
        error = want - candidate[source].astype(np.float64)
        noise = np.sum(error * error)
        snrs.append(float("inf") if noise == 0 else 10 * np.log10(np.sum(want * want) / noise))
        peaks.append(np.max(np.abs(error)))
    print(f"| {name}{' (' + stem + ')' if stem else ''} | {eager:.3f}s | "
          f"{compiled:.3f}s | {min(snrs):.1f} dB | {max(peaks):.3g} |", flush=True)

os.environ.pop("DEMUCS_MLX_COMPILE_FORWARD", None)
os.environ.pop("DEMUCS_MLX_COMPILE_DCONV", None)
