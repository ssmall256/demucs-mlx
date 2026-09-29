"""Measure the supported HTDemucs modes with default inference settings.

Run through ``metalq submit -w`` to serialize Metal workloads.
"""

import time

import numpy as np

from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


def run(separator, audio):
    start = time.perf_counter()
    _, stems = separator.separate_tensor(audio)
    elapsed = time.perf_counter() - start
    assert all(value.shape == audio.shape for value in stems.values())
    return elapsed, stems


print("## HTDemucs model modes", flush=True)
print("**Settings:** shifts=1, overlap=0.25, split=True, batch_size=2, seed=481", flush=True)
print(
    "**Timing:** separation and materialization of all stems; model load and audio I/O excluded",
    flush=True,
)
print(
    "| Mode | Models | Segment | Stems | 30 s run 1 | 30 s run 2 | 60 s run 1 | 60 s run 2 |",
    flush=True,
)
print("|---|---:|---:|---:|---:|---:|---:|---:|", flush=True)

audio = {seconds: signal(seconds) for seconds in (30, 60)}
for name in ("htdemucs", "htdemucs_6s", "htdemucs_ft"):
    separator = Separator(model=name, seed=481)
    models = len(separator.model.models)
    segment = separator.model.models[0].segment
    stems = len(separator.model.sources)
    run(separator, audio[30])
    timings = {}
    for seconds in (30, 60):
        timings[seconds] = [run(separator, audio[seconds])[0] for _ in range(2)]
    print(
        f"| `{name}` | {models} | {segment:g} s | {stems} | "
        f"**{timings[30][0]:.3f} s** | **{timings[30][1]:.3f} s** | "
        f"**{timings[60][0]:.3f} s** | **{timings[60][1]:.3f} s** |",
        flush=True,
    )
    del separator
