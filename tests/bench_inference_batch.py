"""Compare full separation throughput across segment batch sizes.

Run through metalq submit -w. The same Separator instance is reused to keep
model loading and weight cache state out of the timed region.
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


separator = Separator(seed=481)
print("## Inference segment batch-size sweep", flush=True)
print("**Settings:** shifts=1, overlap=0.25, split=True, seed=481", flush=True)
print("| Input | Pass | Batch | Wall | Audio / wall | Minimum stem SNR vs batch 2 |", flush=True)
print("|---:|---:|---:|---:|---:|---:|", flush=True)
for seconds in (30, 60):
    audio = signal(seconds)
    baseline = None
    for pass_number, sizes in ((1, (2, 4, 8)), (2, (8, 4, 2))):
        for batch_size in sizes:
            separator.batch_size = batch_size
            start = time.perf_counter()
            _, stems = separator.separate_tensor(audio)
            elapsed = time.perf_counter() - start
            if baseline is None:
                baseline = stems
            snrs = []
            for stem, want_array in baseline.items():
                want = want_array.astype(np.float64)
                error = want - stems[stem].astype(np.float64)
                snrs.append(
                    10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                )
            print(
                f"| {seconds}s | {pass_number} | {batch_size} | **{elapsed:.3f}s** | "
                f"{seconds / elapsed:.2f}x | {min(snrs):.2f} dB |",
                flush=True,
            )
