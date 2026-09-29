"""Compare a selected fine-tuned stem with the full four-model pass.

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
    return time.perf_counter() - start, stems


full = Separator(model="htdemucs_ft", seed=481)
selected = Separator(model="htdemucs_ft", stem="vocals", seed=481)

print("## Fine-tuned single-stem benchmark", flush=True)
print("**Settings:** shifts=1, overlap=0.25, split=True, batch_size=2, seed=481", flush=True)
print(
    "**Timing:** separation and output materialization; model load and audio I/O excluded",
    flush=True,
)
print("| Input | Pair | Full four stems | Vocals only | Speedup | SNR | Peak error |", flush=True)
print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)

run(full, signal(30))
run(selected, signal(30))
for seconds in (30, 60):
    audio = signal(seconds)
    for pair in (1, 2):
        order = (("full", full), ("selected", selected))
        if pair == 2:
            order = tuple(reversed(order))
        measurements = {name: run(separator, audio) for name, separator in order}
        full_time, all_stems = measurements["full"]
        selected_time, chosen_stems = measurements["selected"]
        assert list(chosen_stems) == ["vocals"]
        reference = all_stems["vocals"].astype(np.float64)
        result = chosen_stems["vocals"].astype(np.float64)
        difference = reference - result
        square_error = np.sum(difference * difference)
        snr = (
            float("inf")
            if square_error == 0
            else 10 * np.log10(np.sum(reference * reference) / square_error)
        )
        peak = np.max(np.abs(difference))
        print(
            f"| {seconds} s | {pair} | **{full_time:.3f} s** | "
            f"**{selected_time:.3f} s** | **{full_time / selected_time:.2f}x** | "
            f"{snr:.2f} dB | {peak:.3g} |",
            flush=True,
        )
