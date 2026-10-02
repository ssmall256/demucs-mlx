"""Alternating default-settings GPU and ANE separation benchmark.

Run with metalq submit -w. Output includes full-stem parity and worker times.
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


def main():
    gpu = Separator(seed=481)
    with Separator(seed=481, ane_time_encoder=True) as ane:
        worker = ane._ane_worker
        print("## Default-inference alternating benchmark", flush=True)
        print(
            "**Settings:** shifts=1, overlap=0.25, split=True, batch_size=2, seed=481",
            flush=True,
        )
        print(
            "| Input | Run | Path | Wall time | Audio / wall | ANE execution | Wait | Transfer |",
            flush=True,
        )
        print("|---:|---:|---|---:|---:|---:|---:|---:|", flush=True)
        for seconds in (30, 60):
            audio = signal(seconds)
            for run in (1, 2):
                previous = {}
                results = {}
                for name, separator in (("GPU", gpu), ("ANE", ane)):
                    before = (worker.busy_seconds, worker.wait_seconds, worker.transfer_seconds)
                    start = time.perf_counter()
                    _, stems = separator.separate_tensor(audio)
                    elapsed = time.perf_counter() - start
                    after = (worker.busy_seconds, worker.wait_seconds, worker.transfer_seconds)
                    metrics = tuple(end - begin for begin, end in zip(before, after))
                    print(
                        f"| {seconds}s | {run} | {name} | **{elapsed:.3f}s** | "
                        f"{seconds / elapsed:.2f}x | {metrics[0]:.3f}s | "
                        f"{metrics[1]:.3f}s | {metrics[2]:.3f}s |",
                        flush=True,
                    )
                    results[name] = stems
                    previous[name] = elapsed
                print(
                    f"> Run {run}, {seconds}s: GPU / ANE = "
                    f"**{previous['GPU'] / previous['ANE']:.2f}x**",
                    flush=True,
                )
                print("| Stem | SNR | Peak error |", flush=True)
                print("|---|---:|---:|", flush=True)
                for stem, reference in results["GPU"].items():
                    want = reference.astype(np.float64)
                    error = want - results["ANE"][stem].astype(np.float64)
                    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                    print(f"| {stem} | {snr:.2f} dB | {np.max(np.abs(error)):.6g} |", flush=True)


if __name__ == "__main__":
    main()
