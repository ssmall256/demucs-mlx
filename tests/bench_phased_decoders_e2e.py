"""Compare both phased decoders with the previous HTDemucs decoder path.

Run through metalq. Both variants use one loaded model and identical inputs.
"""

import time

import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_layers import ConvTranspose1dNCL, ConvTranspose2dNCHW


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


def previous_1d(self, x):
    return self.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def previous_2d(self, x):
    return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


current_1d = ConvTranspose1dNCL.__call__
current_2d = ConvTranspose2dNCHW.__call__


def select_current(enabled):
    ConvTranspose1dNCL.__call__ = current_1d if enabled else previous_1d
    ConvTranspose2dNCHW.__call__ = current_2d if enabled else previous_2d


separator = Separator(seed=481)
print("## Both phased decoders versus previous deconvolution", flush=True)
print(
    "| Input | Pair | Path | Wall | Previous / phased | Minimum stem SNR | Peak error |",
    flush=True,
)
print("|---:|---:|---|---:|---:|---:|---:|", flush=True)
try:
    audio = signal(30)
    select_current(False)
    separator.separate_tensor(audio)
    select_current(True)
    separator.separate_tensor(audio)
    for seconds in (30, 60):
        audio = signal(seconds)
        for pair, order in (
            (1, ("previous", "phased")),
            (2, ("previous", "phased")),
            (3, ("phased", "previous")),
        ):
            times = {}
            results = {}
            for name in order:
                select_current(name == "phased")
                start = time.perf_counter()
                _, stems = separator.separate_tensor(audio)
                times[name] = time.perf_counter() - start
                results[name] = stems
            snrs = []
            peaks = []
            for stem, reference in results["previous"].items():
                want = reference.astype(np.float64)
                error = want - results["phased"][stem].astype(np.float64)
                snrs.append(
                    10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                )
                peaks.append(np.max(np.abs(error)))
            print(
                f"| {seconds}s | {pair} | previous | {times['previous']:.3f}s | — | — | — |",
                flush=True,
            )
            print(
                f"| {seconds}s | {pair} | phased | **{times['phased']:.3f}s** | "
                f"{times['previous'] / times['phased']:.2f}x | "
                f"{min(snrs):.2f} dB | {max(peaks):.6g} |",
                flush=True,
            )
finally:
    select_current(True)
