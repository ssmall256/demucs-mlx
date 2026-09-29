"""Alternate standard and four-phase waveform deconvolution in separation.

Run through metalq. The same loaded model and audio are used for both paths.
"""

import time

import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_layers import ConvTranspose1dNCL


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


phased = ConvTranspose1dNCL.__call__


def original(self, x):
    nlc = x.transpose(0, 2, 1)
    return self.conv(nlc).transpose(0, 2, 1)


separator = Separator(seed=481)
print("## Four-phase waveform deconvolution full separation", flush=True)
print(
    "| Input | Pair | Path | Wall | Original / phased | Minimum stem SNR | Peak error |",
    flush=True,
)
print("|---:|---:|---|---:|---:|---:|---:|", flush=True)
try:
    audio = signal(30)
    ConvTranspose1dNCL.__call__ = original
    separator.separate_tensor(audio)
    ConvTranspose1dNCL.__call__ = phased
    separator.separate_tensor(audio)
    for seconds in (30, 60):
        audio = signal(seconds)
        for pair, order in (
            (1, ("original", "phased")),
            (2, ("original", "phased")),
            (3, ("phased", "original")),
        ):
            times = {}
            results = {}
            for name in order:
                ConvTranspose1dNCL.__call__ = original if name == "original" else phased
                start = time.perf_counter()
                _, stems = separator.separate_tensor(audio)
                times[name] = time.perf_counter() - start
                results[name] = stems
            snrs = []
            peaks = []
            for stem, reference in results["original"].items():
                want = reference.astype(np.float64)
                error = want - results["phased"][stem].astype(np.float64)
                snrs.append(
                    10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                )
                peaks.append(np.max(np.abs(error)))
            print(
                f"| {seconds}s | {pair} | original | {times['original']:.3f}s | — | — | — |",
                flush=True,
            )
            print(
                f"| {seconds}s | {pair} | phased | **{times['phased']:.3f}s** | "
                f"{times['original'] / times['phased']:.2f}x | "
                f"{min(snrs):.2f} dB | {max(peaks):.6g} |",
                flush=True,
            )
finally:
    ConvTranspose1dNCL.__call__ = phased
