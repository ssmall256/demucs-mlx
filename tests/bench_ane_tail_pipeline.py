"""Diagnostic complete-stem comparison for the low-fidelity ANE tail.

Run through metalq. The tail is attached privately for this probe only; it is
not exposed through the Separator API or CLI.
"""

import time

import numpy as np

from demucs_mlx.ane import WaveformTail
from demucs_mlx.api import Separator


def signal(seconds):
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)


audio = signal(30)
gpu = Separator(seed=481)
with Separator(seed=481, ane_time_encoder=True) as ane:
    tail = WaveformTail()
    model = ane._model.models[0]
    model._ane_time_tail = tail
    try:
        print("## Complete-stem ANE tail diagnostic", flush=True)
        print("**Settings:** 30s, default inference, seed=481", flush=True)
        print("| Pair | Path | Wall | Conv execution | Tail execution | Tail wait |", flush=True)
        print("|---:|---|---:|---:|---:|---:|", flush=True)
        for pair in (1, 2):
            outputs = {}
            for name, separator in (("GPU", gpu), ("ANE tail", ane)):
                before_conv = ane._ane_worker.busy_seconds
                before_tail = tail.busy_seconds
                before_wait = tail.wait_seconds
                start = time.perf_counter()
                _, stems = separator.separate_tensor(audio)
                elapsed = time.perf_counter() - start
                outputs[name] = stems
                print(
                    f"| {pair} | {name} | **{elapsed:.3f}s** | "
                    f"{ane._ane_worker.busy_seconds - before_conv:.3f}s | "
                    f"{tail.busy_seconds - before_tail:.3f}s | "
                    f"{tail.wait_seconds - before_wait:.3f}s |",
                    flush=True,
                )
            print("| Stem | SNR | Peak error |", flush=True)
            print("|---|---:|---:|", flush=True)
            for stem, want_array in outputs["GPU"].items():
                want = want_array.astype(np.float64)
                error = want - outputs["ANE tail"][stem].astype(np.float64)
                snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
                print(f"| {stem} | {snr:.2f} dB | {np.max(np.abs(error)):.6g} |", flush=True)
    finally:
        del model._ane_time_tail
        tail.close()
assert not tail._worker.is_alive()
