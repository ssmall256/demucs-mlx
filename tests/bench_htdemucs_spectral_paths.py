"""Compare the installed mlx-spectro compiled and eager paths at model shapes.

Run through ``metalq submit -w`` to serialize Metal workloads.
"""

import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def measure(function, repeats=7):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(function())
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


model = Separator(seed=481).model.models[0]
pair = model._spectral
compiled_stft, compiled_istft = pair.stft, pair.istft
mix = mx.random.normal((2, 2, 343_980), dtype=mx.float32) * 0.1
mx.eval(mix)
spec = model._spec(mix)
mx.eval(spec)
masked = mx.broadcast_to(spec[:, None], (2, 4, *spec.shape[1:]))
mx.eval(masked)

print("## HTDemucs spectral path comparison", flush=True)
print("**Runtime:** installed mlx-spectro; batch 2, 7.8-second segments", flush=True)
print("| Round | Path | STFT | ISTFT |", flush=True)
print("|---:|---|---:|---:|", flush=True)

outputs = {}
for round_number in (1, 2):
    order = ("compiled", "eager") if round_number == 1 else ("eager", "compiled")
    for path in order:
        pair.stft = compiled_stft if path == "compiled" else pair.stft_eager
        pair.istft = compiled_istft if path == "compiled" else pair.istft_eager
        mx.eval(model._spec(mix), model._ispec(masked, length=343_980))
        stft_ms = measure(lambda: model._spec(mix))
        istft_ms = measure(lambda: model._ispec(masked, length=343_980))
        outputs[path] = (np.asarray(model._spec(mix)), np.asarray(model._ispec(masked, 343_980)))
        print(f"| {round_number} | {path} | {stft_ms:.3f} ms | {istft_ms:.3f} ms |", flush=True)

for name, index in (("STFT", 0), ("ISTFT", 1)):
    difference = outputs["compiled"][index] - outputs["eager"][index]
    print(f"> {name} peak difference: **{np.max(np.abs(difference)):.3g}**", flush=True)
