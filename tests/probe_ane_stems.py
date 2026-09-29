"""Compare complete single-segment stems with GPU and ANE convolution paths."""

import numpy as np

from demucs_mlx.ane import LENGTH
from demucs_mlx.api import Separator

rng = np.random.default_rng(481)
audio = (rng.standard_normal((2, LENGTH), dtype=np.float32) * 0.1).astype(np.float32)
gpu = Separator(shifts=0, batch_size=1)
_, reference = gpu.separate_tensor(audio)
with Separator(shifts=0, batch_size=1, ane_time_encoder=True) as ane:
    _, converted = ane.separate_tensor(audio)
    worker = ane._ane_worker
    print("## Single-segment stem fidelity")
    print("| Stem | SNR | Peak error |")
    print("|---|---:|---:|")
    for name in reference:
        want = reference[name].astype(np.float64)
        error = want - converted[name].astype(np.float64)
        snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
        print(f"| {name} | {snr:.2f} dB | {np.max(np.abs(error)):.6g} |")
    print(
        f"**ANE:** {worker.predictions} predictions, {worker.busy_seconds:.3f}s execution, "
        f"{worker.wait_seconds:.3f}s wait, {worker.transfer_seconds:.3f}s transfer"
    )
if worker._worker.is_alive():
    raise AssertionError("ANE worker remained alive")
try:
    ane.separate_tensor(audio)
except RuntimeError as exc:
    assert "closed" in str(exc)
else:
    raise AssertionError("Closed ANE separator silently fell back to GPU")
