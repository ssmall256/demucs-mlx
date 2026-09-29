"""Test whether selective FP32 improves Core ML tail fidelity with ANE placement.

Run through metalq submit -w. The official restricted checkpoint supplies the
test input and reference, so the only variable is Core ML precision.
"""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch

from demucs_mlx.ane import LENGTH, _device_placement, _torch_tail
from demucs_mlx.secure_demucs import get_restricted_demucs_model

torch.manual_seed(481)
model = get_restricted_demucs_model("htdemucs").model.models[0].eval()
with torch.no_grad():
    mix = torch.randn((2, 2, LENGTH), dtype=torch.float32) * 0.1
    normalized = (mix - mix.mean(dim=(1, 2), keepdim=True)) / (
        1e-5 + mix.std(dim=(1, 2), keepdim=True, unbiased=False)
    )
    initial = model.tencoder[0](normalized).numpy()
    reference = []
    x = torch.from_numpy(initial)
    for layer in model.tencoder[1:]:
        x = layer(x)
        reference.append(x.numpy())
    traced = torch.jit.trace(
        _torch_tail(model), torch.zeros((2, 48, 85_995)), check_trace=False
    )


def snr(want, got):
    error = want.astype(np.float64) - got.astype(np.float64)
    return 10 * np.log10(np.sum(want.astype(np.float64) ** 2) / max(np.sum(error**2), 1e-30))


print("## Waveform tail precision probe", flush=True)
for name, precision in (
    ("FP32", ct.precision.FLOAT32),
    (
        "FP16 convolutions only",
        ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type == "conv"),
    ),
    (
        "FP32 normalization reductions",
        ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.op_type
            not in {"reduce_mean", "reduce_sum", "rsqrt", "sqrt", "group_norm", "layer_norm"}
        ),
    ),
):
    print(f"### {name}", flush=True)
    with TemporaryDirectory() as temporary:
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="conv0", shape=initial.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name=f"y{index}", dtype=np.float32) for index in (1, 2, 3)],
            convert_to="mlprogram",
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        operations = converted._mil_program.functions["main"].operations
        print(f"**MIL ops:** `{dict(Counter(op.op_type for op in operations))}`", flush=True)
        package = Path(temporary) / "tail.mlpackage"
        converted.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        print(f"**Placement:** `{_device_placement(compiled)}`", flush=True)
        got = converted.predict({"conv0": initial})
        print("| Stage | SNR vs PyTorch | Peak error |", flush=True)
        print("|---:|---:|---:|", flush=True)
        for stage, want in enumerate(reference, 1):
            candidate = got[f"y{stage}"]
            peak = np.max(np.abs(want.astype(np.float64) - candidate.astype(np.float64)))
            print(f"| {stage + 1} | {snr(want, candidate):.2f} dB | {peak:.6g} |", flush=True)
