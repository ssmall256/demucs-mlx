"""Inspect ANE placement for individual later HTDemucs waveform stages.

Run through metalq. Full-length inputs preserve each stage's GroupNorm
statistics and therefore can be compared exactly with the MLX encoder.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional

from demucs_mlx.ane import _device_placement
from demucs_mlx.secure_demucs import get_restricted_demucs_model

model = get_restricted_demucs_model("htdemucs").model.models[0].eval()


class FixedStage(torch.nn.Module):
    def __init__(self, layer, right_pad):
        super().__init__()
        self.layer = layer
        self.right_pad = right_pad

    def forward(self, x):
        # HEncLayer's dynamic padding traces through aten::Int, which Core ML
        # Tools cannot convert. Fixed-shape right padding is equivalent.
        return self.layer(functional.pad(x, (0, self.right_pad)))


print("## Full-length later waveform stage placement", flush=True)
print("| Stage | Input shape | ANE ops | CPU ops | ANE cost |", flush=True)
print("|---:|---|---:|---:|---:|", flush=True)
for stage, length in ((1, 85_995), (2, 21_499), (3, 5_375)):
    layer = model.tencoder[stage].eval()
    shape = (2, layer.conv.in_channels, length)
    with torch.no_grad():
        traced = torch.jit.trace(
            FixedStage(layer, (-length) % layer.stride), torch.zeros(shape), check_trace=False
        )
    with TemporaryDirectory() as temporary:
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        package = Path(temporary) / "stage.mlpackage"
        converted.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        plan = _device_placement(compiled)
        counts, costs = plan["operations"], plan["estimated_cost"]
        ane_count = sum(v for k, v in counts.items() if "NeuralEngine" in k)
        cpu_count = sum(v for k, v in counts.items() if "CPU" in k)
        ane_cost = sum(v for k, v in costs.items() if "NeuralEngine" in k)
        print(
            f"| {stage + 1} | `{shape}` | {ane_count} | {cpu_count} | {ane_cost:.3f} |",
            flush=True,
        )
