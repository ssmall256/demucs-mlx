"""Find exact-convolution tile sizes that Core ML places on ANE.

Run with metalq submit -w. Each tile width yields an integer number of
stride-four outputs, and each output count divides the full 85,995 outputs.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch

from demucs_mlx.ane import _device_placement


class Conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 48, 8, stride=4, padding=0)

    def forward(self, x):
        return self.conv(x)


print("## Exact first-convolution tile placement", flush=True)
print("| Output/tile | Tiles | Input/tile | ANE ops | CPU ops | ANE cost |", flush=True)
print("|---:|---:|---:|---:|---:|---:|", flush=True)
for output_length in (4095, 5733, 12285, 17199):
    input_length = 4 * output_length + 4
    shape = (2, 2, input_length)
    traced = torch.jit.trace(Conv().eval(), torch.zeros(shape), check_trace=False)
    with TemporaryDirectory() as temporary:
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        package = Path(temporary) / "tile.mlpackage"
        model.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        plan = _device_placement(compiled)
        counts = plan["operations"]
        costs = plan["estimated_cost"]
        ane_count = sum(v for k, v in counts.items() if "NeuralEngine" in k)
        cpu_count = sum(v for k, v in counts.items() if "CPU" in k)
        ane_cost = sum(v for k, v in costs.items() if "NeuralEngine" in k)
        print(
            f"| {output_length} | {85995 // output_length} | {input_length} | "
            f"{ane_count} | {cpu_count} | {ane_cost:.3f} |",
            flush=True,
        )
