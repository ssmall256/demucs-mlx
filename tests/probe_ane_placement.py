"""Check Core ML placement for HTDemucs-sized convolutions.

Run through metalq as shown in docs/ane-prototype.md.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import torch


class Conv1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 48, 8, stride=4, padding=2)

    def forward(self, x):
        return self.conv(x)


class Conv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 48, (1, 8), stride=(1, 4), padding=(0, 2))

    def forward(self, x):
        return self.conv(x)


def preferred_device(module_type, length):
    shape = (2, 2, length) if module_type is Conv1 else (2, 2, 1, length)
    traced = torch.jit.trace(module_type().eval(), torch.zeros(shape), check_trace=False)
    with TemporaryDirectory() as temporary:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=shape)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        package = Path(temporary) / "probe.mlpackage"
        ml.save(str(package))
        compiled = ct.models.utils.compile_model(str(package))
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        operations = plan.model_structure.program.functions["main"].block.operations
        for operation in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is not None:
                return type(usage.preferred_compute_device).__name__
    raise RuntimeError("Core ML returned no device usage for the convolution")


print("## Core ML convolution placement")
print("| Layout | Samples | Preferred device |")
print("|---|---:|---|")
for kind in (Conv1, Conv2):
    for length in (2_048, 16_384, 85_995, 343_980):
        print(f"| `{kind.__name__}` | {length} | `{preferred_device(kind, length)}` |", flush=True)
