"""Probe the full-length waveform tail after the exact tiled first conv.

Run with metalq submit -w. Stage 0 starts from the precomputed first conv;
the model then emits all four stage outputs needed by the decoder.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional

from demucs_mlx.ane import _device_placement
from demucs_mlx.secure_demucs import get_restricted_demucs_model


class Tail(torch.nn.Module):
    def __init__(self, model, include_stage0):
        super().__init__()
        self.stage0 = model.tencoder[0]
        self.later = torch.nn.ModuleList(model.tencoder[1:])
        self.include_stage0 = include_stage0

    def forward(self, x):
        outputs = []
        if self.include_stage0:
            x = functional.gelu(self.stage0.norm1(x))
            x = self.stage0.dconv(x)
            x = functional.glu(self.stage0.norm2(self.stage0.rewrite(x)), dim=1)
            outputs.append(x)
        for stage in self.later:
            # All three full-length inputs are one sample short of a multiple
            # of four. Fixed padding avoids a tracing-only aten::Int op.
            x = stage(functional.pad(x, (0, 1)))
            outputs.append(x)
        return tuple(outputs)


model = get_restricted_demucs_model("htdemucs").model.models[0].eval()
shape = (2, 48, 85_995)
print("## Full-length waveform tail placement", flush=True)
print("| Tail | ANE ops | CPU ops | ANE cost |", flush=True)
print("|---|---:|---:|---:|", flush=True)
for include_stage0 in (False, True):
    module = Tail(model, include_stage0).eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, torch.zeros(shape), check_trace=False)
    with TemporaryDirectory() as temporary:
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="conv0", shape=shape, dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        package = Path(temporary) / "tail.mlpackage"
        converted.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        plan = _device_placement(compiled)
        counts, costs = plan["operations"], plan["estimated_cost"]
        ane_count = sum(v for k, v in counts.items() if "NeuralEngine" in k)
        cpu_count = sum(v for k, v in counts.items() if "CPU" in k)
        ane_cost = sum(v for k, v in costs.items() if "NeuralEngine" in k)
        label = "stages 2–4" if not include_stage0 else "stage 1 remainder + stages 2–4"
        print(f"| {label} | {ane_count} | {cpu_count} | {ane_cost:.3f} |", flush=True)
