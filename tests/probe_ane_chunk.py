"""Probe Core ML placement for a shorter four-stage waveform encoder.

Run through metalq. The fixed input has already undergone global normalization.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch

from demucs_mlx.ane import _device_placement
from demucs_mlx.secure_demucs import get_restricted_demucs_model


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.tencoder

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return tuple(outputs)


length = int(sys.argv[1]) if len(sys.argv) > 1 else 16_384
batch = 2
if length % 256:
    raise ValueError("Length must be divisible by 256 for this probe")
model = get_restricted_demucs_model("htdemucs").model.models[0].eval()
traced = torch.jit.trace(
    Encoder(model).eval(), torch.zeros((batch, 2, length)), check_trace=False
)
with TemporaryDirectory() as temporary:
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(batch, 2, length), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    package = Path(temporary) / "chunk.mlpackage"
    ml.save(str(package))
    compiled = ct.models.utils.compile_model(str(package))
    print(f"## Four-stage waveform encoder placement at length {length}")
    print(_device_placement(Path(compiled)))
