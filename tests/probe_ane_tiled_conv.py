"""Probe exact tiling of the HTDemucs first waveform convolution."""

from pathlib import Path
from tempfile import TemporaryDirectory

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from demucs_mlx.ane import LENGTH, _device_placement
from demucs_mlx.secure_demucs import get_restricted_demucs_model


class TiledConv(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / (1e-5 + std)
        x = F.pad(x, (2, 2))
        return torch.cat(
            [
                F.conv1d(
                    x[:, :, i * 16_380 : i * 16_380 + 16_384],
                    self.conv.weight,
                    self.conv.bias,
                    stride=4,
                )
                for i in range(21)
            ],
            dim=-1,
        )


original = get_restricted_demucs_model("htdemucs").model.models[0].tencoder[0].conv
module = TiledConv(original).eval()
sample = torch.randn((2, 2, LENGTH))
normalized = (sample - sample.mean(dim=(1, 2), keepdim=True)) / (
    1e-5 + sample.std(dim=(1, 2), keepdim=True, unbiased=False)
)
torch.testing.assert_close(module(sample), original(normalized), rtol=0, atol=2e-6)
traced = torch.jit.trace(module, sample, check_trace=False)
with TemporaryDirectory() as temporary:
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=tuple(sample.shape), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    package = Path(temporary) / "tiled_conv.mlpackage"
    ml.save(str(package))
    compiled = ct.models.utils.compile_model(str(package))
    print("## Exact tiled first-convolution placement")
    print(_device_placement(Path(compiled)))
