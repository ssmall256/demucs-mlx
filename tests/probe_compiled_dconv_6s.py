"""Check compiled DConv parity for the six-source HTDemucs mode."""

import os

import numpy as np

from demucs_mlx.api import Separator

rng = np.random.default_rng(481)
audio = rng.standard_normal((2, 44_100), dtype=np.float32) * 0.01
separator = Separator(model="htdemucs_6s", shifts=0, seed=481)

os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "0"
_, eager = separator.separate_tensor(audio)
os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "1"
_, compiled = separator.separate_tensor(audio)

assert len(eager) == len(compiled) == 6
for name, reference in eager.items():
    np.testing.assert_array_equal(compiled[name], reference)

print("## Six-source DConv parity")
print("> ✅ All six complete stems matched the eager path exactly.")
