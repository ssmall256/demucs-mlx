"""Time the deterministic HTDemucs waveform positional embedding."""

import statistics
import time

import mlx.core as mx

from demucs_mlx.mlx_transformer import create_sin_embedding


def sample(function):
    start = time.perf_counter()
    mx.eval(function())
    return (time.perf_counter() - start) * 1000


for _ in range(2):
    sample(lambda: create_sin_embedding(1344, 512))
times = [sample(lambda: create_sin_embedding(1344, 512)) for _ in range(9)]
print("## HTDemucs waveform position encoding")
print("**Shape:** 1,344 positions × 512 channels")
print(f"**Regeneration median:** {statistics.median(times):.3f} ms per segment")
