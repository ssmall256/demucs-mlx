"""Isolated actual-model decoder rewrite probe; run through MetalQ."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from demucs_mlx.mlx_convert import load_mlx_model_from_safetensors
from demucs_mlx.mlx_layers import GLUNCL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    mx.random.seed(481)
    model = load_mlx_model_from_safetensors("htdemucs", cache_dir=str(a.cache))
    model = getattr(model, "models", [model])[0]
    layer = model.decoder[3]
    assert layer._gated_rewrite
    x = mx.random.normal((8, 48, 512, 336))
    mx.eval(x, layer.parameters())

    def before(v):
        return GLUNCL(axis=1)(layer.norm1(layer.rewrite(v)))

    after = layer._rewrite_glu
    np.testing.assert_array_equal(np.asarray(after(x)), np.asarray(before(x)))
    functions = [mx.compile(before), mx.compile(after)]
    for f in functions:
        for _ in range(3):
            mx.eval(f(x))
    np.testing.assert_array_equal(np.asarray(functions[1](x)), np.asarray(functions[0](x)))
    times = [[], []]
    for i in range(8):
        for arm in [0, 1] if i % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            mx.eval(functions[arm](x))
            times[arm].append(time.perf_counter() - start)
    medians = [statistics.median(values) for values in times]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "job_id": os.environ.get("METALQ_JOB_ID"),
                "shape": list(x.shape),
                "times": times,
                "medians": medians,
                "bit_identical": True,
                "boundary": "isolated compiled convolution+bias+GLU",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "## Decoder rewrite\n\n| Arm | Median |\n|---|---:|\n"
        f"| Before | **{medians[0] * 1000:.3f} ms** |\n"
        f"| After | **{medians[1] * 1000:.3f} ms** |\n\n"
        f"> ✅ Exact eager/compiled output; "
        f"{100 * (1 - medians[1] / medians[0]):.2f}% lower isolated latency."
    )


if __name__ == "__main__":
    main()
