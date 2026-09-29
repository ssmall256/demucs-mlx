"""Synchronized submodule profile of the current HTDemucs hot spots.

Run through ``metalq submit -w``. Timings guide experiments; the inserted
evaluations change scheduling and are not additive production timings.
"""

import argparse
import os
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def evaluate(value):
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            evaluate(item)


class Timed:
    def __init__(self, name, wrapped, totals, counts, shapes):
        self.name = name
        self.wrapped = wrapped
        self.totals = totals
        self.counts = counts
        self.shapes = shapes

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        evaluate(args)
        evaluate(tuple(kwargs.values()))
        if args and isinstance(args[0], mx.array):
            self.shapes[self.name] = tuple(args[0].shape)
        start = time.perf_counter()
        result = self.wrapped(*args, **kwargs)
        evaluate(result)
        self.totals[self.name] += time.perf_counter() - start
        self.counts[self.name] += 1
        return result


parser = argparse.ArgumentParser()
parser.add_argument("--detail", choices=("transformer", "dconv"), default="transformer")
args = parser.parse_args()
os.environ["DEMUCS_MLX_COMPILE_DCONV"] = "0"

totals = defaultdict(float)
counts = defaultdict(int)
shapes = {}
separator = Separator(seed=481)
model = separator.model.models[0]
rng = np.random.default_rng(481)
signal = mx.array(rng.standard_normal((2, 2, 343_980), dtype=np.float32) * 0.1)
mx.eval(signal)

for _ in range(2):
    mx.eval(model(signal))

if args.detail == "transformer":
    transformer = model.crosstransformer
    for branch in ("layers", "layers_t"):
        layers = getattr(transformer, branch)
        for index, layer in enumerate(layers):
            prefix = f"{branch}.{index}"
            for member in (
                "attn",
                "cross_attn",
                "linear1",
                "linear2",
                "norm1",
                "norm2",
                "norm3",
            ):
                if hasattr(layer, member):
                    setattr(
                        layer,
                        member,
                        Timed(
                            f"{prefix}.{member}",
                            getattr(layer, member),
                            totals,
                            counts,
                            shapes,
                        ),
                    )
            layers[index] = Timed(prefix, layer, totals, counts, shapes)

for branch, index in (
    ("encoder", 0),
    ("tencoder", 0),
    ("decoder", 3),
    ("tdecoder", 3),
):
    layers = getattr(model, branch)
    layer = layers[index]
    prefix = f"{branch}.{index}"
    if args.detail == "dconv":
        for block_index, block in enumerate(layer.dconv.layers):
            for module_index, module in enumerate(block.layers):
                block.layers[module_index] = Timed(
                    f"{prefix}.dconv.block{block_index}.{module_index}",
                    module,
                    totals,
                    counts,
                    shapes,
                )
    for member in ("conv", "conv_tr", "rewrite", "dconv", "norm1", "norm2"):
        module = getattr(layer, member, None)
        if module is not None:
            setattr(
                layer,
                member,
                Timed(f"{prefix}.{member}", module, totals, counts, shapes),
            )
    layers[index] = Timed(prefix, layer, totals, counts, shapes)

for _ in range(2):
    mx.eval(model(signal))
totals.clear()
counts.clear()
elapsed = []
for _ in range(3):
    start = time.perf_counter()
    mx.eval(model(signal))
    elapsed.append(time.perf_counter() - start)

print(f"## Synchronized HTDemucs {args.detail} profile", flush=True)
print("**Input:** batch 2, 343,980 samples per channel; 3 measured calls", flush=True)
print(f"**Instrumented median:** {np.median(elapsed) * 1000:.2f} ms", flush=True)
print("| Component | Mean per segment | Calls per segment | Input shape |", flush=True)
print("|---|---:|---:|---|", flush=True)
for name, total in sorted(totals.items(), key=lambda item: -item[1]):
    print(
        f"| `{name}` | {total / 3 * 1000:.2f} ms | {counts[name] / 3:g} | "
        f"`{shapes.get(name, '')}` |",
        flush=True,
    )
