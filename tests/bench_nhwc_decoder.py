"""Benchmark native NHWC spectral decoder vs baseline transposed decoder."""
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_demucs import _dconv_block_forward_nlc
from demucs_mlx.mlx_layers import _PhasedWeightCache


def signal(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)

def forward_dec_nhwc(layer, x_nhwc, skip_nhwc):
    B, Fr, T, C = x_nhwc.shape
    x_cur = x_nhwc + skip_nhwc
    # Rewrite conv2d expects NHWC
    y_rew = layer.rewrite.conv(x_cur)
    if type(layer.norm1).__name__ != 'Identity':
        eps = layer.norm1.eps
        n_flat = mx.fast.layer_norm(y_rew.reshape(B, 1, -1), None, None, eps).reshape(y_rew.shape)
        if getattr(layer.norm1, "affine", True) and layer.norm1.weight is not None:
            scale = layer.norm1.weight.reshape(1, 1, 1, -1)
            y_rew = n_flat * scale + layer.norm1.bias.reshape(1, 1, 1, -1)
        else:
            y_rew = n_flat
    # GLU along last axis
    a, b = mx.split(y_rew, 2, axis=-1)
    y_glu = a * mx.sigmoid(b)
    # DConv in NLC (N = B*Fr, L = T, C)
    y_nlc = y_glu.reshape(-1, T, C)
    for block in layer.dconv.layers:
        y_nlc = y_nlc + _dconv_block_forward_nlc(block, y_nlc)
    y_nhwc = y_nlc.reshape(B, Fr, T, C)
    # Phased deconvolution
    conv = layer.conv_tr.conv
    source = conv.weight
    cache = layer.conv_tr._phased_cache
    if cache is None or cache.source is not source:
        phase_weights = [
            mx.stack([source[:, phase + 4, 0, :], source[:, phase, 0, :]], axis=1)
            for phase in range(4)
        ]
        weight = mx.concatenate(phase_weights, axis=0).reshape(-1, 2, 1, source.shape[-1])
        cache = _PhasedWeightCache(source, weight)
        layer.conv_tr._phased_cache = cache
    padded = mx.pad(y_nhwc, [(0, 0), (1, 1), (0, 0), (0, 0)])
    phases = mx.conv2d(padded, cache.weight)
    phased = phases.reshape(B, Fr + 1, T, 4, source.shape[0])
    joined = phased.transpose(0, 1, 3, 2, 4).reshape(B, 4 * (Fr + 1), T, source.shape[0])
    if "bias" in conv:
        joined = joined + conv.bias
    if type(layer.norm2).__name__ != 'Identity':
        flat = joined.reshape(B, 1, -1)
        joined = mx.fast.layer_norm(flat, None, None, layer.norm2.eps).reshape(joined.shape)
        if getattr(layer.norm2, "affine", True) and layer.norm2.weight is not None:
            scale = layer.norm2.weight.reshape(1, 1, 1, -1)
            joined = joined * scale + layer.norm2.bias.reshape(1, 1, 1, -1)
    if layer.pad:
        joined = joined[:, layer.pad:-layer.pad, :, :]
    if not layer.last:
        joined = nn.gelu(joined)
    return joined, y_nhwc

def main():
    print("=" * 78)
    print("HEADROOM 2: NATIVE CHANNELS-LAST (NHWC) SPECTRAL DECODER EXPERIMENT")
    print("=" * 78)
    
    sep = Separator(seed=481, batch_size=8)
    m = sep.model.models[0]
    
    # 1. Microbenchmark on all 4 spectral decoder layers
    print("\n1. Per-Layer Isolated Microbenchmark (Batch Size 8):")
    shapes = [
        (8, 384, 8, 336),
        (8, 192, 32, 336),
        (8, 96, 128, 336),
        (8, 48, 512, 336),
    ]
    
    total_ref_ms = 0.0
    total_nhwc_ms = 0.0
    
    for idx, (layer, shape) in enumerate(zip(m.decoder, shapes)):
        B, C, Fr, T = shape
        rng = np.random.default_rng(100 + idx)
        x_np = rng.standard_normal(shape).astype(np.float32)
        skip_np = rng.standard_normal(shape).astype(np.float32)
        x = mx.array(x_np)
        skip = mx.array(skip_np)
        mx.eval(x, skip)
        
        # Warmup reference
        for _ in range(3):
            z_ref, y_ref = layer(x, skip, None)
            mx.eval(z_ref, y_ref)
        
        times_ref = []
        for _ in range(15):
            t0 = time.perf_counter()
            z_ref, y_ref = layer(x, skip, None)
            mx.eval(z_ref, y_ref)
            times_ref.append((time.perf_counter() - t0) * 1000)
        
        # Warmup NHWC
        x_nhwc = x.transpose(0, 2, 3, 1)
        skip_nhwc = skip.transpose(0, 2, 3, 1)
        mx.eval(x_nhwc, skip_nhwc)
        
        for _ in range(3):
            z_nhwc, y_nhwc = forward_dec_nhwc(layer, x_nhwc, skip_nhwc)
            mx.eval(z_nhwc, y_nhwc)
            
        times_nhwc = []
        for _ in range(15):
            t0 = time.perf_counter()
            z_nhwc, y_nhwc = forward_dec_nhwc(layer, x_nhwc, skip_nhwc)
            mx.eval(z_nhwc, y_nhwc)
            times_nhwc.append((time.perf_counter() - t0) * 1000)
            
        diff_z = float(mx.max(mx.abs(z_nhwc.transpose(0, 3, 1, 2) - z_ref)))
        diff_y = float(mx.max(mx.abs(y_nhwc.transpose(0, 3, 1, 2) - y_ref)))
        
        med_ref = np.median(times_ref)
        med_nhwc = np.median(times_nhwc)
        total_ref_ms += med_ref
        total_nhwc_ms += med_nhwc
        
        print(f"  Layer {idx} (C={C:3d}, Fr={Fr:3d}, T={T:3d}):")
        print(
            f"    Reference NCHW: {med_ref:6.2f} ms | NHWC: {med_nhwc:6.2f} ms | "
            f"Diff z/y: {diff_z:.2e}/{diff_y:.2e} | Speedup: {med_ref/med_nhwc:.2f}x"
        )

    print("-" * 78)
    print(
        f"  Total Spectral Decoder per batch: {total_ref_ms:6.2f} ms -> {total_nhwc_ms:6.2f} ms "
        f"({total_ref_ms/total_nhwc_ms:.2f}x speedup)"
    )
    print(
        f"  Projected savings across 120s (3 batches): {(total_ref_ms - total_nhwc_ms) * 3:.2f} "
        "ms"
    )

if __name__ == "__main__":
    main()
