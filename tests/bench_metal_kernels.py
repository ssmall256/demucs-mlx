"""
Microbenchmarks for custom Metal kernels vs reference MLX implementations.

Measures wall-clock time for each kernel at representative shapes from
HTDemucs / HDemucs / Demucs inference. Reports speedup factors.

Usage:
    python tests/bench_metal_kernels.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn


def _bench(fn, warmup=5, iters=50):
    """Benchmark a function. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def bench_glu():
    from demucs_mlx.metal_kernels import fused_glu

    print("=" * 70)
    print("Fused GLU Benchmark")
    print("=" * 70)

    shapes = [
        ((1, 96, 344),  1, "HDemucs enc-1"),
        ((1, 192, 172), 1, "HDemucs enc-2"),
        ((1, 384, 86),  1, "HDemucs enc-3"),
        ((1, 768, 43),  1, "HDemucs enc-4"),
        ((2, 384, 200), 1, "Batch=2 large"),
    ]

    for shape, axis, desc in shapes:
        x = mx.random.normal(shape) * 2.0
        mx.eval(x)

        def ref():
            a, b = mx.split(x, 2, axis=axis)
            r = a * mx.sigmoid(b)
            mx.eval(r)

        def fused():
            r = fused_glu(x, axis=axis)
            mx.eval(r)

        t_ref = _bench(ref)
        t_fused = _bench(fused)
        speedup = t_ref / t_fused if t_fused > 0 else float('inf')

        print(f"  {desc:25s}  ref={t_ref:7.3f}ms  fused={t_fused:7.3f}ms  speedup={speedup:.2f}x")

    print()


def bench_groupnorm_gelu():
    from demucs_mlx.metal_kernels import fused_groupnorm_gelu

    print("=" * 70)
    print("Fused GroupNorm+GELU Benchmark")
    print("=" * 70)

    shapes = [
        ((1, 48, 688),      4, "HDemucs enc-0 NCL"),
        ((1, 96, 344),      4, "HDemucs enc-1 NCL"),
        ((1, 192, 172),     4, "HDemucs enc-2 NCL"),
        ((1, 384, 86),      4, "HDemucs enc-3 NCL"),
        ((1, 48, 512, 43),  4, "HTDemucs enc-0 NCHW"),
        ((1, 96, 256, 22),  4, "HTDemucs enc-1 NCHW"),
        ((1, 192, 128, 11), 4, "HTDemucs enc-2 NCHW"),
        ((1, 384, 64, 6),   4, "HTDemucs enc-3 NCHW"),
    ]

    for shape, num_groups, desc in shapes:
        C = shape[1]
        x = mx.random.normal(shape).astype(mx.float32)
        weight = mx.random.normal((C,)).astype(mx.float32)
        bias = mx.random.normal((C,)).astype(mx.float32)
        eps = 1e-5
        mx.eval(x, weight, bias)

        # Reference: manual GroupNorm + GELU
        def ref():
            B = shape[0]
            G = num_groups
            x_r = x.reshape(B, G, C // G, *x.shape[2:])
            axes = tuple(range(2, x_r.ndim))
            mean = x_r.mean(axis=axes, keepdims=True)
            var = ((x_r - mean) ** 2).mean(axis=axes, keepdims=True)
            x_norm = (x_r - mean) * mx.rsqrt(var + eps)
            x_out = x_norm.reshape(x.shape)
            w_shape = [1, C] + [1] * (x.ndim - 2)
            r = x_out * weight.reshape(w_shape) + bias.reshape(w_shape)
            r = nn.gelu(r)
            mx.eval(r)

        def fused():
            r = fused_groupnorm_gelu(x, weight, bias, num_groups, eps)
            mx.eval(r)

        t_ref = _bench(ref)
        t_fused = _bench(fused)
        speedup = t_ref / t_fused if t_fused > 0 else float('inf')

        print(f"  {desc:30s}  ref={t_ref:7.3f}ms  fused={t_fused:7.3f}ms  speedup={speedup:.2f}x")

    print()


def bench_complex_to_interleaved():
    from demucs_mlx.metal_kernels import fused_complex_to_interleaved

    print("=" * 70)
    print("Fused Complex-to-Interleaved Benchmark")
    print("=" * 70)

    shapes = [
        ((1, 4, 2049, 43),  "HTDemucs typical"),
        ((1, 4, 2049, 172), "HTDemucs long"),
        ((1, 2, 2049, 43),  "HDemucs typical"),
        ((2, 4, 2049, 43),  "Batch=2"),
    ]

    for shape, desc in shapes:
        B, C, Fr, T = shape
        real_part = mx.random.normal(shape)
        imag_part = mx.random.normal(shape)
        z = real_part + 1j * imag_part
        mx.eval(z)

        def ref():
            real = mx.real(z)
            imag = mx.imag(z)
            r = mx.stack([real, imag], axis=2).reshape(B, C * 2, Fr, T)
            mx.eval(r)

        def fused():
            r = fused_complex_to_interleaved(z)
            mx.eval(r)

        t_ref = _bench(ref)
        t_fused = _bench(fused)
        speedup = t_ref / t_fused if t_fused > 0 else float('inf')

        print(f"  {desc:25s}  ref={t_ref:7.3f}ms  fused={t_fused:7.3f}ms  speedup={speedup:.2f}x")

    print()


def bench_transformer_norm():
    """Benchmark the cached norm optimization in transformer layers."""
    from demucs_mlx.mlx_transformer import TransformerEncoderLayer

    print("=" * 70)
    print("Transformer Norm Caching (already integrated — functional test)")
    print("=" * 70)

    # This is already integrated, so we just measure single-layer throughput
    d_model = 384
    layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=1536,
        dropout=0.0,
        activation=lambda x: nn.gelu(x),
        norm_first=True,
        layer_scale=True,
    )
    layer.eval()

    for seq_len, desc in [(100, "Seq=100"), (343, "Seq=343 (typical)"), (688, "Seq=688")]:
        x = mx.random.normal((1, seq_len, d_model))
        mx.eval(x)

        def fwd():
            r = layer(x)
            mx.eval(r)

        t = _bench(fwd)
        print(f"  {desc:25s}  time={t:7.3f}ms")

    print()


if __name__ == "__main__":
    print()
    print("Metal Kernel Microbenchmarks")
    print(f"Device: {mx.default_device()}")
    print()

    bench_glu()
    bench_groupnorm_gelu()
    bench_complex_to_interleaved()
    bench_transformer_norm()

    print("Done.")
