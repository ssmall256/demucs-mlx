"""Benchmark custom RMSNorm Metal kernel against MLX built-in.

Compares:
1. Custom metal_kernel RMSNorm (from rmsnorm_kernel.py)
2. mx.fast.rms_norm (MLX built-in, already optimized)
3. Naive Python RMSNorm (pure MLX ops, no custom kernel)

Usage:
    python benchmark_rmsnorm.py
    python benchmark_rmsnorm.py --dtype float16
    python benchmark_rmsnorm.py --warmup 5 --iters 50
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import mlx.core as mx

# Import the custom kernel from sibling module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rmsnorm_kernel import rmsnorm as custom_rmsnorm


def naive_rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """Reference RMSNorm using pure MLX ops (no custom kernel)."""
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    x_normed = x * mx.rsqrt(variance + eps)
    return x_normed * weight


def benchmark_fn(fn, args, warmup: int = 3, iters: int = 20) -> float:
    """Time a function, returning mean seconds per call."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    # Timed
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.3f} ms"


def compute_bandwidth(shape, dtype, elapsed: float) -> float:
    """Compute achieved bandwidth in GB/s."""
    elements = 1
    for d in shape:
        elements *= d
    D = shape[-1]

    bytes_per_elem = 4 if dtype in (mx.float32,) else 2
    # Read x + weight, write output
    total_bytes = elements * bytes_per_elem * 2 + D * bytes_per_elem
    return total_bytes / elapsed / 1e9


def main():
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm kernels")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Data type",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    dtype = mx.float32 if args.dtype == "float32" else mx.float16

    shapes = [
        (1, 128, 4096),
        (1, 512, 4096),
        (1, 1024, 4096),
        (1, 2048, 4096),
        (4, 512, 4096),
        (1, 4096, 4096),
        (1, 512, 2048),
        (1, 512, 8192),
    ]

    print(f"RMSNorm Benchmark — dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print(f"Device: {mx.device_info().get('device_name', 'Apple Silicon')}")
    print()

    header = f"{'Shape':>22s} | {'Custom':>10s} | {'Built-in':>10s} | {'Naive':>10s} | {'vs Built-in':>11s} | {'BW (GB/s)':>10s} | {'Correct':>7s}"
    print(header)
    print("-" * len(header))

    for shape in shapes:
        D = shape[-1]
        x = mx.random.normal(shape).astype(dtype)
        w = mx.random.normal((D,)).astype(dtype)

        # Correctness check
        ref = mx.fast.rms_norm(x, w, eps=1e-5)
        custom_out = custom_rmsnorm(x, w, eps=1e-5)
        mx.eval(ref, custom_out)
        max_diff = mx.max(mx.abs(ref - custom_out)).item()
        atol = 1e-3 if dtype == mx.float16 else 1e-5
        correct = "OK" if max_diff < atol else f"FAIL({max_diff:.1e})"

        # Benchmark
        t_custom = benchmark_fn(custom_rmsnorm, (x, w), args.warmup, args.iters)
        builtin_fn = lambda x, w: mx.fast.rms_norm(x, w, eps=1e-5)
        t_builtin = benchmark_fn(builtin_fn, (x, w), args.warmup, args.iters)
        t_naive = benchmark_fn(naive_rmsnorm, (x, w), args.warmup, args.iters)

        speedup_vs_builtin = t_builtin / t_custom if t_custom > 0 else float("inf")
        bw = compute_bandwidth(shape, dtype, t_custom)

        print(
            f"{str(shape):>22s} | "
            f"{format_time(t_custom):>10s} | "
            f"{format_time(t_builtin):>10s} | "
            f"{format_time(t_naive):>10s} | "
            f"{speedup_vs_builtin:>10.2f}x | "
            f"{bw:>9.1f} | "
            f"{correct:>7s}"
        )

    print()
    print("Notes:")
    print("  - 'Custom' = metal_kernel RMSNorm from this skill")
    print("  - 'Built-in' = mx.fast.rms_norm (MLX's optimized implementation)")
    print("  - 'Naive' = pure MLX ops (mean + rsqrt + multiply)")
    print("  - 'vs Built-in' > 1.0x means custom kernel is faster")
    print("  - BW = achieved memory bandwidth for the custom kernel")


if __name__ == "__main__":
    main()
