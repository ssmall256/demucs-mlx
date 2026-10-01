"""Empirical evaluation of fused GLU, GroupNorm+GELU, and GroupNorm+GLU custom Metal kernels.

Investigates:
1. Microbenchmark: GLU (fused_glu vs native mx.split + sigmoid vs @mx.compile).
2. Microbenchmark: GroupNorm + GELU (fused Metal kernel vs _group_norm_via_layer_norm + nn.gelu).
3. Microbenchmark: GroupNorm + GLU (fused Metal kernel vs _group_norm_via_layer_norm + GLU).
4. End-to-end 120s separation with DEMUCS_MLX_USE_FUSED_GN_GLU = 0 vs 1.
"""
import os
import time
import Foundation
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from demucs_mlx.metal_kernels import fused_glu, fused_groupnorm_gelu, fused_groupnorm_glu, HAS_METAL
from demucs_mlx.mlx_layers import _group_norm_via_layer_norm, GLUNCL
from demucs_mlx.api import Separator

THERMAL_NAMES = ["nominal", "fair", "serious", "critical"]

def get_thermal_state() -> tuple[int, str]:
    state = int(Foundation.NSProcessInfo.processInfo().thermalState())
    label = THERMAL_NAMES[state] if state < len(THERMAL_NAMES) else f"unknown({state})"
    return state, label

def wait_for_cool_silicon(min_cooldown: float = 3.0) -> float:
    start = time.perf_counter()
    state, _ = get_thermal_state()
    while state != 0:
        time.sleep(1.0)
        state, _ = get_thermal_state()
    time.sleep(min_cooldown)
    return time.perf_counter() - start

def bench_glu():
    print("\n" + "=" * 78)
    print("1. GLU MICROBENCHMARK (fused_glu vs MLX split+sigmoid vs @mx.compile)")
    print("=" * 78)

    # Typical HTDemucs shapes: (B, 2*C, L) and (B, 2*C, Fr, T)
    shapes = [
        ("NCL (DConv waveform)", (8, 96, 344), 1),
        ("NCL (DConv mid)", (8, 192, 172), 1),
        ("NCL (DConv deep)", (8, 384, 86), 1),
        ("NCHW (Spectral mid)", (8, 96, 256, 43), 1),
        ("NCL (Encoder tail)", (8, 96, 85995), 1),
    ]

    @mx.compile
    def compiled_glu(x, axis):
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    def native_glu(x, axis):
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    for desc, shape, axis in shapes:
        x = mx.random.normal(shape).astype(mx.float32)
        mx.eval(x)

        # Warmup
        _ = native_glu(x, axis)
        mx.eval(_)
        _ = compiled_glu(x, axis)
        mx.eval(_)
        _ = fused_glu(x, axis)
        mx.eval(_)

        # Measure Native
        t_native = []
        for _ in range(20):
            t0 = time.perf_counter()
            out_nat = native_glu(x, axis)
            mx.eval(out_nat)
            t_native.append((time.perf_counter() - t0) * 1000)

        # Measure Compiled
        t_comp = []
        for _ in range(20):
            t0 = time.perf_counter()
            out_comp = compiled_glu(x, axis)
            mx.eval(out_comp)
            t_comp.append((time.perf_counter() - t0) * 1000)

        # Measure Fused Metal
        t_fused = []
        for _ in range(20):
            t0 = time.perf_counter()
            out_fused = fused_glu(x, axis)
            mx.eval(out_fused)
            t_fused.append((time.perf_counter() - t0) * 1000)

        med_nat = np.median(t_native)
        med_comp = np.median(t_comp)
        med_fused = np.median(t_fused)

        err_fused = float(mx.max(mx.abs(out_nat - out_fused)))
        err_comp = float(mx.max(mx.abs(out_nat - out_comp)))

        print(f"\nShape {shape} - {desc}:")
        print(f"  Native (split+sigmoid): {med_nat:6.3f} ms")
        print(f"  Compiled (@mx.compile):  {med_comp:6.3f} ms ({med_nat/med_comp:.2f}x vs native)")
        print(f"  Fused Metal (fused_glu): {med_fused:6.3f} ms ({med_nat/med_fused:.2f}x vs native, {med_comp/med_fused:.2f}x vs comp)")
        print(f"  Max Error vs Native: Fused={err_fused:.2e}, Comp={err_comp:.2e}")

def bench_groupnorm_gelu():
    print("\n" + "=" * 78)
    print("2. GROUPNORM + GELU MICROBENCHMARK (fused_groupnorm_gelu vs native layer_norm)")
    print("=" * 78)

    # Real HTDemucs shapes
    # (B, C, L) with num_groups = 1 or 4
    shapes = [
        ("Freq DConv small", (8, 6, 336), 1),
        ("Freq DConv mid", (8, 24, 336), 1),
        ("Freq DConv large", (8, 48, 336), 1),
        ("Waveform DConv", (8, 48, 1344), 1),
        ("Waveform Encoder c1", (8, 48, 85995), 1),
        ("Spectral NCHW", (8, 48, 512, 10), 4),
    ]

    for desc, shape, G in shapes:
        x = mx.random.normal(shape).astype(mx.float32)
        C = shape[1]
        weight = mx.random.normal((C,)).astype(mx.float32)
        bias = mx.random.normal((C,)).astype(mx.float32)
        eps = 1e-5
        mx.eval(x, weight, bias)

        # Unfused: _group_norm_via_layer_norm + nn.gelu
        def run_unfused():
            normed = _group_norm_via_layer_norm(x, G, eps, weight, bias)
            return nn.gelu(normed)

        # Fused Metal
        def run_fused():
            return fused_groupnorm_gelu(x, weight, bias, G, eps)

        # Warmup
        _ = run_unfused()
        mx.eval(_)
        _ = run_fused()
        mx.eval(_)

        t_unfused = []
        for _ in range(20):
            t0 = time.perf_counter()
            out_unf = run_unfused()
            mx.eval(out_unf)
            t_unfused.append((time.perf_counter() - t0) * 1000)

        t_fused = []
        for _ in range(20):
            t0 = time.perf_counter()
            out_fused = run_fused()
            mx.eval(out_fused)
            t_fused.append((time.perf_counter() - t0) * 1000)

        med_unf = np.median(t_unfused)
        med_fused = np.median(t_fused)
        err = float(mx.max(mx.abs(out_unf - out_fused)))
        rel_err = float(mx.max(mx.abs(out_unf - out_fused)) / mx.max(mx.abs(out_unf)))

        print(f"\nShape {shape}, G={G} - {desc}:")
        print(f"  Unfused (mx.fast.layer_norm + gelu): {med_unf:6.3f} ms")
        print(f"  Fused Metal (fused_groupnorm_gelu):  {med_fused:6.3f} ms ({med_unf/med_fused:.2f}x speedup)")
        print(f"  Max Abs Diff: {err:.2e} | Rel Diff: {rel_err:.2e}")

def bench_end_to_end_switch():
    print("\n" + "=" * 78)
    print("3. END-TO-END 120s AUDIO SEPARATION: DEMUCS_MLX_USE_FUSED_GN_GLU = 0 vs 1")
    print("=" * 78)

    def signal(seconds: int) -> np.ndarray:
        rng = np.random.default_rng(481 + seconds)
        n = 44_100 * seconds
        t = np.arange(n, dtype=np.float32) / 44_100
        tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
        return (0.05 * tones[None, :] + noise).astype(np.float32)

    audio = signal(120)

    for val in ("0", "1"):
        os.environ["DEMUCS_MLX_USE_FUSED_GN_GLU"] = val
        mode_label = "UNFUSED (Default mx.fast.layer_norm)" if val == "0" else "FUSED (Custom Metal GN+GELU/GLU)"
        print(f"\n--- Testing DEMUCS_MLX_USE_FUSED_GN_GLU={val} [{mode_label}] ---")

        wait_for_cool_silicon(min_cooldown=5.0)
        sep = Separator(seed=481, batch_size=8)
        # Warmup
        sep.separate_tensor(signal(30))
        mx.clear_cache()
        wait_for_cool_silicon(min_cooldown=4.0)

        runs = []
        for r in range(3):
            wait_for_cool_silicon(min_cooldown=4.0)
            t_in_s, t_in_l = get_thermal_state()
            t0 = time.perf_counter()
            _, stems = sep.separate_tensor(audio)
            elapsed = time.perf_counter() - t0
            t_out_s, t_out_l = get_thermal_state()
            runs.append(elapsed)
            print(f"  Run {r+1}: {elapsed:.3f}s (Audio/Wall: {120/elapsed:.1f}x) [{t_in_l} -> {t_out_l}]")

        med = np.median(runs)
        print(f"  => Median: {med:.3f}s | Audio/Wall Rate: {120/med:.1f}x RTFx")

if __name__ == "__main__":
    bench_glu()
    bench_groupnorm_gelu()
    bench_end_to_end_switch()
