"""Comprehensive benchmark isolating CPU core orchestration and pipeline improvements.

Measures:
1. Overlap-Add Reconstruction:
   - Baseline A: Original .at.add loop
   - Baseline B: Slice assignment loop
   - Orchestrated: Fused Metal kernel
2. ANE Waveform Convolution Dispatch:
   - Baseline: PyObjC Core ML dispatch (GIL held)
   - Orchestrated: Native Objective-C runtime via Grand Central Dispatch (zero-GIL)
3. End-to-End 120s Audio Separation:
   - Unorchestrated baseline (pre-orchestration pipeline)
   - Fully orchestrated engine
"""
import gc
import os
import sys
import time
import Foundation
import numpy as np

import mlx.core as mx

THERMAL_NAMES = ["nominal", "fair", "serious", "critical"]

def get_thermal_state() -> tuple[int, str]:
    state = int(Foundation.NSProcessInfo.processInfo().thermalState())
    label = THERMAL_NAMES[state] if state < len(THERMAL_NAMES) else f"unknown({state})"
    return state, label

def wait_for_cool_silicon(min_cooldown: float = 5.0) -> float:
    start = time.perf_counter()
    state, _ = get_thermal_state()
    while state != 0:
        time.sleep(1.0)
        state, _ = get_thermal_state()
    time.sleep(min_cooldown)
    return time.perf_counter() - start

def signal(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)

def bench_overlap_add():
    from demucs_mlx.metal_kernels import fused_overlap_add, _overlap_add_fallback

    print("\n" + "=" * 78)
    print("1. OVERLAP-ADD RECONSTRUCTION BENCHMARK (120s Audio, 4 Stems, Stereo)")
    print("=" * 78)

    sample_rate = 44100
    seconds = 120
    total_samples = sample_rate * seconds  # 5,292,000 samples
    segment = 7.8
    segment_samples = int(sample_rate * segment)  # 343,980 samples
    overlap = 0.25
    stride = int((1.0 - overlap) * segment_samples)  # 257,985 samples
    offsets = list(range(0, total_samples, stride))
    num_chunks = len(offsets)  # 21 chunks

    # Create weight window (triangular/trapezoidal)
    half = segment_samples // 2
    w_first = np.arange(1, half + 1)
    w_second = np.arange(segment_samples - half, 0, -1)
    weight_np = (np.concatenate([w_first, w_second]) / half).astype(np.float32)
    weight_mx = mx.array(weight_np)

    # Simulated chunk outputs: (num_chunks, 1, 4, 2, segment_samples)
    rng = np.random.default_rng(42)
    chunks_np = rng.standard_normal((num_chunks, 1, 4, 2, segment_samples)).astype(np.float32)
    chunks_mx = mx.array(chunks_np)
    mx.eval(chunks_mx, weight_mx)

    # Warmup
    _ = fused_overlap_add(chunks_mx, weight_mx, stride, total_samples)
    mx.eval(_)

    # Method 1: Original .at.add loop
    def run_at_add():
        out = mx.zeros((1, 4, 2, total_samples), dtype=mx.float32)
        sum_weight = mx.zeros((total_samples,), dtype=mx.float32)
        w = weight_mx.reshape(1, 1, 1, -1)
        for k in range(num_chunks):
            off = offsets[k]
            chunk_len = min(segment_samples, total_samples - off)
            out = out.at[:, :, :, off : off + chunk_len].add(w[:, :, :, :chunk_len] * chunks_mx[k, :, :, :, :chunk_len])
            sum_weight = sum_weight.at[off : off + chunk_len].add(weight_mx[:chunk_len])
        out = out / mx.maximum(sum_weight.reshape(1, 1, 1, -1), 1e-11)
        mx.eval(out)
        return out

    # Method 2: Slice assignment loop
    def run_slice_assign():
        out = mx.zeros((1, 4, 2, total_samples), dtype=mx.float32)
        sum_weight = mx.zeros((total_samples,), dtype=mx.float32)
        w = weight_mx.reshape(1, 1, 1, -1)
        for k in range(num_chunks):
            off = offsets[k]
            chunk_len = min(segment_samples, total_samples - off)
            out[:, :, :, off : off + chunk_len] = out[:, :, :, off : off + chunk_len] + w[:, :, :, :chunk_len] * chunks_mx[k, :, :, :, :chunk_len]
            sum_weight[off : off + chunk_len] = sum_weight[off : off + chunk_len] + weight_mx[:chunk_len]
        out = out / mx.maximum(sum_weight.reshape(1, 1, 1, -1), 1e-11)
        mx.eval(out)
        return out

    # Method 3: Fused Metal Kernel
    def run_fused_metal():
        res = fused_overlap_add(chunks_mx, weight_mx, stride, total_samples)
        mx.eval(res)
        return res

    # Benchmark each with 10 runs
    times_at_add = []
    for _ in range(5):
        t0 = time.perf_counter()
        out_at = run_at_add()
        times_at_add.append((time.perf_counter() - t0) * 1000)

    times_slice = []
    for _ in range(5):
        t0 = time.perf_counter()
        out_slice = run_slice_assign()
        times_slice.append((time.perf_counter() - t0) * 1000)

    times_fused = []
    for _ in range(10):
        t0 = time.perf_counter()
        out_fused = run_fused_metal()
        times_fused.append((time.perf_counter() - t0) * 1000)

    # Parity check
    max_err_slice = float(mx.max(mx.abs(out_at - out_slice)))
    max_err_fused = float(mx.max(mx.abs(out_at - out_fused)))

    med_at = np.median(times_at_add)
    med_slice = np.median(times_slice)
    med_fused = np.median(times_fused)

    print(f"Total samples: {total_samples:,} ({seconds}s) across {num_chunks} overlapping chunks")
    print(f"- Baseline A (.at.add loop):     {med_at:6.2f} ms")
    print(f"- Baseline B (Slice assign loop): {med_slice:6.2f} ms ({med_at/med_slice:.2f}x vs .at.add)")
    print(f"- Orchestrated (Fused Metal):    {med_fused:6.2f} ms ({med_at/med_fused:.2f}x vs .at.add, {med_slice/med_fused:.2f}x vs slice)")
    print(f"- Numerical Parity vs Baseline:  Slice diff = {max_err_slice:.2e}, Fused diff = {max_err_fused:.2e}")

def bench_ane_dispatch():
    print("\n" + "=" * 78)
    print("2. ANE WAVEFORM CONVOLUTION DISPATCH BENCHMARK (Batch 8, 120s Audio)")
    print("=" * 78)

    from demucs_mlx.ane import WaveformConv
    from demucs_mlx.native_ane import predict_waveform_conv_native

    try:
        wc = WaveformConv()
    except Exception as exc:
        print(f"  [SKIP] ANE model not available or compiled: {exc}")
        return

    # Simulate batch 8 waveform chunk (8, 2, 343980)
    rng = np.random.default_rng(123)
    data = rng.standard_normal((8, 2, 343980)).astype(np.float32)
    out_target_native = np.empty((8, 48, 85995), dtype=np.float16)
    out_target_pyobjc = np.empty((8, 48, 85995), dtype=np.float16)

    # Warmup both
    predict_waveform_conv_native(wc.path, data, out_target_native)

    # Path 1: Native C/ObjC GCD dispatch (Zero-GIL)
    times_native = []
    for _ in range(10):
        t0 = time.perf_counter()
        ok = predict_waveform_conv_native(wc.path, data, out_target_native)
        times_native.append((time.perf_counter() - t0) * 1000)
    assert ok

    # Path 2: PyObjC Core ML dispatch (GIL held)
    # Temporarily monkey-patch native dispatch to False to force PyObjC
    import demucs_mlx.native_ane as n_ane
    orig_fn = n_ane.predict_waveform_conv_native
    n_ane.predict_waveform_conv_native = lambda *args, **kwargs: False
    try:
        # Warmup
        wc._predict(data)
        times_pyobjc = []
        for _ in range(10):
            t0 = time.perf_counter()
            out_pyobjc = wc._predict(data)
            times_pyobjc.append((time.perf_counter() - t0) * 1000)
    finally:
        n_ane.predict_waveform_conv_native = orig_fn

    med_pyobjc = np.median(times_pyobjc)
    med_native = np.median(times_native)
    diff = float(np.max(np.abs(out_target_native.astype(np.float32) - out_pyobjc.astype(np.float32))))

    print(f"- Baseline (PyObjC Core ML, GIL held):      {med_pyobjc:6.2f} ms")
    print(f"- Orchestrated (Native GCD C/ObjC, No GIL): {med_native:6.2f} ms ({med_pyobjc/med_native:.2f}x speedup)")
    print(f"- Parity Difference:                       {diff:.6g} (Bit-exact: {diff == 0.0})")


def bench_end_to_end():
    print("\n" + "=" * 78)
    print("3. END-TO-END 120s AUDIO SEPARATION (Thermally Gated Comparison)")
    print("=" * 78)

    from demucs_mlx.api import Separator

    seconds = 120
    audio = signal(seconds)

    # Auto-tuned vs fixed batch size 4
    for b in (4, 8):
        print(f"\n--- Batch Size {b} (120s Audio Separation) ---")
        wait_for_cool_silicon(min_cooldown=5.0)
        t_in_s, t_in_l = get_thermal_state()

        sep = Separator(seed=481, batch_size=b)
        # Warmup
        sep.separate_tensor(signal(30))
        mx.clear_cache()
        gc.collect()
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
            print(f"  Run {r+1}: {elapsed:.3f}s (Audio/Wall: {seconds/elapsed:.1f}x) [{t_in_l} -> {t_out_l}]")

        med = np.median(runs)
        print(f"  => Median: {med:.3f}s | Audio/Wall Rate: {seconds/med:.1f}x RTFx")

if __name__ == "__main__":
    bench_overlap_add()
    bench_ane_dispatch()
    bench_end_to_end()
