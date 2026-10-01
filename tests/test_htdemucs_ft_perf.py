"""Benchmark and verify htdemucs_ft with our optimizations."""
import time
import Foundation
import numpy as np
import mlx.core as mx
from demucs_mlx.api import Separator

def signal(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(481 + seconds)
    n = 44_100 * seconds
    t = np.arange(n, dtype=np.float32) / 44_100
    tones = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal((2, n), dtype=np.float32) * 0.01
    return (0.05 * tones[None, :] + noise).astype(np.float32)

def main():
    print("=" * 70)
    print("HTDEMUCS_FT PERFORMANCE & OPTIMIZATION AUDIT")
    print("=" * 70)

    # 1. Full ensemble (4 models)
    print("\n1. Initializing Separator(model='htdemucs_ft')...")
    t0 = time.perf_counter()
    sep_ft = Separator(model="htdemucs_ft")
    print(f"  Load time: {(time.perf_counter() - t0)*1000:.2f} ms")
    print(f"  Batch size: {sep_ft.batch_size} (auto-tuned)")
    print(f"  Ensemble sub-models: {len(sep_ft.model.models)}")

    # Check attention fusion across all sub-models
    for idx, sub in enumerate(sep_ft.model.models):
        fused_count = 0
        ct = getattr(sub, "crosstransformer", None)
        if ct is not None:
            for layer in getattr(ct, "layers", []) + getattr(ct, "layers_t", []):
                for a in ("attn", "cross_attn"):
                    m = getattr(layer, a, None)
                    if m is not None and getattr(m, "_fused_initialized", False):
                        fused_count += 1
        print(f"  Sub-model {idx}: {fused_count} attention layers pre-fused")

    # Warmup
    print("\n2. Warming up full ensemble...")
    audio_30 = signal(30)
    sep_ft.separate_tensor(audio_30)

    # Benchmark 30s full ensemble
    print("\n3. Benchmarking 30s full ensemble (4 models):")
    times_30 = []
    for _ in range(3):
        t0 = time.perf_counter()
        _, stems = sep_ft.separate_tensor(audio_30)
        elapsed = time.perf_counter() - t0
        times_30.append(elapsed)
        print(f"  30s full: {elapsed:.3f}s ({30/elapsed:.1f}x RTFx)")
    med_30 = np.median(times_30)
    print(f"  => 30s Full Ensemble: {med_30:.3f}s median ({30/med_30:.1f}x RTFx)")

    # 4. Single-stem acceleration on htdemucs_ft
    print("\n4. Benchmarking single-stem acceleration on htdemucs_ft (e.g. vocals):")
    sep_vocals = Separator(model="htdemucs_ft", stem="vocals")
    print(f"  Vocals-only separator: source_index={sep_vocals._source_index}")
    # Warmup
    sep_vocals.separate_tensor(audio_30)
    times_stem = []
    for _ in range(3):
        t0 = time.perf_counter()
        _, stems = sep_vocals.separate_tensor(audio_30)
        elapsed = time.perf_counter() - t0
        times_stem.append(elapsed)
        print(f"  30s vocals-only: {elapsed:.3f}s ({30/elapsed:.1f}x RTFx)")
    med_stem = np.median(times_stem)
    print(f"  => 30s Single-Stem (Vocals): {med_stem:.3f}s median ({30/med_stem:.1f}x RTFx) [{med_30/med_stem:.2f}x speedup over full ensemble!]")

    # 5. Benchmark 120s audio on full ensemble and single-stem
    audio_120 = signal(120)
    print("\n5. Benchmarking 120s audio on single-stem (vocals):")
    t0 = time.perf_counter()
    _, stems = sep_vocals.separate_tensor(audio_120)
    elapsed_120_stem = time.perf_counter() - t0
    print(f"  120s Vocals-Only: {elapsed_120_stem:.3f}s ({120/elapsed_120_stem:.1f}x RTFx)")

    print("\n6. Benchmarking 120s audio on full ensemble (4 models):")
    t0 = time.perf_counter()
    _, stems = sep_ft.separate_tensor(audio_120)
    elapsed_120_full = time.perf_counter() - t0
    print(f"  120s Full Ensemble (4 models): {elapsed_120_full:.3f}s ({120/elapsed_120_full:.1f}x RTFx)")
    print("=" * 70)

if __name__ == "__main__":
    main()
