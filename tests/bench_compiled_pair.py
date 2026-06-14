"""
Benchmark: compiled_pair() vs eager STFT/iSTFT for Demucs inference.

Compares four approaches:
  1. spectro()/ispectro() — old path (recreates transform each call)
  2. Cached transform — reuse SpectralTransform, call .stft()/.istft() directly
  3. compiled_pair() — pre-compiled graph from mlx-spectro 0.2.2
  4. CachedSpectralPair — integrated class used by Demucs models

Uses realistic HTDemucs parameters: n_fft=4096, hop_length=1024.

Usage:
    python tests/bench_compiled_pair.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from mlx_spectro import get_transform_mlx, resolve_fft_params

from demucs_mlx.spec_mlx import CachedSpectralPair, ispectro, spectro

N_FFT = 4096
HOP_LENGTH = 1024
SR = 44100


def _bench(fn, warmup=5, iters=50):
    """Benchmark a function. Returns median time in ms."""
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
    return times[len(times) // 2]


def bench_roundtrip():
    """Benchmark full STFT->iSTFT roundtrip (most realistic for Demucs)."""
    print("=" * 70)
    print("STFT -> iSTFT Roundtrip Benchmark")
    print("=" * 70)

    B = 2  # stereo, the typical Demucs case

    eff_n_fft, hop, win = resolve_fft_params(N_FFT, HOP_LENGTH, None, 0)
    transform = get_transform_mlx(
        n_fft=eff_n_fft, hop_length=hop, win_length=win,
        window_fn="hann", periodic=True, center=True, normalized=False,
        window=None,
    )

    pair = CachedSpectralPair(n_fft=N_FFT, hop_length=HOP_LENGTH)

    for dur_label, dur_s in [("5s", 5.0), ("10s", 10.0), ("30s", 30.0)]:
        t_len = int(SR * dur_s)
        x2 = mx.random.normal((B, t_len)).astype(mx.float32)
        x3 = x2[None, :, :]  # [1, 2, T] for spectro 3D path
        mx.eval(x2, x3)

        # --- Approach 1: spectro + ispectro (old eager path) ---
        def eager():
            z = spectro(x3, n_fft=N_FFT, hop_length=HOP_LENGTH)
            w = ispectro(z, n_fft=N_FFT, hop_length=HOP_LENGTH, length=t_len)
            mx.eval(w)

        # --- Approach 2: cached transform (no compiled graphs) ---
        def cached():
            z = transform.stft(x2)
            w = transform.istft(z, length=t_len)
            mx.eval(w)

        # --- Approach 3: raw compiled_pair ---
        stft_fn, istft_fn = transform.compiled_pair(
            length=t_len, layout="bfn", warmup_batch=B,
        )

        def compiled():
            z = stft_fn(x2)
            w = istft_fn(z)
            mx.eval(w)

        # --- Approach 4: CachedSpectralPair (integrated, with 3D reshape) ---
        def integrated():
            z = pair.stft(x3)
            w = pair.istft(z, length=t_len)
            mx.eval(w)

        t_eager = _bench(eager)
        t_cached = _bench(cached)
        t_compiled = _bench(compiled)
        t_integrated = _bench(integrated)

        print(f"  {dur_label} audio (B={B}, T={t_len})")
        print(f"    spectro+ispectro:     {t_eager:7.3f}ms  (baseline)")
        print(f"    cached transform:     {t_cached:7.3f}ms  ({t_eager / t_cached:.2f}x)")
        print(f"    compiled_pair (raw):  {t_compiled:7.3f}ms  ({t_eager / t_compiled:.2f}x)")
        print(f"    CachedSpectralPair:   {t_integrated:7.3f}ms  ({t_eager / t_integrated:.2f}x)")
        print()


def bench_repeated_chunks():
    """Benchmark repeated same-size chunks (Demucs segment processing)."""
    print("=" * 70)
    print("Repeated Chunk Benchmark (simulates Demucs segment loop)")
    print("=" * 70)

    B = 2
    segment_length = int(SR * 10.0)  # 10s segments (HTDemucs default)
    n_chunks = 10

    pair = CachedSpectralPair(n_fft=N_FFT, hop_length=HOP_LENGTH)

    chunks = []
    for _ in range(n_chunks):
        x = mx.random.normal((1, B, segment_length)).astype(mx.float32)
        mx.eval(x)
        chunks.append(x)

    # --- Old path: spectro+ispectro per chunk ---
    def eager_loop():
        for x in chunks:
            z = spectro(x, n_fft=N_FFT, hop_length=HOP_LENGTH)
            w = ispectro(z, n_fft=N_FFT, hop_length=HOP_LENGTH, length=segment_length)
            mx.eval(w)

    # --- New path: CachedSpectralPair per chunk ---
    def integrated_loop():
        for x in chunks:
            z = pair.stft(x)
            w = pair.istft(z, length=segment_length)
            mx.eval(w)

    t_eager = _bench(eager_loop, warmup=2, iters=10)
    t_integrated = _bench(integrated_loop, warmup=2, iters=10)

    print(f"  {n_chunks} chunks x 10s (segment_length={segment_length})")
    print(f"    spectro+ispectro:     {t_eager:7.3f}ms  (baseline)")
    print(f"    CachedSpectralPair:   {t_integrated:7.3f}ms  ({t_eager / t_integrated:.2f}x)")
    print()


if __name__ == "__main__":
    print()
    print("compiled_pair() Benchmark — mlx-spectro 0.2.2")
    print(f"Device: {mx.default_device()}")
    print(f"Parameters: n_fft={N_FFT}, hop_length={HOP_LENGTH}, sr={SR}")
    print()

    bench_roundtrip()
    bench_repeated_chunks()

    print("Done.")
