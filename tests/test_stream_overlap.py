"""Diagnose whether multi-stream operations in MLX actually execute concurrently on Metal."""
import time

import mlx.core as mx
import numpy as np

from demucs_mlx.api import Separator


def main():
    sep = Separator(seed=481, batch_size=8)
    m = sep.model.models[0]
    
    rng = np.random.default_rng(481)
    b = mx.array(rng.standard_normal((8, 2, 343980)).astype(np.float32))
    mx.eval(b)

    z = m._spec(b)
    mag = m._magnitude(z)
    mx.eval(z, mag)

    s_side = mx.new_stream(mx.default_device())

    # --- ENCODER TEST ---
    def run_spectral_encoder():
        x = mag
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)
        saved = []
        lengths = []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            x = encode(x, None)
            if idx == 0 and m.freq_emb is not None:
                frs = mx.arange(x.shape[-2], dtype=mx.int32)
                emb = m.freq_emb(frs).transpose(1, 0)[None, :, :, None]
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        return x, saved, lengths

    def run_waveform_encoder():
        with mx.stream(s_side):
            xt = b
            meant = mx.mean(xt, axis=(1, 2), keepdims=True)
            stdt = mx.std(xt, axis=(1, 2), keepdims=True)
            xt = (xt - meant) / (1e-5 + stdt)
            saved_t = []
            lengths_t = []
            for tenc in m.tencoder:
                lengths_t.append(xt.shape[-1])
                xt = tenc(xt)
                saved_t.append(xt)
            return xt, saved_t, lengths_t

    # Warmup
    x, saved, lengths = run_spectral_encoder()
    xt, saved_t, lengths_t = run_waveform_encoder()
    mx.eval(x, xt)

    # 1. Spectral alone
    times_spec = []
    for _ in range(10):
        t0 = time.perf_counter()
        x, _, _ = run_spectral_encoder()
        mx.eval(x)
        times_spec.append((time.perf_counter() - t0) * 1000)

    # 2. Waveform alone
    times_wave = []
    for _ in range(10):
        t0 = time.perf_counter()
        xt, _, _ = run_waveform_encoder()
        mx.eval(xt)
        times_wave.append((time.perf_counter() - t0) * 1000)

    # 3. Both concurrently (dual stream)
    times_concurrent = []
    for _ in range(10):
        t0 = time.perf_counter()
        xt, _, _ = run_waveform_encoder()  # on s_side
        x, _, _ = run_spectral_encoder()   # on default stream
        mx.eval(x, xt)
        times_concurrent.append((time.perf_counter() - t0) * 1000)

    # 4. Both sequentially (single stream)
    times_sequential = []
    for _ in range(10):
        t0 = time.perf_counter()
        xt, _, _ = run_waveform_encoder()
        mx.eval(xt)
        x, _, _ = run_spectral_encoder()
        mx.eval(x)
        times_sequential.append((time.perf_counter() - t0) * 1000)

    t_spec = np.median(times_spec)
    t_wave = np.median(times_wave)
    t_conc = np.median(times_concurrent)
    t_seq = np.median(times_sequential)

    print("=" * 60)
    print("ENCODER CONCURRENCY ANALYSIS (M4 Max)")
    print("=" * 60)
    print(f"Spectral alone (default stream): {t_spec:6.2f} ms")
    print(f"Waveform alone (s_side stream):  {t_wave:6.2f} ms")
    print(f"Sequential (wave then spec):     {t_seq:6.2f} ms (expected: {t_spec + t_wave:.2f} ms)")
    print(
        f"Concurrent (dual stream):        {t_conc:6.2f} ms (ideal: {max(t_spec, t_wave):.2f} "
        "ms)"
    )
    overlap_pct = (t_seq - t_conc) / min(t_spec, t_wave) * 100
    print(f"Effective Overlap:               {overlap_pct:5.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
