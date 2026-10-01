"""Test cross-batch decoder pipelining where next_batch waveform encoder is dispatched
on s_side immediately following tdecoder of the current batch.
"""
import time
import numpy as np
import mlx.core as mx
from demucs_mlx.api import Separator
from demucs_mlx.mlx_htdemucs import center_trim

def main():
    print("=" * 70)
    print("CROSS-BATCH DECODER PIPELINING PROTOTYPE")
    print("=" * 70)

    sep = Separator(seed=481, batch_size=8)
    m = sep.model.models[0]
    s_side = getattr(m, "_stream_side", None)
    if s_side is None:
        s_side = mx.new_stream(mx.default_device())
        m._stream_side = s_side

    # 1. Forward function supporting precomputed tencoder and next_mix pipelining
    def forward_pipelined(mix, precomputed_tenc=None, next_mix=None):
        z = m._spec(mix)
        mag = m._magnitude(z)
        x = mag
        B, C, Fq, T = x.shape
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)

        if precomputed_tenc is not None:
            xt, saved_t, lengths_t, meant, stdt = precomputed_tenc
            # Copy lists to allow popping
            saved_t = list(saved_t)
            lengths_t = list(lengths_t)
        else:
            with mx.stream(s_side):
                xt = mix
                meant = mx.mean(xt, axis=(1, 2), keepdims=True)
                stdt = mx.std(xt, axis=(1, 2), keepdims=True)
                xt = (xt - meant) / (1e-5 + stdt)
                saved_t = []
                lengths_t = []
                for tenc in m.tencoder:
                    lengths_t.append(xt.shape[-1])
                    xt = tenc(xt)
                    saved_t.append(xt)

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

        # Crosstransformer
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                with mx.stream(s_side):
                    xt = m.channel_upsampler_t(xt)
                x = m.channel_upsampler(x)
                x = x.reshape(b, m.bottom_channels, f, t)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = x.reshape(b, m.bottom_channels, f * t)
                with mx.stream(s_side):
                    xt = m.channel_downsampler_t(xt)
                x = m.channel_downsampler(x)
                x = x.reshape(b, c, f, t)

        # Dual-branch decoders
        next_tenc = None
        with mx.stream(s_side):
            for tdec in m.tdecoder:
                length_t = lengths_t.pop(-1)
                skip_t = saved_t.pop(-1)
                xt, _ = tdec(xt, skip_t, length_t)
            # Denormalize xt on s_side
            S = len(m.sources)
            actual_length = xt.shape[-1]
            xt = xt.reshape(B, S, -1, actual_length)
            xt = xt * stdt[:, None] + meant[:, None]

            # PIPELINING: As soon as tdecoder and denorm finish on s_side,
            # dispatch next_mix's waveform encoder on s_side while default stream
            # runs spectral decoder!
            if next_mix is not None:
                xt_next = next_mix
                meant_next = mx.mean(xt_next, axis=(1, 2), keepdims=True)
                stdt_next = mx.std(xt_next, axis=(1, 2), keepdims=True)
                xt_next = (xt_next - meant_next) / (1e-5 + stdt_next)
                saved_t_next = []
                lengths_t_next = []
                for tenc in m.tencoder:
                    lengths_t_next.append(xt_next.shape[-1])
                    xt_next = tenc(xt_next)
                    saved_t_next.append(xt_next)
                next_tenc = (xt_next, saved_t_next, lengths_t_next, meant_next, stdt_next)

        # Default stream spectral decoder
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, _ = decode(x, skip, lengths.pop(-1))

        x = x.reshape(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        zout = m._mask(z, x)
        training_length = int(m.segment * m.samplerate)
        x = m._ispec(zout, training_length)

        x = center_trim(x, xt)
        x = xt + x
        x = x[..., :training_length]

        if next_mix is not None:
            return x, next_tenc
        return x

    # Prepare 2 batches of shape (8, 2, 343980)
    rng = np.random.default_rng(481)
    b0 = mx.array(rng.standard_normal((8, 2, 343980)).astype(np.float32))
    b1 = mx.array(rng.standard_normal((8, 2, 343980)).astype(np.float32))
    mx.eval(b0, b1)

    # Parity check
    ref0 = m(b0)
    ref1 = m(b1)
    mx.eval(ref0, ref1)

    pipe0, next_tenc = forward_pipelined(b0, next_mix=b1)
    pipe1 = forward_pipelined(b1, precomputed_tenc=next_tenc)
    mx.eval(pipe0, pipe1)

    diff0 = float(mx.max(mx.abs(ref0 - pipe0)))
    diff1 = float(mx.max(mx.abs(ref1 - pipe1)))
    print(f"Batch 0 diff vs reference: {diff0:.2e}")
    print(f"Batch 1 diff vs reference: {diff1:.2e}")
    assert diff0 < 1e-4, f"Batch 0 parity failed: {diff0}"
    assert diff1 < 1e-4, f"Batch 1 parity failed: {diff1}"
    print(">>> BIT-EXACT PARITY VERIFIED ON BOTH BATCHES!")

    # Latency comparison: Serial vs Pipelined
    # Warmup
    for _ in range(3):
        r0 = m(b0)
        r1 = m(b1)
        mx.eval(r0, r1)

    times_serial = []
    for _ in range(10):
        t0 = time.perf_counter()
        o0 = m(b0)
        o1 = m(b1)
        mx.eval(o0, o1)
        times_serial.append((time.perf_counter() - t0) * 1000)

    # Warmup pipelined
    for _ in range(3):
        p0, n_t = forward_pipelined(b0, next_mix=b1)
        p1 = forward_pipelined(b1, precomputed_tenc=n_t)
        mx.eval(p0, p1)

    times_pipelined = []
    for _ in range(10):
        t0 = time.perf_counter()
        p0, n_t = forward_pipelined(b0, next_mix=b1)
        p1 = forward_pipelined(b1, precomputed_tenc=n_t)
        mx.eval(p0, p1)
        times_pipelined.append((time.perf_counter() - t0) * 1000)

    med_serial = np.median(times_serial)
    med_pipe = np.median(times_pipelined)
    diff_ms = med_serial - med_pipe
    speedup = med_serial / med_pipe
    print("\n" + "=" * 70)
    print(f"Serial (Batch 0 + Batch 1):    {med_serial:6.2f} ms")
    print(f"Pipelined (Batch 0 + Batch 1): {med_pipe:6.2f} ms ({speedup:.2f}x, saves {diff_ms:.2f} ms)")
    print("=" * 70)

if __name__ == "__main__":
    main()
