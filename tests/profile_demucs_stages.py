"""Profile each stage of HTDemucs forward pass in isolation and with streams."""
import time
import numpy as np
import mlx.core as mx
from demucs_mlx.api import Separator

def profile():
    sep = Separator(seed=481, batch_size=8)
    m = sep.model.models[0]
    
    # Input batch: 8 segments of length 343980
    rng = np.random.default_rng(481)
    b = mx.array(rng.standard_normal((8, 2, 343980)).astype(np.float32))
    mx.eval(b)

    s_side = getattr(m, "_stream_side", None)
    if s_side is None:
        s_side = mx.new_stream(mx.default_device())
        m._stream_side = s_side

    # Warmup
    for _ in range(3):
        out = m(b)
        mx.eval(out)

    print("=" * 60)
    print("HTDemucs Stage Latency Breakdown (Batch Size 8)")
    print("=" * 60)

    # 1. Spec
    times_spec = []
    for _ in range(10):
        t0 = time.perf_counter()
        z = m._spec(b)
        mag = m._magnitude(z)
        mx.eval(z, mag)
        times_spec.append((time.perf_counter() - t0) * 1000)
    print(f"1. STFT & Mag:             {np.median(times_spec):6.2f} ms")

    # 2. Spectral Encoder
    z = m._spec(b)
    mag = m._magnitude(z)
    mx.eval(z, mag)
    times_enc = []
    for _ in range(10):
        x = mag
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)
        saved = []
        lengths = []
        t0 = time.perf_counter()
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            x = encode(x, None)
            if idx == 0 and m.freq_emb is not None:
                frs = mx.arange(x.shape[-2], dtype=mx.int32)
                emb = m.freq_emb(frs).transpose(1, 0)[None, :, :, None]
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        mx.eval(x, *saved)
        times_enc.append((time.perf_counter() - t0) * 1000)
    print(f"2. Spectral Encoder (4L):   {np.median(times_enc):6.2f} ms")

    # 3. Waveform Encoder (tencoder)
    times_tenc = []
    for _ in range(10):
        t0 = time.perf_counter()
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
        mx.eval(xt, *saved_t)
        times_tenc.append((time.perf_counter() - t0) * 1000)
    print(f"3. Waveform Encoder (4L):   {np.median(times_tenc):6.2f} ms")

    # 4. CrossTransformer
    # Prepare inputs to XT
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
    mx.eval(x, xt)

    times_xt = []
    for _ in range(10):
        x_in = x
        xt_in = xt
        t0 = time.perf_counter()
        if m.bottom_channels:
            b_sz, c, f, t_dim = x_in.shape
            x_in = x_in.reshape(b_sz, c, f * t_dim)
            xt_in = m.channel_upsampler_t(xt_in)
            x_in = m.channel_upsampler(x_in)
            x_in = x_in.reshape(b_sz, m.bottom_channels, f, t_dim)
        x_out, xt_out = m.crosstransformer(x_in, xt_in)
        if m.bottom_channels:
            x_out = x_out.reshape(b_sz, m.bottom_channels, f * t_dim)
            xt_out = m.channel_downsampler_t(xt_out)
            x_out = m.channel_downsampler(x_out)
            x_out = x_out.reshape(b_sz, c, f, t_dim)
        mx.eval(x_out, xt_out)
        times_xt.append((time.perf_counter() - t0) * 1000)
    print(f"4. CrossTransformer (5L):  {np.median(times_xt):6.2f} ms")

    # 5. Spectral Decoder
    times_dec = []
    for _ in range(10):
        x_in = x_out
        saved_copy = list(saved)
        lengths_copy = list(lengths)
        t0 = time.perf_counter()
        for idx, decode in enumerate(m.decoder):
            skip = saved_copy.pop(-1)
            x_in, _ = decode(x_in, skip, lengths_copy.pop(-1))
        mx.eval(x_in)
        times_dec.append((time.perf_counter() - t0) * 1000)
    print(f"5. Spectral Decoder (4L):  {np.median(times_dec):6.2f} ms")

    # 6. Waveform Decoder (tdecoder)
    times_tdec = []
    for _ in range(10):
        xt_in = xt_out
        saved_t_copy = list(saved_t)
        lengths_t_copy = list(lengths_t)
        t0 = time.perf_counter()
        for tdec in m.tdecoder:
            length_t = lengths_t_copy.pop(-1)
            skip_t = saved_t_copy.pop(-1)
            xt_in, _ = tdec(xt_in, skip_t, length_t)
        mx.eval(xt_in)
        times_tdec.append((time.perf_counter() - t0) * 1000)
    print(f"6. Waveform Decoder (4L):  {np.median(times_tdec):6.2f} ms")

    # 7. Post-processing: iSTFT & masking
    times_post = []
    for _ in range(10):
        x_in = x_out
        saved_copy = list(saved)
        lengths_copy = list(lengths)
        for idx, decode in enumerate(m.decoder):
            skip = saved_copy.pop(-1)
            x_in, _ = decode(x_in, skip, lengths_copy.pop(-1))
        t0 = time.perf_counter()
        B, C, Fq, T_dim = mag.shape
        S = len(m.sources)
        x_in = x_in.reshape(B, S, -1, Fq, T_dim)
        x_in = x_in * std[:, None] + mean[:, None]
        zout = m._mask(z, x_in)
        training_length = int(m.segment * m.samplerate)
        x_spec_out = m._ispec(zout, training_length)
        mx.eval(x_spec_out)
        times_post.append((time.perf_counter() - t0) * 1000)
    print(f"7. Masking & iSTFT:        {np.median(times_post):6.2f} ms")

    # Total full forward
    times_full = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = m(b)
        mx.eval(out)
        times_full.append((time.perf_counter() - t0) * 1000)
    print("=" * 60)
    print(f"Full Forward:              {np.median(times_full):6.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    profile()
