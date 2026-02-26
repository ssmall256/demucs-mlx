"""
MLX implementation of HTDemucs (inference-only).
"""
from __future__ import annotations

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .mlx_hdemucs import HDecLayer, HEncLayer, MultiWrap, ScaledEmbedding, pad1d
from .mlx_layers import Conv1dNCL
from .mlx_transformer import CrossTransformerEncoder
from .mlx_utils import MLXStateDictMixin, center_trim
from .spec_mlx import CachedSpectralPair
from .wiener_mlx import wiener


class HTDemucsMLX(MLXStateDictMixin, nn.Module):
    def __init__(
        self,
        sources,
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        depth=4,
        rewrite=True,
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        norm_starts=4,
        norm_groups=4,
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        bottom_channels=0,
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        t_cross_first=False,
        rescale=0.1,
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
        super().__init__()
        self._init_args_kwargs = ([], dict(
            sources=sources,
            audio_channels=audio_channels,
            channels=channels,
            channels_time=channels_time,
            growth=growth,
            nfft=nfft,
            wiener_iters=wiener_iters,
            end_iters=end_iters,
            wiener_residual=wiener_residual,
            cac=cac,
            depth=depth,
            rewrite=rewrite,
            multi_freqs=multi_freqs,
            multi_freqs_depth=multi_freqs_depth,
            freq_emb=freq_emb,
            emb_scale=emb_scale,
            emb_smooth=emb_smooth,
            kernel_size=kernel_size,
            time_stride=time_stride,
            stride=stride,
            context=context,
            context_enc=context_enc,
            norm_starts=norm_starts,
            norm_groups=norm_groups,
            dconv_mode=dconv_mode,
            dconv_depth=dconv_depth,
            dconv_comp=dconv_comp,
            dconv_init=dconv_init,
            bottom_channels=bottom_channels,
            t_layers=t_layers,
            t_emb=t_emb,
            t_hidden_scale=t_hidden_scale,
            t_heads=t_heads,
            t_dropout=t_dropout,
            t_max_positions=t_max_positions,
            t_norm_in=t_norm_in,
            t_norm_in_group=t_norm_in_group,
            t_group_norm=t_group_norm,
            t_norm_first=t_norm_first,
            t_norm_out=t_norm_out,
            t_max_period=t_max_period,
            t_weight_decay=t_weight_decay,
            t_lr=t_lr,
            t_layer_scale=t_layer_scale,
            t_gelu=t_gelu,
            t_weight_pos_embed=t_weight_pos_embed,
            t_sin_random_shift=t_sin_random_shift,
            t_cape_mean_normalize=t_cape_mean_normalize,
            t_cape_augment=t_cape_augment,
            t_cape_glob_loc_scale=t_cape_glob_loc_scale,
            t_sparse_self_attn=t_sparse_self_attn,
            t_sparse_cross_attn=t_sparse_cross_attn,
            t_mask_type=t_mask_type,
            t_mask_random_seed=t_mask_random_seed,
            t_sparse_attn_window=t_sparse_attn_window,
            t_global_window=t_global_window,
            t_sparsity=t_sparsity,
            t_auto_sparsity=t_auto_sparsity,
            t_cross_first=t_cross_first,
            rescale=rescale,
            samplerate=samplerate,
            segment=segment,
            use_train_segment=use_train_segment,
        ))
        if t_sparse_self_attn or t_sparse_cross_attn:
            raise RuntimeError("Sparse attention is not supported in MLX backend.")
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment

        self.nfft = nfft
        self.hop_length = nfft // 4
        self._spectral = CachedSpectralPair(n_fft=nfft, hop_length=nfft // 4)
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        if wiener_iters != end_iters:
            raise ValueError("HTDemucs requires wiener_iters == end_iters")

        self.encoder = []
        self.decoder = []
        self.tencoder = []
        self.tdecoder = []

        chin = audio_channels
        chin_z = chin
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                ker = time_stride * 2
                stri = time_stride
            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu_act": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = False
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw)
            if freq:
                tenc = HEncLayer(
                    chin, chout, dconv=dconv_mode & 1,
                    context=context_enc, empty=last_freq, **kwt)
                self.tencoder.append(tenc)
            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * (2 if self.cac else 1)

            dec = HDecLayer(
                chout_z, chin_z, dconv=dconv_mode & 2,
                last=index == 0, context=context, **kw_dec)
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout, chin, dconv=dconv_mode & 2,
                    empty=last_freq, last=index == 0,
                    context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        transformer_channels = channels * (growth ** (depth - 1))
        self.bottom_channels = bottom_channels
        if bottom_channels:
            self.channel_upsampler = Conv1dNCL(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = Conv1dNCL(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = Conv1dNCL(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = Conv1dNCL(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_pos_embed=t_weight_pos_embed,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x: mx.array) -> mx.array:
        hl = self.hop_length
        nfft = self.nfft
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = self._spectral.stft(x)[..., :-1, :]
        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z: mx.array, length: tp.Optional[int] = None, scale: int = 0) -> mx.array:
        hl = self.hop_length // (4 ** scale)
        # Handle both 4D (B, C, Fr, T) and 5D (B, S, C, Fr, T) inputs
        if z.ndim == 5:
            z = mx.pad(z, [(0, 0), (0, 0), (0, 0), (0, 1), (2, 2)])
        else:
            z = mx.pad(z, [(0, 0), (0, 0), (0, 1), (2, 2)])
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = self._spectral.istft(z, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z: mx.array) -> mx.array:
        if self.cac:
            B, C, Fr, T = z.shape
            m = mx.stack([mx.real(z), mx.imag(z)], axis=2).reshape(B, C * 2, Fr, T)
        else:
            m = mx.abs(z)
        return m

    def _mask(self, z: mx.array, m: mx.array) -> mx.array:
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.reshape(B, S, -1, 2, Fr, T).transpose(0, 1, 2, 4, 5, 3)
            real = out[..., 0]
            imag = out[..., 1]
            return real + 1j * imag
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + mx.abs(z)) * m
        return self._wiener(m, z, niters)

    def _wiener(self, mag_out: mx.array, mix_stft: mx.array, niters: int) -> mx.array:
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        # [B, S, C, Fq, T]
        B, S, C, Fq, T = mag_out.shape
        mag_out_mx = mag_out.transpose(0, 4, 3, 2, 1)  # [B, T, F, C, S]
        
        mix_stft_ri = mx.stack([mx.real(mix_stft), mx.imag(mix_stft)], axis=-1)
        mix_stft_mx = mix_stft_ri.transpose(0, 3, 2, 1, 4)  # [B, T, F, C, 2]

        # OPTIMIZATION: Use vmap to parallelize batch processing
        def _process_one_sample(mag_sample, mix_sample):
            out_chunks = []
            # We still keep the time chunking loop to save memory
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_sample[frame],
                    mix_sample[frame],
                    niters,
                    residual=residual,
                )
                # z_out: [T_chunk, F, C, S, 2]
                out_chunks.append(z_out.transpose(0, 1, 2, 4, 3))
            # Concatenate time chunks: [T, F, C, S, 2]
            return mx.concatenate(out_chunks, axis=0)

        # Apply vmap over the batch dimension (axis 0)
        out = mx.vmap(_process_one_sample)(mag_out_mx, mix_stft_mx)

        # Post-process: [B, T, F, C, S, 2] -> ...
        if residual:
            out = out[..., :-1, :]
        
        out = out.transpose(0, 4, 3, 2, 1, 5)  # [B, S, C, F, T, 2]
        out = out[..., 0] + 1j * out[..., 1]
        return out.astype(init)

    def valid_length(self, length: int):
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                f"Given length {length} is longer than training length {training_length}"
            )
        return training_length

    def __call__(self, mix: mx.array) -> mx.array:
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length_pre_pad)])
        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.std(xt, axis=(1, 2), keepdims=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = mx.arange(x.shape[-2], dtype=mx.int32)
                emb = self.freq_emb(frs).transpose(1, 0)[None, :, :, None]
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = self.channel_upsampler(x)
                x = x.reshape(b, self.bottom_channels, f, t)
                xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt)
            if self.bottom_channels:
                x = x.reshape(b, self.bottom_channels, f * t)
                x = self.channel_downsampler(x)
                x = x.reshape(b, c, f, t)
                xt = self.channel_downsampler_t(xt)

        offset = self.depth - len(self.tdecoder)
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        if len(saved) != 0 or len(lengths_t) != 0 or len(saved_t) != 0:
            raise RuntimeError("Skip connections not fully consumed")

        S = len(self.sources)
        x = x.reshape(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(z, x)
        if self.use_train_segment:
            x = self._ispec(zout, training_length)
        else:
            x = self._ispec(zout, length)

        # Reshape xt to match expected output shape
        actual_length = xt.shape[-1]
        xt = xt.reshape(B, S, -1, actual_length)
        xt = xt * stdt[:, None] + meant[:, None]
        # Trim x to match xt length before adding
        x = center_trim(x, xt)
        x = xt + x
        # Trim to final target length
        if self.use_train_segment:
            x = x[..., :training_length]
        else:
            x = x[..., :length]
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x
