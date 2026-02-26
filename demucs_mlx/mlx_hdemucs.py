"""
MLX implementation of HDemucs building blocks (inference-only).
"""
from __future__ import annotations

import math
import typing as tp
from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn

from .mlx_demucs import DConv
from .mlx_layers import (
    GELUNCL,
    GLUNCL,
    Conv1dNCL,
    Conv2dNCHW,
    ConvTranspose1dNCL,
    ConvTranspose2dNCHW,
    FusedGroupNormGELU,
    FusedGroupNormGLU,
    GroupNormNCHW,
    GroupNormNCL,
    Identity,
)
from .mlx_utils import MLXStateDictMixin
from .spec_mlx import CachedSpectralPair
from .wiener_mlx import wiener


def pad1d(x: mx.array, paddings: tp.Tuple[int, int], mode: str = "constant", value: float = 0.0):
    """Pad on the last dimension, with reflect support for short inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            if extra_pad_left or extra_pad_right:
                x = mx.pad(
                    x,
                    [(0, 0)] * (x.ndim - 1) + [(extra_pad_left, extra_pad_right)],
                    mode="constant",
                    constant_values=value,
                )
                length = x.shape[-1]
        left, right = paddings
        if left:
            left_ref = x[..., 1:left + 1][..., ::-1]
            x = mx.concatenate([left_ref, x], axis=-1)
        if right:
            right_ref = x[..., -right - 1:-1][..., ::-1]
            x = mx.concatenate([x, right_ref], axis=-1)
    else:
        x = mx.pad(
            x,
            [(0, 0)] * (x.ndim - 1) + [(padding_left, padding_right)],
            mode=mode,
            constant_values=value,
        )
    # assert x.shape[-1] == length + padding_left + padding_right
    # assert mx.all(x[..., padding_left: padding_left + length] == x0)
    return x


class ScaledEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        scale: float = 10.0, smooth: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = mx.cumsum(self.embedding.weight, axis=0)
            weight = weight / mx.sqrt(mx.arange(1, num_embeddings + 1, dtype=weight.dtype))[:, None]
            self.embedding.weight = weight
        self.embedding.weight = self.embedding.weight / scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def __call__(self, x):
        return self.embedding(x) * self.scale


class HEncLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=0,
        dconv_kw={},
        pad=True,
        rewrite=True,
    ):
        super().__init__()
        def norm_fn(d):
            return Identity()
        if norm:
            def norm_fn(d):
                return GroupNormNCHW(norm_groups, d) if freq else GroupNormNCL(norm_groups, d)
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            self.conv = Conv2dNCHW(chin, chout, kernel_size, stride, pad)
        else:
            self.conv = Conv1dNCL(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        # Use fused GroupNorm+GELU when norm is enabled
        self._fused_norm1 = bool(norm)
        if norm:
            self.norm1 = FusedGroupNormGELU(norm_groups, chout)
        else:
            self.norm1 = norm_fn(chout)
        self.rewrite = None
        self._fused_norm2 = False
        if rewrite:
            if freq:
                self.rewrite = Conv2dNCHW(chout, 2 * chout, 1 + 2 * context, 1, context)
            else:
                self.rewrite = Conv1dNCL(chout, 2 * chout, 1 + 2 * context, 1, context)
            if norm:
                self.norm2 = FusedGroupNormGLU(norm_groups, 2 * chout)
                self._fused_norm2 = True
            else:
                self.norm2 = norm_fn(2 * chout)
        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def __call__(self, x, inject=None):
        if not self.freq and x.ndim == 4:
            B, C, Fr, T = x.shape
            x = x.reshape(B, -1, T)
        if not self.freq:
            le = x.shape[-1]
            if le % self.stride != 0:
                x = pad1d(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.ndim == 3 and y.ndim == 4:
                inject = inject[:, :, None]
            y = y + inject
        if self._fused_norm1:
            y = self.norm1(y)
        else:
            y = GELUNCL()(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
        if self.rewrite:
            if self._fused_norm2:
                z = self.norm2(self.rewrite(y))
            else:
                z = GLUNCL(axis=1)(self.norm2(self.rewrite(y)))
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw={},
        pad=True,
        context_freq=True,
        rewrite=True,
    ):
        super().__init__()
        def norm_fn(d):
            return Identity()
        if norm:
            def norm_fn(d):
                return GroupNormNCHW(norm_groups, d) if freq else GroupNormNCL(norm_groups, d)
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            self.conv_tr = ConvTranspose2dNCHW(chin, chout, kernel_size, stride)
        else:
            self.conv_tr = ConvTranspose1dNCL(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        self._fused_norm1 = False
        if rewrite:
            if freq:
                if context_freq:
                    self.rewrite = Conv2dNCHW(chin, 2 * chin, 1 + 2 * context, 1, context)
                else:
                    self.rewrite = Conv2dNCHW(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])
            else:
                self.rewrite = Conv1dNCL(chin, 2 * chin, 1 + 2 * context, 1, context)
            if norm:
                self.norm1 = FusedGroupNormGLU(norm_groups, 2 * chin)
                self._fused_norm1 = True
            else:
                self.norm1 = norm_fn(2 * chin)
        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def __call__(self, x, skip, length):
        if self.freq and x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B, self.chin, -1, T)
        if not self.empty:
            x = x + skip
            if self.rewrite:
                if self._fused_norm1:
                    y = self.norm1(self.rewrite(x))
                else:
                    y = GLUNCL(axis=1)(self.norm1(self.rewrite(x)))
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
        else:
            y = x
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
        if not self.last:
            z = GELUNCL()(z)
        return z, y


class MultiWrap(nn.Module):
    def __init__(self, layer, split_ratios):
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = []
        self.conv = isinstance(layer, HEncLayer)
        self.layers = [deepcopy(layer) for _ in range(len(split_ratios) + 1)]

    def __call__(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape
        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = pad1d(y, (pad, 0))
                if ratio == 1:
                    y = pad1d(y, (0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True
                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride:] += out[:, :, :layer.stride]
                    out = out[:, :, layer.stride:]
                if ratio == 1:
                    out = out[:, :, :-layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = mx.concatenate(outs, axis=2)
        if not self.conv and not last:
            out = GELUNCL()(out)
        if self.conv:
            return out
        return out, None


class HDemucsMLX(MLXStateDictMixin, nn.Module):
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
        depth=6,
        rewrite=True,
        hybrid=True,
        hybrid_old=False,
        multi_freqs=None,
        multi_freqs_depth=2,
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
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-4,
        rescale=0.1,
        samplerate=44100,
        segment=4 * 10,
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
            hybrid=hybrid,
            hybrid_old=hybrid_old,
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
            dconv_attn=dconv_attn,
            dconv_lstm=dconv_lstm,
            dconv_init=dconv_init,
            samplerate=samplerate,
            segment=segment,
        ))
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

        self.nfft = nfft
        self.hop_length = nfft // 4
        self._spectral = CachedSpectralPair(n_fft=nfft, hop_length=nfft // 4)
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        if hybrid_old and not hybrid:
            raise ValueError("hybrid_old must come with hybrid=True")
        if hybrid and wiener_iters != end_iters:
            raise ValueError("Hybrid requires wiener_iters == end_iters")

        self.encoder = []
        self.decoder = []
        if hybrid:
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
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
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
                    "lstm": lstm,
                    "attn": attn,
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
            if hybrid and freq:
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
            if hybrid and freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt,
                )
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

    def _spec(self, x: mx.array) -> mx.array:
        hl = self.hop_length
        nfft = self.nfft
        if self.hybrid:
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
            else:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]))
        
        z = self._spectral.stft(x)[..., :-1, :]
        
        if self.hybrid:
            z = z[..., 2:2 + le]
        return z

    def _ispec(self, z: mx.array, length: tp.Optional[int] = None, scale: int = 0) -> mx.array:
        hl = self.hop_length // (4 ** scale)
        def _pad_last_dims(arr: mx.array, pads: tp.List[tp.Tuple[int, int]]):
            lead = arr.ndim - len(pads)
            if lead < 0:
                raise ValueError("pad spec has more dims than array")
            full_pads = [(0, 0)] * lead + pads
            return mx.pad(arr, full_pads)

        z = _pad_last_dims(z, [(0, 0), (0, 1), (0, 0)])
        if self.hybrid:
            z = _pad_last_dims(z, [(0, 0), (0, 0), (2, 2)])
            pad = hl // 2 * 3
            if not self.hybrid_old:
                le = hl * int(math.ceil(length / hl)) + 2 * pad
            else:
                le = hl * int(math.ceil(length / hl))
            
            x = self._spectral.istft(z, length=le)

            if not self.hybrid_old:
                x = x[..., pad:pad + length]
            else:
                x = x[..., :length]
        else:
            x = self._spectral.istft(z, length=length)
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

        B, S, C, Fq, T = mag_out.shape
        mag_out_mx = mag_out.transpose(0, 4, 3, 2, 1)  # [B, T, F, C, S]
        mix_stft_ri = mx.stack([mx.real(mix_stft), mx.imag(mix_stft)], axis=-1)
        mix_stft_mx = mix_stft_ri.transpose(0, 3, 2, 1, 4)  # [B, T, F, C, 2]

        # OPTIMIZATION: Use vmap to parallelize batch processing
        def _process_one_sample(mag_sample, mix_sample):
            out_chunks = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_sample[frame],
                    mix_sample[frame],
                    niters,
                    residual=residual,
                )
                out_chunks.append(z_out.transpose(0, 1, 2, 4, 3))
            return mx.concatenate(out_chunks, axis=0)

        # Apply vmap over batch dimension
        out = mx.vmap(_process_one_sample)(mag_out_mx, mix_stft_mx)

        if residual:
            out = out[..., :-1, :]
        out = out.transpose(0, 4, 3, 2, 1, 5)  # [B, S, C, F, T, 2]
        out = out[..., 0] + 1j * out[..., 1]
        return out.astype(init)

    def __call__(self, mix: mx.array) -> mx.array:
        x = mix
        length = x.shape[-1]

        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)

        if self.hybrid:
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
            if self.hybrid and idx < len(self.tencoder):
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

        x = mx.zeros_like(x)
        if self.hybrid:
            xt = mx.zeros_like(x)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
            if self.hybrid and idx >= offset:
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
        x = self._ispec(zout, length)

        if self.hybrid:
            xt = xt.reshape(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x
        return x
