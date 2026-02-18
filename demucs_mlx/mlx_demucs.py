"""
MLX implementation of Demucs (inference-only).
Optimized and Corrected: 
- Fixed GroupNorm reshape bug for 3D tensors.
- Uses optimized matmul/sigmoid.
- Removed Method-level JIT to prevent runtime binding errors.
"""
from __future__ import annotations

import math
import typing as tp
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .mlx_layers import Conv1dNCL, ConvTranspose1dNCL, Lambda
from .mlx_utils import MLXStateDictMixin, center_trim, unfold

# ---------------------------------------------------------------------------
# Pure-MLX resampling (factor-2 only)
#
# Demucs uses a simple 2x upsample at the input of the time-domain model and a
# corresponding 2x downsample at the output when `self.resample` is enabled.
#
# This implementation avoids Torch/Torchaudio and keeps the inference path
# entirely in MLX. It uses a windowed-sinc low-pass (Hann window) + polyphase
# up/down for good quality and stable performance.
# ---------------------------------------------------------------------------

_RESAMPLE_FIR_CACHE: dict[tuple[int, float, str], mx.array] = {}

def _lowpass_fir_hann(numtaps: int, cutoff_cycles: float, dtype: mx.Dtype) -> mx.array:
    """Design a symmetric low-pass FIR using a Hann-windowed sinc.

    Args:
        numtaps: odd number of taps (e.g., 63)
        cutoff_cycles: cutoff in cycles/sample (Nyquist=0.5)
        dtype: mx.float32 / mx.float16, etc.

    Returns:
        (numtaps,) filter, normalized to DC gain 1.0
    """
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for symmetric 'same' padding")

    key = (int(numtaps), float(cutoff_cycles), str(dtype))
    h = _RESAMPLE_FIR_CACHE.get(key)
    if h is not None:
        return h

    n = mx.arange(numtaps, dtype=mx.float32)
    M = (numtaps - 1) / 2.0
    t = n - M

    # sinc(2*fc*t) where fc is cycles/sample, Nyquist = 0.5
    x = 2.0 * float(cutoff_cycles) * t
    # sin(pi*x)/(pi*x), define sinc(0)=1
    sinc = mx.where(t == 0, mx.ones_like(t), mx.sin(math.pi * x) / (math.pi * x))

    # Hann window
    w = 0.5 - 0.5 * mx.cos(2.0 * math.pi * n / float(numtaps - 1))

    h = (2.0 * float(cutoff_cycles)) * sinc * w
    h = h / mx.sum(h)

    h = h.astype(dtype)
    _RESAMPLE_FIR_CACHE[key] = h
    return h


def _reflect_pad_last(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """Reflect-pad on the last dimension. Works for any ndim >= 1."""
    L = x.shape[-1]
    if L <= 1:
        # Can't reflect with 1 or 0 elements, fall back to edge
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)], mode="edge")
    parts = []
    if pad_left > 0:
        left = x[..., 1: pad_left + 1][..., ::-1]
        parts.append(left)
    parts.append(x)
    if pad_right > 0:
        right = x[..., -pad_right - 1: -1][..., ::-1]
        parts.append(right)
    return mx.concatenate(parts, axis=-1)


_RESAMPLE_CONV_CACHE: dict[tuple[int, str], nn.Conv1d] = {}


def _get_depthwise_fir_conv(num_channels: int, h: mx.array) -> nn.Conv1d:
    """Get or create a depthwise Conv1d that applies FIR filter h to each channel."""
    key = (num_channels, id(h))
    conv = _RESAMPLE_CONV_CACHE.get(key)
    if conv is not None:
        return conv

    k = int(h.shape[0])
    # groups=num_channels makes it depthwise (each channel filtered independently)
    conv = nn.Conv1d(
        in_channels=num_channels,
        out_channels=num_channels,
        kernel_size=k,
        stride=1,
        padding=0,  # we handle reflect padding ourselves
        bias=False,
        groups=num_channels,
    )
    # Weight shape for MLX depthwise Conv1d: (C, K, 1)
    # Same filter for every channel
    w = mx.broadcast_to(h.reshape(1, -1, 1), (num_channels, k, 1))
    conv.weight = w
    conv.eval()
    _RESAMPLE_CONV_CACHE[key] = conv
    return conv


def _conv1d_same_ncl(x: mx.array, h: mx.array) -> mx.array:
    """Depthwise FIR 'same' conv for x in (N, C, L) using native Conv1d."""
    k = int(h.shape[0])
    pad = k // 2
    # Reflect padding reduces edge artifacts vs constant(0)
    xpad = _reflect_pad_last(x, pad, pad)
    # NCL -> NLC for MLX Conv1d
    xpad_nlc = xpad.transpose(0, 2, 1)
    C = x.shape[1]
    conv = _get_depthwise_fir_conv(C, h)
    y = conv(xpad_nlc)
    # NLC -> NCL
    return y.transpose(0, 2, 1)


def _resample_2x(x: mx.array) -> mx.array:
    """Upsample by 2 with zero-insertion + low-pass FIR."""
    B, C, L = x.shape
    # Interleave zeros: (B, C, L, 2) -> (B, C, 2L)
    x_up = mx.stack([x, mx.zeros_like(x)], axis=-1).reshape(B, C, 2 * L)

    # Half-band cutoff: fs/4 -> cycles/sample=0.25 (Nyquist=0.5)
    h = _lowpass_fir_hann(numtaps=63, cutoff_cycles=0.25, dtype=mx.float32)

    y = _conv1d_same_ncl(x_up.astype(mx.float32), h) * 2.0  # scale by up factor
    return y.astype(x.dtype)


def _resample_half(x: mx.array) -> mx.array:
    """Downsample by 2 with low-pass FIR + decimation."""
    h = _lowpass_fir_hann(numtaps=63, cutoff_cycles=0.25, dtype=mx.float32)
    y = _conv1d_same_ncl(x.astype(mx.float32), h)
    y = y[..., ::2]
    return y.astype(x.dtype)



def _dtype_from_str(dtype_str: str) -> mx.Dtype:
    if hasattr(mx, dtype_str):
        return getattr(mx, dtype_str)
    return mx.float32


@lru_cache(maxsize=32)
def _localstate_delta_eye_cached(T: int, dtype_str: str) -> tp.Tuple[mx.array, mx.array]:
    dtype = _dtype_from_str(dtype_str)
    indexes = mx.arange(T, dtype=dtype)
    delta = indexes[:, None] - indexes[None, :]
    eye = mx.eye(T, dtype=mx.bool_)
    return delta, eye


def _localstate_delta_eye(T: int, dtype: mx.Dtype) -> tp.Tuple[mx.array, mx.array]:
    T = int(T)
    if T > 10000:
        dtype_obj = _dtype_from_str(str(dtype))
        indexes = mx.arange(T, dtype=dtype_obj)
        delta = indexes[:, None] - indexes[None, :]
        eye = mx.eye(T, dtype=mx.bool_)
        return delta, eye
    return _localstate_delta_eye_cached(T, str(dtype))


def gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


def glu(x: mx.array, axis: int = 1) -> mx.array:
    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = mx.ones((num_channels,), dtype=mx.float32)
            self.bias = mx.zeros((num_channels,), dtype=mx.float32)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        B, C = x.shape[0], x.shape[1]
        G = self.num_groups
        if C % G != 0:
            raise ValueError(f"num_channels {C} not divisible by num_groups {G}")
        
        # Reshape to (B, G, C//G, ...)
        x_reshaped = x.reshape(B, G, C // G, *x.shape[2:])
        
        # Calculate stats
        axes = tuple(range(2, x_reshaped.ndim))
        mean = x_reshaped.mean(axis=axes, keepdims=True)
        var = ((x_reshaped - mean) ** 2).mean(axis=axes, keepdims=True)
        
        # Normalize
        x_norm = (x_reshaped - mean) * mx.rsqrt(var + self.eps)
        
        # Restore original shape
        # Fix: Use x.shape explicitly to handle both 3D (Audio) and 4D (Image) inputs correctly
        x_out = x_norm.reshape(x.shape)
        
        if self.affine:
            # Broadcast weight/bias: (1, C, 1...)
            shape = [1, C] + [1] * (x_out.ndim - 2)
            x_out = x_out * self.weight.reshape(shape) + self.bias.reshape(shape)
            
        return x_out


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 0.0, channel_last: bool = False):
        super().__init__()
        self.channel_last = bool(channel_last)
        self.scale = mx.zeros((channels,), dtype=mx.float32) + float(init)

    def __call__(self, x: mx.array) -> mx.array:
        if self.channel_last:
            return x * self.scale
        return x * self.scale[:, None]


class BLSTM(nn.Module):
    def __init__(
        self, dim: int, layers: int = 1,
        max_steps: tp.Optional[int] = None, skip: bool = False,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.skip = skip
        self.layers = layers
        self.forward_lstms = [
            nn.LSTM(input_size=dim if i == 0 else 2 * dim, hidden_size=dim)
            for i in range(layers)
        ]
        self.backward_lstms = [
            nn.LSTM(input_size=dim if i == 0 else 2 * dim, hidden_size=dim)
            for i in range(layers)
        ]
        self.linear = nn.Linear(2 * dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T = x.shape
        y = x
        framed = False
        
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.transpose(0, 2, 3, 1).reshape(-1, width, C)
        else:
            x = x.transpose(0, 2, 1)

        seq = x
        for lstm_f, lstm_b in zip(self.forward_lstms, self.backward_lstms):
            f_out, _ = lstm_f(seq)
            b_in = seq[:, ::-1, :]
            b_out, _ = lstm_b(b_in)
            b_out = b_out[:, ::-1, :]
            seq = mx.concatenate([f_out, b_out], axis=-1)

        x = self.linear(seq)
        x = x.transpose(0, 2, 1)
        
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = mx.concatenate(out, axis=-1)
            out = out[..., :T]
            x = out
            
        if self.skip:
            x = x + y
        return x


class LocalState(nn.Module):
    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = Conv1dNCL(channels, channels, 1)
        self.query = Conv1dNCL(channels, channels, 1)
        self.key = Conv1dNCL(channels, channels, 1)
        if nfreqs:
            self.query_freqs = Conv1dNCL(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = Conv1dNCL(channels, heads * ndecay, 1)
            self.query_decay.conv.weight *= 0.01
            self.query_decay.conv.bias = self.query_decay.conv.bias - 2
        self.proj = Conv1dNCL(channels + heads * nfreqs, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T = x.shape
        heads = self.heads
        delta, eye = _localstate_delta_eye(T, x.dtype)
        
        queries = self.query(x).reshape(B, heads, -1, T)
        keys = self.key(x).reshape(B, heads, -1, T)
        
        # Optimization: Use MatMul instead of Einsum for better hardware utilization
        keys_t = keys.transpose(0, 1, 3, 2)
        dots = mx.matmul(keys_t, queries)
        dots = dots * (1.0 / math.sqrt(keys.shape[2]))
        
        if self.nfreqs:
            periods = mx.arange(1, self.nfreqs + 1, dtype=x.dtype)
            freq_kernel = mx.cos(2 * math.pi * delta / periods.reshape(-1, 1, 1))
            freq_q = self.query_freqs(x).reshape(B, heads, -1, T)
            freq_scale = 1.0 / math.sqrt(self.nfreqs)
            dots = dots + mx.einsum(
                "fts,bhfs->bhts", freq_kernel, freq_q * freq_scale)

        if self.ndecay:
            # Memory-efficient decay term:
            # Original: decay_kernel[f,t,s] = -decays[f] * |t-s| / sqrt(ndecay)
            #           dots[b,h,t,s] += sum_f decay_kernel[f,t,s] * decay_q[b,h,f,s]
            #
            # Since |t-s| is independent of f, we can collapse the f dimension first:
            #   coeff[b,h,s] = sum_f decays[f] * decay_q[b,h,f,s]
            #   dots[b,h,t,s] += -|t-s| * coeff[b,h,s] / sqrt(ndecay)
            decays = mx.arange(1, self.ndecay + 1, dtype=x.dtype)  # (F,)
            decay_q = self.query_decay(x).reshape(B, heads, -1, T)  # (B,H,F,T)
            decay_q = mx.sigmoid(decay_q) * 0.5

            coeff = (decay_q * decays.reshape(1, 1, -1, 1)).sum(axis=2)  # (B,H,T)
            abs_delta = mx.abs(delta)  # (T,T)
            decay_scale = 1.0 / math.sqrt(self.ndecay)
            dots = dots - (
                abs_delta.reshape(1, 1, T, T)
                * coeff.reshape(B, heads, 1, T)
                * decay_scale
            )

        dots = mx.where(eye, mx.array(-100.0, dtype=dots.dtype), dots)
        weights = mx.softmax(dots, axis=2)
        
        content = self.content(x).reshape(B, heads, -1, T)
        
        content_t = content.transpose(0, 1, 3, 2)
        result_t = mx.matmul(weights, content_t)
        result = result_t.transpose(0, 1, 3, 2)

        if self.nfreqs:
            time_sig = mx.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = mx.concatenate([result, time_sig], axis=2)
            
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class DConv(nn.Module):
    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm: bool = True,
        attn: bool = False,
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,
        gelu_act: bool = True,
        kernel: int = 3,
        dilate: bool = True,
    ):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError("kernel must be odd")
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        def norm_fn(d: int) -> nn.Module:
            if norm:
                return GroupNorm(1, d)
            return nn.Identity()

        hidden = int(channels / compress)

        act = gelu if gelu_act else (lambda x: mx.maximum(x, 0))

        # Use FusedGroupNormGELU when norm is enabled and activation is gelu
        use_fused_gn_gelu = norm and gelu_act

        self.layers = []
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            if use_fused_gn_gelu:
                from .mlx_layers import FusedGroupNormGELU, FusedGroupNormGLU
                # FusedGroupNormGELU replaces GroupNorm + Lambda(gelu)
                # FusedGroupNormGLU replaces GroupNorm + Lambda(glu)
                # nn.Identity() keeps sequential indices stable for weight loading
                norm1_mod = FusedGroupNormGELU(1, hidden)
                act_mod = nn.Identity()
                norm2_mod = FusedGroupNormGLU(1, 2 * channels)
                glu_mod = nn.Identity()
            else:
                norm1_mod = norm_fn(hidden)
                act_mod = Lambda(act)
                norm2_mod = norm_fn(2 * channels)
                glu_mod = Lambda(lambda x: glu(x, axis=1))
            mods: tp.List[nn.Module] = [
                Conv1dNCL(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm1_mod,
                act_mod,
                Conv1dNCL(hidden, 2 * channels, 1),
                norm2_mod,
                glu_mod,
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            self.layers.append(nn.Sequential(*mods))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = x + layer(x)
        return x


class DemucsMLX(MLXStateDictMixin, nn.Module):
    def __init__(
        self,
        sources,
        audio_channels=2,
        channels=64,
        growth=2.0,
        depth=6,
        rewrite=True,
        lstm_layers=0,
        kernel_size=8,
        stride=4,
        context=1,
        gelu_act=True,
        glu_act=True,
        norm_starts=4,
        norm_groups=4,
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-4,
        normalize=True,
        resample=True,
        samplerate=44100,
        segment=4 * 10,
    ):
        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = []
        self.decoder = []
        self.skip_scales = []

        if glu_act:
            def act(x):
                return glu(x, axis=1)
            ch_scale = 2
        else:
            def act(x):
                return mx.maximum(x, 0)
            ch_scale = 1
        act2 = gelu if gelu_act else (lambda x: mx.maximum(x, 0))

        in_channels = audio_channels
        padding = 0
        for index in range(depth):
            def norm_fn(d):
                return nn.Identity()
            if index >= norm_starts:
                def norm_fn(d):
                    return GroupNorm(norm_groups, d)

            encode = [
                Conv1dNCL(in_channels, channels, kernel_size, stride),
                norm_fn(channels),
                Lambda(act2),
            ]
            attn = index >= dconv_attn
            lstm = index >= dconv_lstm
            if dconv_mode & 1:
                encode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            if rewrite:
                encode += [
                    Conv1dNCL(channels, ch_scale * channels, 1),
                    norm_fn(ch_scale * channels),
                    Lambda(act),
                ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [
                    Conv1dNCL(channels, ch_scale * channels, 2 * context + 1, padding=context),
                    norm_fn(ch_scale * channels),
                    Lambda(act),
                ]
            if dconv_mode & 2:
                decode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            decode += [ConvTranspose1dNCL(
                channels, out_channels, kernel_size, stride,
                padding=padding)]
            if index > 0:
                decode += [norm_fn(out_channels), Lambda(act2)]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

    def valid_length(self, length: int) -> int:
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def __call__(self, mix: mx.array) -> mx.array:
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mx.mean(mix, axis=1, keepdims=True)
            mean = mx.mean(mono, axis=-1, keepdims=True)
            std = mx.std(mono, axis=-1, keepdims=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = mx.pad(x, [(0, 0), (0, 0), (delta // 2, delta - delta // 2)], mode="constant")

        if self.resample:
            x = _resample_2x(x)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        if self.resample:
            x = _resample_half(x)
            
        x = x * std + mean
        x = center_trim(x, length)
        x = x.reshape(x.shape[0], len(self.sources), self.audio_channels, x.shape[-1])
        return x
