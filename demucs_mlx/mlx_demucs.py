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

from .mlx_layers import (
    Conv1dNCL,
    ConvTranspose1dNCL,
    Lambda,
    _group_norm_via_layer_norm,
    _use_fused_gn_glu,
)
from .mlx_utils import (
    MLXStateDictMixin,
    center_trim,
    is_dconv_compile_enabled,
    unfold,
)

# ---------------------------------------------------------------------------
# Pure-MLX resampling (factor-2 only)
#
# Demucs upsamples its input x2 before the time-domain network and downsamples
# the output /2 when `self.resample` is set, with julius.resample_frac. This is
# the same algorithm in MLX: a windowed-sinc kernel with `zeros` zero crossings
# and a `rolloff` low-pass, one phase per output sample, edge padding, and the
# output truncated to floor(new_sr * length / old_sr). Any other filter changes
# the model's output well above float noise.
# ---------------------------------------------------------------------------

_RESAMPLE_KERNEL_CACHE: dict[tuple[int, int, int, float], tuple[mx.array, int]] = {}


def _julius_kernel(old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945):
    """Return (kernel [new_sr, K, 1] for an NLC conv1d, width), as julius.ResampleFrac."""
    key = (old_sr, new_sr, zeros, rolloff)
    cached = _RESAMPLE_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    sr = min(new_sr, old_sr) * rolloff
    width = math.ceil(zeros * old_sr / sr)
    idx = mx.arange(-width, width + old_sr, dtype=mx.float32)
    kernels = []
    for i in range(new_sr):
        t = (-i / new_sr + idx / old_sr) * sr
        t = mx.clip(t, -zeros, zeros) * math.pi
        window = mx.cos(t / zeros / 2) ** 2
        sinc = mx.where(t == 0, mx.ones_like(t), mx.sin(t) / mx.where(t == 0, mx.ones_like(t), t))
        kernel = sinc * window
        kernels.append(kernel / mx.sum(kernel))
    kernel = mx.stack(kernels)[..., None]
    mx.eval(kernel)
    _RESAMPLE_KERNEL_CACHE[key] = (kernel, width)
    return kernel, width


def _resample_frac(x: mx.array, old_sr: int, new_sr: int) -> mx.array:
    """julius.resample_frac(x, old_sr, new_sr) for x of shape (..., T)."""
    kernel, width = _julius_kernel(old_sr, new_sr)
    shape = x.shape
    length = shape[-1]
    flat = x.reshape(-1, length).astype(mx.float32)
    flat = mx.pad(flat, [(0, 0), (width, width + old_sr)], mode="edge")
    # NLC conv with stride old_sr: (N, L', new_sr), one channel per phase.
    phases = mx.conv1d(flat[..., None], kernel, stride=old_sr)
    y = phases.reshape(*shape[:-1], -1)
    out_length = (new_sr * length) // old_sr
    return y[..., :out_length].astype(x.dtype)


def _resample_2x(x: mx.array) -> mx.array:
    """Upsample by 2 (julius.resample_frac(x, 1, 2))."""
    return _resample_frac(x, 1, 2)


def _resample_half(x: mx.array) -> mx.array:
    """Downsample by 2 (julius.resample_frac(x, 2, 1))."""
    return _resample_frac(x, 2, 1)


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
        return _group_norm_via_layer_norm(
            x, self.num_groups, self.eps, self.weight, self.bias
        )


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
        # result[c, s] = sum_t content[c, t] * weights[t, s]  (upstream einsum
        # "bhts,bhct->bhcs"); the softmax above normalizes over t (keys).
        result = mx.matmul(content, weights)

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
        self._compile_inference = not (attn or lstm)
        self._compiled_layers = None
        self._compiled_signatures = None
        dilate = depth > 0

        def norm_fn(d: int) -> nn.Module:
            if norm:
                return GroupNorm(1, d)
            return nn.Identity()

        hidden = int(channels / compress)

        act = gelu if gelu_act else (lambda x: mx.maximum(x, 0))

        # Use FusedGroupNormGELU when norm is enabled and activation is gelu
        use_fused_gn_gelu = norm and gelu_act and _use_fused_gn_glu()

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
        layers = self.layers
        compile_enabled = is_dconv_compile_enabled()
        use_nlc = not self.training and self._compile_inference and _can_use_dconv_nlc(layers)
        if use_nlc:
            if compile_enabled:
                signature = _dconv_chain_signature(layers)
                compiled = self._compiled_layers
                if compiled is None or self._compiled_signatures != signature:
                    compiled = _compile_dconv_chain(layers)
                    self._compiled_layers = compiled
                    self._compiled_signatures = signature
                return compiled(x)
            return _dconv_chain_forward_nlc(layers, x)
        for layer in layers:
            x = x + layer(x)
        return x


def _can_use_dconv_nlc(layers: list[nn.Module]) -> bool:
    """Whether every DConv block has the standard 7-layer structure (no attention or LSTM)."""
    for block in layers:
        if not hasattr(block, "layers") or len(block.layers) != 7:
            return False
        # Expected modules: Conv1dNCL, norm, act, Conv1dNCL, norm, glu, LayerScale
        if not hasattr(block.layers[0], "conv") or not hasattr(block.layers[3], "conv"):
            return False
        if not hasattr(block.layers[6], "scale"):
            return False
    return True


def _dconv_block_forward_nlc(block: nn.Module, x_nlc: mx.array) -> mx.array:
    """Forward pass of a single DConv block in native NLC channels-last layout."""
    N, L, C = x_nlc.shape
    conv1 = block.layers[0].conv
    norm1 = block.layers[1]
    conv2 = block.layers[3].conv
    norm2 = block.layers[4]
    scale = block.layers[6].scale

    # Conv 1 in native NLC layout
    h = conv1(x_nlc)
    hidden = h.shape[-1]

    # Norm 1 via fast layer_norm
    eps1 = getattr(norm1, "eps", 1e-5)
    h_norm = mx.fast.layer_norm(h.reshape(N, 1, -1), None, None, eps1).reshape(N, L, hidden)
    if getattr(norm1, "affine", True) and getattr(norm1, "weight", None) is not None:
        h_norm = h_norm * norm1.weight[None, None, :] + norm1.bias[None, None, :]
    h_act = nn.gelu(h_norm)

    # Conv 2 in native NLC layout
    out = conv2(h_act)
    out_c = out.shape[-1]

    # Norm 2 via fast layer_norm
    eps2 = getattr(norm2, "eps", 1e-5)
    out_norm = mx.fast.layer_norm(out.reshape(N, 1, -1), None, None, eps2).reshape(N, L, out_c)
    if getattr(norm2, "affine", True) and getattr(norm2, "weight", None) is not None:
        out_norm = out_norm * norm2.weight[None, None, :] + norm2.bias[None, None, :]

    # GLU along last dimension
    a, b = mx.split(out_norm, 2, axis=-1)
    glu_out = a * mx.sigmoid(b)
    return glu_out * scale[None, None, :]


def _dconv_chain_forward_nlc(layers: list[nn.Module], x_ncl: mx.array) -> mx.array:
    """Run all DConv blocks with one input and one output transpose, not one per layer."""
    x_nlc = x_ncl.transpose(0, 2, 1)
    for block in layers:
        x_nlc = x_nlc + _dconv_block_forward_nlc(block, x_nlc)
    return x_nlc.transpose(0, 2, 1)


def _compile_dconv_chain(layers: list[nn.Module]):
    """Compile the entire DConv chain in native NLC layout."""
    def forward(value):
        return _dconv_chain_forward_nlc(layers, value)

    return mx.compile(forward)


def _dconv_chain_signature(layers: list[nn.Module]) -> tuple[tuple[int, ...], ...]:
    """Invalidate captured weights when any block in the chain is edited."""
    return tuple(_dconv_block_signature(layer) for layer in layers)


def _compile_dconv_block(block: nn.Module):
    """Keep compiled graphs outside the parameter tree of an inference block."""

    def forward(value):
        return block(value)

    return mx.compile(forward)


def _dconv_block_signature(block: nn.Module) -> tuple[int, ...]:
    """Invalidate captured weights when an inference-only block is edited."""
    modules = (
        block.layers[0].conv,
        block.layers[1],
        block.layers[3].conv,
        block.layers[4],
        block.layers[6],
    )
    return (id(block),) + tuple(
        id(getattr(module, name, None))
        for module in modules
        for name in ("weight", "bias", "scale")
    )


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
            std = mx.std(mono, axis=-1, keepdims=True, ddof=1)
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
