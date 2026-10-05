"""
Shared MLX layers and NCL/NCHW wrappers.
Optimized for memory layout efficiency.
"""
from __future__ import annotations

import os
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from .mlx_utils import materialize_cached


def _spatial_pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else (value[0], value[1])


def _conv_channels_last(conv: nn.Conv1d | nn.Conv2d, x: mx.array) -> mx.array:
    """Use a fused-bias GEMM for ungrouped FP32 kernel-one projections.

    Keep the original Conv module/parameter names and read its current weights
    on every call, including after checkpoint updates. All other cases retain
    MLX's convolution path.
    """
    weight = conv.weight
    if (
        conv.groups == 1
        and all(size == 1 for size in weight.shape[1:-1])
        and _spatial_pair(conv.stride) == (1, 1)
        and _spatial_pair(conv.padding) == (0, 0)
        and x.dtype == mx.float32
        and weight.dtype == mx.float32
    ):
        matrix = weight.reshape(weight.shape[0], weight.shape[-1]).T
        return mx.addmm(conv.bias, x, matrix) if "bias" in conv else x @ matrix
    return conv(x)


class Lambda(nn.Module):
    def __init__(self, fn: tp.Callable[[mx.array], mx.array]):
        super().__init__()
        self.fn = fn

    def __call__(self, x: mx.array) -> mx.array:
        return self.fn(x)


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class Sequential(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Conv1dNCL(nn.Module):
    """
    Conv1d wrapper for NCL (Batch, Channels, Length) layout.
    MLX Conv1d expects NLC, so we transpose inputs/outputs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(0, 2, 1)
        y = _conv_channels_last(self.conv, x)
        # y: (N, L, C) -> (N, C, L)
        return y.transpose(0, 2, 1)


class ConvTranspose1dNCL(nn.Module):
    """
    ConvTranspose1d wrapper for NCL (Batch, Channels, Length) layout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            bias=bias,
        )
        self._phased_cache = None

    def __call__(self, x: mx.array) -> mx.array:
        conv = self.conv
        if (
            conv.weight.shape[1] == 8
            and conv.stride == 4
            and conv.padding == 0
            and conv.dilation == 1
            and conv.output_padding == 0
            and x.dtype == mx.float32
            and conv.weight.dtype == mx.float32
        ):
            return self._phased_convolution(x)
        x = x.transpose(0, 2, 1)
        y = conv(x)
        return y.transpose(0, 2, 1)

    def _phased_convolution(self, x: mx.array) -> mx.array:
        """Interleave four two-tap Conv1d outputs for stride-four deconvolution."""
        conv = self.conv
        source = conv.weight
        cache = self._phased_cache
        if cache is None or cache.source is not source:
            phase_weights = [
                mx.stack([source[:, phase + 4, :], source[:, phase, :]], axis=1)
                for phase in range(4)
            ]
            weight = mx.concatenate(phase_weights, axis=0)
            materialize_cached(weight)
            cache = _PhasedWeightCache(source, weight)
            self._phased_cache = cache

        batch, _, length = x.shape
        nlc = x.transpose(0, 2, 1)
        padded = mx.pad(nlc, [(0, 0), (1, 1), (0, 0)])
        phases = mx.conv1d(padded, cache.weight)
        out_channels = source.shape[0]
        joined = phases.reshape(batch, length + 1, 4, out_channels)
        joined = joined.reshape(batch, 4 * (length + 1), out_channels)
        if "bias" in conv:
            joined = joined + conv.bias
        return joined.transpose(0, 2, 1)


class Conv2dNCHW(nn.Module):
    """
    Conv2d wrapper for NCHW (Batch, Channels, Height, Width) layout.
    MLX Conv2d expects NHWC.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, C, H, W) -> (N, H, W, C)
        x = x.transpose(0, 2, 3, 1)
        y = _conv_channels_last(self.conv, x)
        # y: (N, H, W, C) -> (N, C, H, W)
        return y.transpose(0, 3, 1, 2)


class ConvTranspose2dNCHW(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        output_padding=0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            bias=bias,
        )
        self._phased_cache = None

    def __call__(self, x: mx.array) -> mx.array:
        conv = self.conv
        if (
            conv.weight.shape[1:3] == (8, 1)
            and _spatial_pair(conv.stride) == (4, 1)
            and _spatial_pair(conv.padding) == (0, 0)
            and _spatial_pair(conv.dilation) == (1, 1)
            and _spatial_pair(conv.output_padding) == (0, 0)
            and x.dtype == mx.float32
            and conv.weight.dtype == mx.float32
        ):
            return self._phased_convolution(x)
        x = x.transpose(0, 2, 3, 1)
        y = conv(x)
        return y.transpose(0, 3, 1, 2)

    def _phased_convolution(self, x: mx.array) -> mx.array:
        """Compute stride-four deconvolution as four two-tap output phases."""
        conv = self.conv
        source = conv.weight
        cache = self._phased_cache
        if cache is None or cache.source is not source:
            phase_weights = [
                mx.stack([source[:, phase + 4, 0, :], source[:, phase, 0, :]], axis=1)
                for phase in range(4)
            ]
            weight = mx.concatenate(phase_weights, axis=0).reshape(
                -1, 2, 1, source.shape[-1]
            )
            materialize_cached(weight)
            cache = _PhasedWeightCache(source, weight)
            self._phased_cache = cache

        batch, _, frequency, frames = x.shape
        nhwc = x.transpose(0, 2, 3, 1)
        padded = mx.pad(nhwc, [(0, 0), (1, 1), (0, 0), (0, 0)])
        phases = mx.conv2d(padded, cache.weight)
        out_channels = source.shape[0]
        phased = phases.reshape(batch, frequency + 1, frames, 4, out_channels)
        joined = phased.transpose(0, 1, 3, 2, 4).reshape(
            batch, 4 * (frequency + 1), frames, out_channels
        )
        if "bias" in conv:
            joined = joined + conv.bias
        return joined.transpose(0, 3, 1, 2)


class _PhasedWeightCache:
    """Keep derived weights outside MLX Module's serializable parameter tree."""

    def __init__(self, source: mx.array, weight: mx.array):
        self.source = source
        self.weight = weight


def _use_fused_gn_glu() -> bool:
    """Whether to build fused GroupNorm+activation Metal kernels.

    Defaults to disabled. These kernels were on unconditionally until they were
    measured against the unfused path on a 45 s clip through htdemucs: they
    cost roughly 20 dB SNR (19.7 dB on drums, 23.7 dB on other) -- audible, not
    float noise -- because the kernel uses an erf-approximation GELU and
    threadgroup reductions whose width varies with tensor shape. They are also
    not faster: 0.783 s fused vs 0.776 s unfused, median of five timed runs,
    since at real Demucs shapes the group size mostly exceeds the hybrid
    threshold and falls back anyway. Strictly worse on both axes.

    Set DEMUCS_MLX_USE_FUSED_GN_GLU=1 to re-enable them for benchmarking. The
    fused and unfused layers expose the same parameter names, so an existing
    converted cache loads either way.
    """
    raw = os.getenv("DEMUCS_MLX_USE_FUSED_GN_GLU", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _group_norm_via_layer_norm(
    x: mx.array,
    num_groups: int,
    eps: float,
    weight: mx.array | None,
    bias: mx.array | None,
) -> mx.array:
    """Normalize each channel group with MLX's fused last-axis kernel."""
    batch, channels = x.shape[:2]
    if channels % num_groups:
        raise ValueError(f"num_channels {channels} not divisible by num_groups {num_groups}")
    grouped = x.reshape(batch, num_groups, -1)
    normalized = mx.fast.layer_norm(grouped, None, None, eps).reshape(x.shape)
    if weight is None:
        return normalized
    affine_shape = (1, channels) + (1,) * (x.ndim - 2)
    return normalized * weight.reshape(affine_shape) + bias.reshape(affine_shape)


class GroupNormNCL(nn.Module):
    """
    Optimized GroupNorm for NCL layout.
    Avoids transposing NCL -> NLC, which causes strided memory access.
    Performs reduction on contiguous dimensions (L) instead.
    """
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


class GroupNormNCHW(nn.Module):
    """
    Optimized GroupNorm for NCHW layout.
    """
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


class GLUNCL(nn.Module):
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def __call__(self, x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=self.axis)
        return a * mx.sigmoid(b)


class GELUNCL(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(x)


class FusedGroupNormGELU(nn.Module):
    """Fused GroupNorm + GELU using a custom Metal kernel.

    Replaces the pattern: GELUNCL()(GroupNormNCL/NCHW(x))
    into a single kernel launch.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = mx.ones((num_channels,), dtype=mx.float32)
        self.bias = mx.zeros((num_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        from .metal_kernels import fused_groupnorm_gelu
        return fused_groupnorm_gelu(x, self.weight, self.bias, self.num_groups, self.eps)


class FusedGroupNormGLU(nn.Module):
    """Fused GroupNorm + GLU using a custom Metal kernel.

    Replaces the pattern: GLUNCL()(GroupNormNCL/NCHW(x))
    into a single kernel launch. Input has 2C channels, output has C channels.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)  # This is 2C (the input channels)
        self.eps = float(eps)
        self.weight = mx.ones((num_channels,), dtype=mx.float32)
        self.bias = mx.zeros((num_channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        from .metal_kernels import fused_groupnorm_glu
        return fused_groupnorm_glu(x, self.weight, self.bias, self.num_groups, self.eps)
