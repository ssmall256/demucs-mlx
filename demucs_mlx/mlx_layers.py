"""
Shared MLX layers and NCL/NCHW wrappers.
Optimized for memory layout efficiency.
"""
from __future__ import annotations

import typing as tp

import mlx.core as mx
import mlx.nn as nn


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
        y = self.conv(x)
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

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)
        y = self.conv(x)
        return y.transpose(0, 2, 1)


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
        y = self.conv(x)
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

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 3, 1)
        y = self.conv(x)
        return y.transpose(0, 3, 1, 2)


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
        B, C = x.shape[0], x.shape[1]
        G = self.num_groups
        if C % G != 0:
            raise ValueError(f"num_channels {C} not divisible by num_groups {G}")
        
        # Reshape (N, C, L) -> (N, G, C//G, L)
        # We split the channel dim (1).
        # Since memory is likely N-C-L, this is a metadata view.
        x_reshaped = x.reshape(B, G, C // G, *x.shape[2:])
        
        # Calculate stats over (C//G, L).
        # L is the last dim (contiguous), so this reduction is fast.
        axes = tuple(range(2, x_reshaped.ndim))
        mean = x_reshaped.mean(axis=axes, keepdims=True)
        var = ((x_reshaped - mean) ** 2).mean(axis=axes, keepdims=True)
        
        # Normalize
        x_norm = (x_reshaped - mean) * mx.rsqrt(var + self.eps)
        
        # Restore (N, C, L)
        x_out = x_norm.reshape(x.shape)
        
        if self.affine:
            # Broadcast weight/bias: (1, C, 1)
            shape = [1, C] + [1] * (x_out.ndim - 2)
            x_out = x_out * self.weight.reshape(shape) + self.bias.reshape(shape)
            
        return x_out


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
        B, C = x.shape[0], x.shape[1]
        G = self.num_groups
        
        # Reshape (N, C, H, W) -> (N, G, C//G, H, W)
        x_reshaped = x.reshape(B, G, C // G, *x.shape[2:])
        
        axes = tuple(range(2, x_reshaped.ndim))
        mean = x_reshaped.mean(axis=axes, keepdims=True)
        var = ((x_reshaped - mean) ** 2).mean(axis=axes, keepdims=True)
        
        x_norm = (x_reshaped - mean) * mx.rsqrt(var + self.eps)
        x_out = x_norm.reshape(x.shape)
        
        if self.affine:
            shape = [1, C] + [1] * (x_out.ndim - 2)
            x_out = x_out * self.weight.reshape(shape) + self.bias.reshape(shape)
            
        return x_out


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
