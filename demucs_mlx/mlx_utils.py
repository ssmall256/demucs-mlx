from __future__ import annotations

import math
import typing as tp

import mlx.core as mx


class MLXStateDictMixin:
    """Mixin to add PyTorch-style state dict methods to MLX models."""

    def state_dict(self):
        """Return state dict (compatible with PyTorch interface)."""
        return dict(self.parameters())

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into model (PyTorch-compatible interface)."""
        # MLX update handles keys gracefully; simple wrapper is fine.
        self.update(state_dict)

def center_trim(x: mx.array, reference: tp.Union[mx.array, int]) -> mx.array:
    if isinstance(reference, mx.array):
        ref_size = reference.shape[-1]
    else:
        ref_size = int(reference)
    delta = x.shape[-1] - ref_size
    if delta < 0:
        raise ValueError(f"tensor must be larger than reference. Delta is {delta}.")
    if delta:
        start = delta // 2
        end = x.shape[-1] - (delta - start)
        # Slicing is zero-copy in MLX
        x = x[..., start:end]
    return x

def unfold(x: mx.array, kernel_size: int, stride: int) -> mx.array:
    *shape, length = x.shape
    n_frames = int(math.ceil(length / stride))
    tgt_length = (n_frames - 1) * stride + kernel_size
    pad = tgt_length - length
    
    if pad > 0:
        # Pad only the last dimension
        pads = [(0, 0)] * len(shape) + [(0, pad)]
        x = mx.pad(x, pads, mode="constant")
    
    # Ensure memory is contiguous before stride tricks
    x = mx.contiguous(x)
    
    # Calculate strides for the last dimension
    # MLX arrays are row-major. The stride of the last dimension is 1.
    # The stride of the second to last is x.shape[-1], etc.
    
    # We want to view the last dim 'L' as (n_frames, kernel_size)
    # The stride for 'kernel_size' is 1 (element-wise).
    # The stride for 'n_frames' is 'stride' elements.
    
    # Existing strides for shape [*shape, length]
    # We reconstruct strides manually to ensure robustness
    current_strides = [1] * x.ndim
    for i in range(x.ndim - 2, -1, -1):
        current_strides[i] = current_strides[i + 1] * x.shape[i + 1]
        
    # Construct new strides
    # Original: [...batch_strides, 1]
    # New:      [...batch_strides, stride * 1, 1]
    new_strides = current_strides[:-1] + [stride, 1]
    
    return mx.as_strided(
        x,
        shape=[*shape, n_frames, kernel_size],
        strides=new_strides,
    )
