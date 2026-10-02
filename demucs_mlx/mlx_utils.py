from __future__ import annotations

import contextlib
import contextvars
import math
import os
import threading
import typing as tp

import mlx.core as mx

_OUTER_COMPILE_ACTIVE = contextvars.ContextVar("outer_compile_active", default=False)


def is_outer_compile_active() -> bool:
    """Return whether outer forward graph compilation is currently active."""
    return _OUTER_COMPILE_ACTIVE.get()


@contextlib.contextmanager
def outer_compile_context(active: bool = True):
    """Context manager setting whether outer forward graph compilation is active."""
    token = _OUTER_COMPILE_ACTIVE.set(bool(active))
    try:
        yield
    finally:
        _OUTER_COMPILE_ACTIVE.reset(token)


def is_dconv_compile_enabled() -> bool:
    """Check if DConv layers should compile their internal blocks into separate subgraphs.

    If DEMUCS_MLX_COMPILE_DCONV is explicitly set in the environment, that takes precedence.
    Otherwise, if outer forward compilation is active, DConv compilation is automatically
    suppressed so MLX can fuse kernels globally across the entire model graph in a single trace.
    """
    env = os.getenv("DEMUCS_MLX_COMPILE_DCONV")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    if is_outer_compile_active():
        return False
    return True


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


_THREAD_STREAMS = threading.local()


def thread_side_stream() -> mx.Stream:
    """A secondary stream on the default device for the calling thread.

    MLX streams belong to the thread that created them, so a stream stored on a
    model fails ("There is no Stream(gpu, N) in current thread") as soon as the
    model is used from another thread. Keep one per thread instead.
    """
    device = mx.default_device()
    streams = getattr(_THREAD_STREAMS, "streams", None)
    if streams is None:
        streams = _THREAD_STREAMS.streams = {}
    key = str(device)
    stream = streams.get(key)
    if stream is None:
        stream = streams[key] = mx.new_stream(device)
    return stream
