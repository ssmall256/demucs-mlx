"""
MLX inference apply_model equivalent.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import random
import typing as tp

import mlx.core as mx

from .mlx_utils import center_trim


class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)


class TensorChunk:
    def __init__(self, tensor: mx.array, offset=0, length=None):
        total_length = tensor.shape[-1]
        if offset < 0 or offset >= total_length:
            raise ValueError("Invalid offset.")
        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)
        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        if delta < 0:
            raise ValueError("target_length must be >= length.")
        total_length = self.tensor.shape[-1]
        start = self.offset - delta // 2
        end = start + target_length
        correct_start = max(0, start)
        correct_end = min(total_length, end)
        pad_left = correct_start - start
        pad_right = end - correct_end
        out = self.tensor[..., correct_start:correct_end]
        if pad_left or pad_right:
            out = mx.pad(out, [(0, 0), (0, 0), (pad_left, pad_right)], mode="constant")
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    if not isinstance(tensor_or_chunk, mx.array):
        raise TypeError("Expected mx.array.")
    return TensorChunk(tensor_or_chunk)


def apply_model(
    model,
    mix: tp.Union[mx.array, TensorChunk],
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    num_workers: int = 0,
    segment: tp.Optional[float] = None,
):
    if progress:
        raise RuntimeError("progress not supported in MLX apply_model yet.")
    if num_workers > 0:
        pool = ThreadPoolExecutor(num_workers)
    else:
        pool = DummyPoolExecutor()
    if transitions_power := transition_power < 1:
        raise ValueError("transition_power < 1 leads to weird behavior.")

    if isinstance(model, (list, tuple)):
        raise RuntimeError("BagOfModels MLX not supported yet.")

    mix_array = mix.tensor if isinstance(mix, TensorChunk) else mix
    batch, channels, length = mix_array.shape

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix_array)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(
                model,
                shifted,
                shifts=0,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                progress=False,
                num_workers=num_workers,
                segment=segment,
            )
            out = out + shifted_out[..., max_shift - offset:]
        return out / shifts
    if split:
        out = mx.zeros((batch, len(model.sources), channels, length), dtype=mix_array.dtype)
        sum_weight = mx.zeros((length,), dtype=mix_array.dtype)
        if segment is None:
            segment = model.segment
        if segment is None or segment <= 0:
            raise ValueError("segment must be > 0")
        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = list(range(0, length, stride))
        weight = mx.concatenate(
            [
                mx.arange(1, segment_length // 2 + 1),
                mx.arange(segment_length - segment_length // 2, 0, -1),
            ],
            axis=0,
        )
        weight = (weight / mx.max(weight)) ** transition_power
        for offset in offsets:
            chunk = TensorChunk(mix_array, offset, segment_length)
            valid_len = model.valid_length(chunk.length) if hasattr(model, "valid_length") else chunk.length
            padded = chunk.padded(valid_len)
            chunk_out = model(padded)
            chunk_out = center_trim(chunk_out, chunk.length)
            end = offset + chunk.length
            out = out.at[..., offset:end].add(weight[:chunk.length] * chunk_out)
            sum_weight = sum_weight.at[offset:end].add(weight[:chunk.length])
        out = out / sum_weight
        return out

    valid_length = length
    if hasattr(model, "valid_length"):
        valid_length = model.valid_length(length)
    mix = tensor_chunk(mix_array)
    padded_mix = mix.padded(valid_length)
    out = model(padded_mix)
    return center_trim(out, length)

