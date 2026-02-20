"""
MLX inference apply_model equivalent.
"""
from __future__ import annotations

import random
import typing as tp
import warnings

import mlx.core as mx

from .mlx_utils import center_trim

_WEIGHT_CACHE: dict[tuple[int, float, str], mx.array] = {}


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
    mix: tp.Union[mx.array, "TensorChunk"],
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    num_workers: int = 0,
    segment: tp.Optional[float] = None,
    batch_size: int = 10,
):
    progress_enabled = bool(progress)
    if num_workers > 0:
        warnings.warn("num_workers > 0 ignored on MLX.", RuntimeWarning)
        num_workers = 0

    # --- BagOfModels Handling ---
    from .mlx_convert import BagOfModelsMLX
    if isinstance(model, BagOfModelsMLX):
        totals = [0.0] * len(model.sources)
        estimates = None
        min_length = None

        for sub_model, model_weights in zip(model.models, model.weights):
            res = apply_model(
                sub_model, mix, shifts, split, overlap, transition_power,
                progress, num_workers, segment, batch_size
            )
            out = mx.array(res)

            # Vectorized per-source weighting: (1, S, 1, 1)
            w = mx.array(model_weights, dtype=out.dtype).reshape(1, -1, 1, 1)
            out = out * w

            # Track totals in Python (tiny), keep math vectorized above.
            for k, inst_weight in enumerate(model_weights):
                totals[k] += float(inst_weight)

            if min_length is None:
                min_length = out.shape[-1]
                estimates = out
            else:
                if out.shape[-1] < min_length:
                    min_length = out.shape[-1]
                    estimates = estimates[..., :min_length]
                elif out.shape[-1] > min_length:
                    out = out[..., :min_length]
                estimates = estimates + out

        # Vectorized normalization by totals.
        denom = mx.array(totals, dtype=estimates.dtype).reshape(1, -1, 1, 1)
        estimates = estimates / denom
        mx.eval(estimates)  # Single eval for entire BagOfModels path
        return estimates

    # --- Standard Inference ---
    mix_array = mix.tensor if hasattr(mix, "tensor") else mix
    batch, channels, length = mix_array.shape

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix_chunk = TensorChunk(mix_array)
        padded_mix = mix_chunk.padded(length + 2 * max_shift)
        out = 0.0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(
                model, shifted, 0, split, overlap, transition_power,
                False, num_workers, segment, batch_size
            )
            out = out + shifted_out[..., max_shift - offset:]
        out = out / shifts
        mx.eval(out)  # Single eval after all shifts
        return out

    if split:
        out = mx.zeros((batch, len(model.sources), channels, length), dtype=mix_array.dtype)
        sum_weight = mx.zeros((length,), dtype=mix_array.dtype)

        if segment is None:
            segment = model.segment
        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = list(range(0, length, stride))

        # Prepare Weight
        cache_key = (segment_length, float(transition_power), mix_array.dtype)
        weight = _WEIGHT_CACHE.get(cache_key)
        if weight is None:
            weight = mx.concatenate([
                mx.arange(1, segment_length // 2 + 1),
                mx.arange(segment_length - segment_length // 2, 0, -1),
            ], axis=0)
            weight = (weight / mx.max(weight)) ** transition_power
            _WEIGHT_CACHE[cache_key] = weight

        progress_bar = None
        if progress_enabled:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(offsets), desc="segments", unit="seg", leave=False)

        # --- BATCHING STATE ---
        batch_inputs = []
        batch_indices = []
        if hasattr(model, "valid_length"):
            std_valid_len = model.valid_length(segment_length)
        else:
            std_valid_len = segment_length

        def flush_batch():
            nonlocal batch_inputs, batch_indices, out, sum_weight
            if not batch_inputs:
                return

            # 1. Stack: (Batch_Segments, Audio_Batch, Channels, Time)
            batch_tensor = mx.stack(batch_inputs)

            # 2. Reshape: Flatten Segments into Batch -> (Total_Batch, Channels, Time)
            b_seg, b_audio, channels, length = batch_tensor.shape
            batch_tensor_flat = batch_tensor.reshape(b_seg * b_audio, channels, length)

            # 3. Run Model (Standard 3D Input)
            batch_out_flat = model(batch_tensor_flat)

            # 4. Unflatten: (Batch_Segments, Audio_Batch, Sources, Channels, Time)
            _, sources, out_c, out_t = batch_out_flat.shape
            batch_out = batch_out_flat.reshape(b_seg, b_audio, sources, out_c, out_t)

            for i, idx in enumerate(batch_indices):
                chunk_out = center_trim(batch_out[i], segment_length)

                offset = offsets[idx]
                end = offset + segment_length

                out = out.at[:, :, :, offset:end].add(weight.reshape(1, 1, 1, -1) * chunk_out)
                sum_weight = sum_weight.at[offset:end].add(weight)

                if progress_bar is not None:
                    progress_bar.update(1)

            mx.eval(out, sum_weight)
            batch_inputs = []
            batch_indices = []

        try:
            for i, offset in enumerate(offsets):
                this_chunk_len = min(segment_length, length - offset)
                chunk = TensorChunk(mix_array, offset, this_chunk_len)

                # Batch only standard-sized chunks
                if this_chunk_len == segment_length:
                    padded = chunk.padded(std_valid_len)
                    batch_inputs.append(padded)
                    batch_indices.append(i)

                    if len(batch_inputs) >= batch_size:
                        flush_batch()
                else:
                    # Flush pending batch
                    flush_batch()

                    # Run odd-sized chunk individually
                    if hasattr(model, "valid_length"):
                        valid_len = model.valid_length(this_chunk_len)
                    else:
                        valid_len = this_chunk_len
                    padded = chunk.padded(valid_len)

                    # FIX: Pass 'padded' directly. It is already (Batch, Channels, Time).
                    chunk_out = model(padded)
                    chunk_out = center_trim(chunk_out, this_chunk_len)

                    end = offset + this_chunk_len
                    w = weight[:this_chunk_len].reshape(1, 1, 1, -1)
                    out = out.at[:, :, :, offset:end].add(w * chunk_out)
                    sum_weight = sum_weight.at[offset:end].add(weight[:this_chunk_len])
                    mx.eval(out, sum_weight)  # Eval to bound graph size
                    if progress_bar is not None:
                        progress_bar.update(1)

            flush_batch()
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if bool(mx.any(sum_weight == 0).item()):
            raise ValueError("sum_weight has zeros; check segment and overlap settings")

        out = out / sum_weight
        mx.eval(out)
        return out

    # No split path
    valid_length = model.valid_length(length) if hasattr(model, "valid_length") else length
    mix_chunk = TensorChunk(mix_array)
    padded_mix = mix_chunk.padded(valid_length)
    out = model(padded_mix)
    return center_trim(out, length)
