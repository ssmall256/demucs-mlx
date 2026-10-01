"""
MLX inference apply_model equivalent.
"""
from __future__ import annotations

import os
import random
import time
import typing as tp
import warnings
import weakref

import mlx.core as mx
from packaging import version

from .defaults import DEFAULT_BATCH_SIZE
from .mlx_utils import center_trim
from .metal_kernels import fused_overlap_add

_WEIGHT_CACHE: dict[tuple[int, float, str], mx.array] = {}
_COMPILED_FORWARDS: dict[int, tuple[weakref.ReferenceType, dict]] = {}


def _forward(
    model: tp.Any,
    x: mx.array,
    compile: tp.Optional[bool] = None,
    precomputed_conv: tp.Optional[mx.array] = None,
    **kwargs: tp.Any,
) -> mx.array:
    """Optionally compile repeated GPU forward shapes after an eager first call."""
    if compile is None:
        enabled = os.getenv("DEMUCS_MLX_COMPILE_FORWARD", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }
    else:
        enabled = bool(compile)
    def _call(m, tensor):
        if precomputed_conv is not None:
            return m(tensor, precomputed_conv=precomputed_conv, **kwargs)
        return m(tensor, **kwargs)

    if not enabled or kwargs or (getattr(model, "_ane_time_conv", None) is not None and precomputed_conv is None):
        return _call(model, x)

    model_id = id(model)
    slot = _COMPILED_FORWARDS.get(model_id)
    if slot is None or slot[0]() is not model:
        def forget(_ref, *, key=model_id):
            current = _COMPILED_FORWARDS.get(key)
            if current is not None and current[0] is _ref:
                _COMPILED_FORWARDS.pop(key, None)

        try:
            ref = weakref.ref(model, forget)
        except TypeError:
            return _call(model, x)
        slot = (ref, {})
        _COMPILED_FORWARDS[model_id] = slot
    ref, per_shape = slot
    conv_key = (tuple(precomputed_conv.shape), str(precomputed_conv.dtype)) if precomputed_conv is not None else None
    key = (tuple(x.shape), str(x.dtype), conv_key, os.getenv("DEMUCS_MLX_COMPILE_DCONV", "1"))
    if key not in per_shape:
        # Spectral tuning may evaluate candidate kernels, which cannot happen
        # inside an MLX compile trace. This useful first call populates it.
        per_shape[key] = None
        return _call(model, x)

    compiled = per_shape[key]
    if compiled is None:
        if precomputed_conv is not None:
            compiled = mx.compile(lambda t, c, _ref=ref: _ref()(t, precomputed_conv=c))
        else:
            compiled = mx.compile(lambda t, _ref=ref: _ref()(t))
        per_shape[key] = compiled
    if precomputed_conv is not None:
        return compiled(x, precomputed_conv)
    return compiled(x)


# MLX below 0.32.0 corrupts strided (non-leading-axis) slice scatter-add: the
# Metal slice_update kernel linearizes a 2-D/3-D dispatch grid incorrectly, so
# `arr.at[..., a:b].add(x)` silently accumulates into aliased cells and the
# overlap-add output spikes. Fixed in MLX 0.32.0. See
# https://github.com/ml-explore/mlx/issues/3676 (duplicate of #3477).
# This package floors MLX at 0.31.2, so the safe path below is always taken and
# the `.at(...).add(...)` branch is kept only as a reference implementation.
_USE_SAFE_SLICE_ACCUMULATION = version.parse(mx.__version__) >= version.parse("0.31.2")


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
    batch_size: tp.Union[int, str] = DEFAULT_BATCH_SIZE,
    seed: tp.Optional[int] = None,
    _rng: tp.Optional[random.Random] = None,
    *,
    source_index: tp.Optional[int] = None,
    compile: tp.Optional[bool] = None,
):
    progress_enabled = bool(progress)
    if num_workers > 0:
        warnings.warn("num_workers > 0 ignored on MLX.", RuntimeWarning)
        num_workers = 0

    rng: tp.Any
    if _rng is None:
        rng = random if seed is None else random.Random(int(seed))
    else:
        rng = _rng

    # --- BagOfModels Handling ---
    from .mlx_convert import BagOfModelsMLX
    if isinstance(model, BagOfModelsMLX):
        if source_index is not None:
            if not isinstance(source_index, int) or not 0 <= source_index < len(model.sources):
                raise ValueError("source_index must identify a source in the model")
            active = [
                index
                for index, weights in enumerate(model.weights)
                if weights[source_index] != 0
            ]
            if len(active) != 1:
                raise ValueError("Selected source requires exactly one contributing model")
            model_index = active[0]
            # Each earlier model would consume one random offset per shift.
            # Advance the same RNG so the chosen model receives its usual offsets.
            for earlier in model.models[:model_index]:
                for _ in range(shifts):
                    rng.randint(0, int(0.5 * earlier.samplerate))
            result = apply_model(
                model.models[model_index], mix, shifts, split, overlap,
                transition_power, progress, num_workers, segment, batch_size,
                seed=seed, _rng=rng, compile=compile,
            )
            for later in model.models[model_index + 1 :]:
                for _ in range(shifts):
                    rng.randint(0, int(0.5 * later.samplerate))
            selected = result[:, source_index : source_index + 1]
            weight = float(model.weights[model_index][source_index])
            return selected * weight / float(model.totals[source_index])
        totals = [0.0] * len(model.sources)
        estimates = None
        min_length = None

        for sub_model, model_weights in zip(model.models, model.weights):
            res = apply_model(
                sub_model, mix, shifts, split, overlap, transition_power,
                progress, num_workers, segment, batch_size, seed=seed, _rng=rng,
                compile=compile,
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
            mx.async_eval(estimates)

        # Vectorized normalization by totals.
        denom = mx.array(totals, dtype=estimates.dtype).reshape(1, -1, 1, 1)
        estimates = estimates / denom
        mx.eval(estimates)  # Final sync eval for BagOfModels path
        return estimates

    if source_index is not None:
        raise ValueError("Selected source requires a model bag")

    # --- Standard Inference ---
    mix_chunk = tensor_chunk(mix)
    batch, channels, length = mix_chunk.shape
    mix_dtype = mix_chunk.tensor.dtype

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        padded_mix = mix_chunk.padded(length + 2 * max_shift)
        padded_chunk = TensorChunk(padded_mix)
        out = 0.0
        for _ in range(shifts):
            offset = rng.randint(0, max_shift)
            shifted = TensorChunk(padded_chunk, offset, length + max_shift - offset)
            shifted_out = apply_model(
                model, shifted, 0, split, overlap, transition_power,
                False, num_workers, segment, batch_size, seed=seed, _rng=rng,
                compile=compile,
            )
            out = out + shifted_out[..., max_shift - offset:]
            mx.async_eval(out)
        out = out / shifts
        mx.eval(out)  # Final sync eval after all shifts
        return out

    if split:
        if segment is None:
            segment = model.segment
        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = list(range(0, length, stride))

        # Prepare Weight
        cache_key = (segment_length, float(transition_power), mix_dtype)
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
        if batch_size is None or str(batch_size).lower() == "auto":
            from .hardware import optimal_batch_size
            effective_batch_size = optimal_batch_size()
        else:
            effective_batch_size = int(batch_size)
            if effective_batch_size <= 0:
                raise ValueError("batch_size must be > 0.")

        if hasattr(model, "valid_length"):
            std_valid_len = model.valid_length(segment_length)
        else:
            std_valid_len = segment_length

        # Check if ANE worker is present for pipelined prefetching
        ane_worker = getattr(model, "_ane_time_conv", None)
        if ane_worker is None and hasattr(model, "models") and len(model.models) > 0:
            ane_worker = getattr(model.models[0], "_ane_time_conv", None)

        batches_indices = []
        current = []
        for i, offset in enumerate(offsets):
            this_chunk_len = min(segment_length, length - offset)
            current.append((i, offset, this_chunk_len))
            if len(current) >= effective_batch_size:
                batches_indices.append(current)
                current = []
        if current:
            batches_indices.append(current)

        def prepare_batch(group):
            inputs = []
            for i, offset, this_chunk_len in group:
                chunk = TensorChunk(mix_chunk, offset, this_chunk_len)
                padded = chunk.padded(std_valid_len)
                inputs.append(padded)
            actual_count = len(inputs)
            if compile and len(batches_indices) > 1 and actual_count < effective_batch_size:
                while len(inputs) < effective_batch_size:
                    inputs.append(inputs[-1])
            stacked = mx.stack(inputs)
            b_seg, b_audio, ch, l = stacked.shape
            flat = stacked.reshape(b_seg * b_audio, ch, l)
            if ane_worker is not None:
                mx.eval(flat)
            return flat, group, actual_count, b_seg, b_audio

        all_chunk_outputs = []
        try:
            next_batch_data = None
            next_fut = None
            if batches_indices:
                next_batch_data = prepare_batch(batches_indices[0])
                if ane_worker is not None:
                    next_fut = ane_worker.submit(next_batch_data[0])

            for b_idx in range(len(batches_indices)):
                flat, group, actual_count, b_seg, b_audio = next_batch_data
                curr_fut = next_fut

                if b_idx + 1 < len(batches_indices):
                    next_batch_data = prepare_batch(batches_indices[b_idx + 1])
                    if ane_worker is not None:
                        next_fut = ane_worker.submit(next_batch_data[0])
                else:
                    next_batch_data = None
                    next_fut = None

                conv_mx = None
                if curr_fut is not None:
                    wait_start = time.perf_counter()
                    conv = curr_fut.result()
                    ane_worker.wait_seconds += time.perf_counter() - wait_start
                    transfer_start = time.perf_counter()
                    conv_mx = mx.asarray(conv, copy=False)
                    ane_worker.transfer_seconds += time.perf_counter() - transfer_start

                batch_out_flat = _forward(model, flat, compile=compile, precomputed_conv=conv_mx)
                _, sources, out_c, out_t = batch_out_flat.shape
                batch_out = batch_out_flat.reshape(b_seg, b_audio, sources, out_c, out_t)

                if actual_count == b_seg and all(cl == out_t for _, _, cl in group):
                    all_chunk_outputs.append(batch_out)
                else:
                    for i in range(actual_count):
                        idx, offset, this_chunk_len = group[i]
                        chunk_out = batch_out[i : i + 1]
                        if this_chunk_len < out_t:
                            chunk_trimmed = center_trim(batch_out[i], this_chunk_len)
                            pad_r = out_t - this_chunk_len
                            chunk_out = mx.pad(
                                chunk_trimmed, [(0, 0), (0, 0), (0, 0), (0, pad_r)]
                            )[None, ...]
                        all_chunk_outputs.append(chunk_out)

                if progress_bar is not None:
                    progress_bar.update(actual_count)

                mx.async_eval(batch_out)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if all_chunk_outputs:
            stacked_frames = (
                mx.concatenate(all_chunk_outputs, axis=0)
                if len(all_chunk_outputs) > 1
                else all_chunk_outputs[0]
            )
            out = fused_overlap_add(stacked_frames, weight, stride, length)
        else:
            out = mx.zeros((batch, len(model.sources), channels, length), dtype=mix_dtype)
        mx.eval(out)
        return out


    # No split path
    valid_length = model.valid_length(length) if hasattr(model, "valid_length") else length
    padded_mix = mix_chunk.padded(valid_length)
    out = _forward(model, padded_mix, compile=compile)
    return center_trim(out, length)
