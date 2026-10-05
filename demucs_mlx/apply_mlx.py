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

from .defaults import DEFAULT_BATCH_SIZE
from .metal_kernels import fused_overlap_add
from .mlx_utils import center_trim, is_dconv_compile_enabled, outer_compile_context

_WEIGHT_CACHE: dict[tuple[int, float, str], mx.array] = {}
_COMPILED_FORWARDS: dict[int, tuple[weakref.ReferenceType, dict]] = {}


def _live(ref: "weakref.ReferenceType[tp.Any]") -> tp.Any:
    """Dereference a model weakref held by a compiled forward."""
    model = ref()
    if model is None:
        raise RuntimeError("The model behind a compiled forward was garbage-collected")
    return model


def _forward(
    model: tp.Any,
    x: mx.array,
    compile: tp.Optional[bool] = None,
    precomputed_conv: tp.Optional[mx.array] = None,
    **kwargs: tp.Any,
) -> mx.array:
    """Optionally compile repeated GPU forward shapes after an eager first call."""
    if compile is None:
        enabled = os.getenv("DEMUCS_MLX_COMPILE_FORWARD", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
    else:
        enabled = bool(compile)
    def _call(m, tensor):
        if precomputed_conv is not None:
            return m(tensor, precomputed_conv=precomputed_conv, **kwargs)
        return m(tensor, **kwargs)

    needs_ane_conv = getattr(model, "_ane_time_conv", None) is not None and precomputed_conv is None
    if not enabled or kwargs or needs_ane_conv:
        return _call(model, x)

    with outer_compile_context(True):
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
        conv_key = None
        if precomputed_conv is not None:
            conv_key = (tuple(precomputed_conv.shape), str(precomputed_conv.dtype))
        dconv_enabled = "1" if is_dconv_compile_enabled() else "0"
        key = (tuple(x.shape), str(x.dtype), conv_key, dconv_enabled)
        if key not in per_shape:
            # Spectral tuning may evaluate candidate kernels, which cannot happen
            # inside an MLX compile trace. This useful first call populates it.
            per_shape[key] = None
            return _call(model, x)

        compiled = per_shape[key]
        if compiled is None:
            if precomputed_conv is not None:
                compiled = mx.compile(lambda t, c, _ref=ref: _live(_ref)(t, precomputed_conv=c))
            else:
                compiled = mx.compile(lambda t, _ref=ref: _live(_ref)(t))
            per_shape[key] = compiled
        if precomputed_conv is not None:
            return compiled(x, precomputed_conv)
        return compiled(x)




class _StreamingOverlapAdd:
    """Overlap-add chunk outputs as they arrive, in bounded memory.

    Chunk k covers [k * stride, k * stride + segment). Once chunks 0..n-1 have
    arrived, no later chunk reaches below n * stride, so that span is final: it
    is normalized with the fused kernel, written to the output, and every chunk
    that cannot reach past it is released. At most the chunks of one batch plus
    the few that overlap the next span are held, instead of the whole track.

    Each output sample sums the same chunks in the same order as one kernel pass
    over every chunk, so the result is bit-identical to that.
    """

    def __init__(self, window: mx.array, stride: int, length: int, out_shape: tuple, dtype):
        self.window = window
        self.stride = int(stride)
        self.segment = int(window.shape[0])
        self.length = int(length)
        self.out = mx.zeros(out_shape, dtype=dtype)
        self.pending: list[mx.array] = []
        self.first_chunk = 0
        self.received = 0
        self.done = 0

    def add(self, frames: mx.array) -> None:
        """Queue chunk outputs shaped (n, *out_shape[:-1], segment), in chunk order."""
        self.pending.append(frames)
        self.received += int(frames.shape[0])

    def flush(self, final: bool = False) -> None:
        end = self.length if final else min(self.length, self.received * self.stride)
        if end <= self.done or not self.pending:
            return
        frames = mx.concatenate(self.pending, axis=0) if len(self.pending) > 1 else self.pending[0]
        base = self.first_chunk * self.stride
        span = fused_overlap_add(frames, self.window, self.stride, end - base)
        self.out[..., self.done:end] = span[..., self.done - base:]
        self.done = end
        keep_from = max(self.first_chunk, (end - self.segment) // self.stride + 1)
        frames = frames[keep_from - self.first_chunk:]
        self.first_chunk = keep_from
        self.pending = [frames] if frames.shape[0] else []

    def finish(self) -> mx.array:
        self.flush(final=True)
        return self.out


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
    _check_cancel: tp.Optional[tp.Callable[[], None]] = None,
):
    if _check_cancel is not None:
        _check_cancel()
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
                seed=seed, _rng=rng, compile=compile, _check_cancel=_check_cancel,
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
                compile=compile, _check_cancel=_check_cancel,
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
                compile=compile, _check_cancel=_check_cancel,
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
            # Cached arrays must be materialized: a lazy graph is bound to the
            # stream of the thread that built it.
            mx.eval(weight)
            _WEIGHT_CACHE[cache_key] = weight

        progress_bar = None
        if progress_enabled:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(offsets), desc="segments", unit="seg", leave=False)

        # --- BATCHING STATE ---
        if batch_size is None or str(batch_size).lower() == "auto":
            from .hardware import fit_batch_size, optimal_batch_size
            target_b = optimal_batch_size()
            effective_batch_size = fit_batch_size(len(offsets), target_b)
        else:
            effective_batch_size = int(batch_size)
            if effective_batch_size <= 0:
                raise ValueError("batch_size must be > 0.")

        from .mlx_htdemucs import HTDemucsMLX

        is_htdemucs = isinstance(model, HTDemucsMLX)
        if is_htdemucs:
            # valid_length() rejects a segment longer than the training length.
            model.valid_length(segment_length)

        def padded_length(chunk_len: int) -> int:
            """The input length upstream runs a chunk of chunk_len samples at.

            HTDemucs: the segment, zero-padded to the training length inside
            the model. Other models: their valid length for this chunk alone,
            so a short chunk is never padded with silence or neighbouring
            audio, which would change the model's normalization and context.
            """
            if is_htdemucs:
                return segment_length
            if hasattr(model, "valid_length"):
                return model.valid_length(chunk_len)
            return chunk_len

        # Check if ANE worker is present for pipelined prefetching
        ane_worker = getattr(model, "_ane_time_conv", None)
        if ane_worker is None and hasattr(model, "models") and len(model.models) > 0:
            ane_worker = getattr(model.models[0], "_ane_time_conv", None)

        # Batch consecutive chunks that run at the same input length.
        batches_indices = []
        current = []
        current_len = None
        for i, offset in enumerate(offsets):
            this_chunk_len = min(segment_length, length - offset)
            this_padded = padded_length(this_chunk_len)
            if current and (this_padded != current_len or len(current) >= effective_batch_size):
                batches_indices.append(current)
                current = []
            current.append((i, offset, this_chunk_len))
            current_len = this_padded
        if current:
            batches_indices.append(current)

        compile_enabled = (
            bool(compile)
            if compile is not None
            else os.getenv("DEMUCS_MLX_COMPILE_FORWARD", "1").strip().lower() not in {
                "0", "false", "no", "off",
            }
        )
        pad_tail = os.getenv("DEMUCS_MLX_PAD_TAIL", "0").strip().lower() not in {
            "0", "false", "no", "off",
        }

        def prepare_batch(group):
            inputs = []
            for i, offset, this_chunk_len in group:
                chunk = TensorChunk(mix_chunk, offset, this_chunk_len)
                padded = chunk.padded(padded_length(this_chunk_len))
                inputs.append(padded)
            actual_count = len(inputs)
            short_batch = len(batches_indices) > 1 and actual_count < effective_batch_size
            if compile_enabled and pad_tail and short_batch:
                while len(inputs) < effective_batch_size:
                    inputs.append(inputs[-1])
            stacked = mx.stack(inputs)
            b_seg, b_audio, ch, seg_len = stacked.shape
            flat = stacked.reshape(b_seg * b_audio, ch, seg_len)
            if ane_worker is not None:
                mx.eval(flat)
            return flat, group, actual_count, b_seg, b_audio

        ola = _StreamingOverlapAdd(
            weight, stride, length, (batch, len(model.sources), channels, length), mix_dtype
        )
        with outer_compile_context(compile_enabled):
            try:
                next_batch_data = None
                next_fut = None
                if batches_indices:
                    next_batch_data = prepare_batch(batches_indices[0])
                    if ane_worker is not None:
                        next_fut = ane_worker.submit(next_batch_data[0])

                for b_idx in range(len(batches_indices)):
                    if _check_cancel is not None:
                        _check_cancel()
                    assert next_batch_data is not None
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
                        conv_mx = conv  # the Core ML bridge returns MLX arrays
                        ane_worker.transfer_seconds += time.perf_counter() - transfer_start

                    batch_out_flat = _forward(
                        model, flat, compile=compile, precomputed_conv=conv_mx
                    )
                    _, sources, out_c, out_t = batch_out_flat.shape
                    batch_out = batch_out_flat.reshape(b_seg, b_audio, sources, out_c, out_t)

                    # Every overlap-add frame is exactly segment_length long so it
                    # lines up with the weight window. The model output can be
                    # longer (valid_length padding), so trim each chunk to its own
                    # length, as upstream does, then zero-pad short tail chunks;
                    # a tail chunk always ends at the input's end, so its padding
                    # falls outside the output and adds no weight.
                    if actual_count < b_seg:
                        batch_out = batch_out[:actual_count]
                    if all(cl == segment_length for _, _, cl in group):
                        ola.add(center_trim(batch_out, segment_length))
                    else:
                        for i in range(actual_count):
                            _, _, this_chunk_len = group[i]
                            chunk_out = center_trim(batch_out[i], this_chunk_len)
                            if this_chunk_len < segment_length:
                                chunk_out = mx.pad(
                                    chunk_out,
                                    [(0, 0), (0, 0), (0, 0), (0, segment_length - this_chunk_len)],
                                )
                            ola.add(chunk_out[None, ...])

                    if progress_bar is not None:
                        progress_bar.update(actual_count)

                    ola.flush()
                    mx.async_eval(ola.out)
            finally:
                if progress_bar is not None:
                    progress_bar.close()

        out = ola.finish()
        mx.eval(out)
        return out


    # No split path
    from .mlx_htdemucs import HTDemucsMLX

    if isinstance(model, HTDemucsMLX) and segment is not None:
        valid_length = int(segment * model.samplerate)
    elif hasattr(model, "valid_length"):
        valid_length = model.valid_length(length)
    else:
        valid_length = length
    padded_mix = mix_chunk.padded(valid_length)
    if _check_cancel is not None:
        _check_cancel()
    out = _forward(model, padded_mix, compile=compile)
    return center_trim(out, length)
