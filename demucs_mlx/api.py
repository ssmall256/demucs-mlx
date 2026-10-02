"""Public MLX API (Demucs-style)."""
from __future__ import annotations

import typing as tp
from pathlib import Path

import mlx.core as mx

from .defaults import DEFAULT_BATCH_SIZE
from .mlx_registry import MLX_MODEL_REGISTRY


class Separator:
    def __init__(
        self,
        model: str = "htdemucs",
        repo: tp.Optional[Path] = None,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jobs: int = 0,
        progress: bool = False,
        batch_size: tp.Optional[int | str] = DEFAULT_BATCH_SIZE,
        callback: tp.Optional[tp.Callable[[dict], None]] = None,
        callback_arg: tp.Optional[dict] = None,
        ane_time_encoder: bool = False,
        stem: tp.Optional[str] = None,
        compile: tp.Optional[bool] = None,
        auto_tune: bool = False,
        attention_precision: tp.Optional[str] = None,
    ):
        if model not in MLX_MODEL_REGISTRY:
            known = ", ".join(sorted(MLX_MODEL_REGISTRY.keys()))
            raise ValueError(f"Unknown model '{model}'. Available: {known}")
        if repo is not None:
            raise NotImplementedError("Custom repos are not supported in MLX mode.")
        if jobs not in (0, 1):
            raise ValueError("MLX backend does not support multi-process jobs.")
        if callback is not None:
            raise NotImplementedError("Callbacks are not supported in MLX mode.")
        if int(shifts) < 0:
            raise ValueError("shifts must be >= 0.")
        if not (0.0 <= float(overlap) < 1.0):
            raise ValueError("overlap must be in [0, 1).")
        if segment is not None and float(segment) <= 0:
            raise ValueError("segment must be > 0 when provided.")
        if auto_tune or batch_size is None or str(batch_size).lower() == "auto":
            if ane_time_encoder:
                from .hardware import optimal_batch_size
                effective_batch_size = optimal_batch_size()
            else:
                effective_batch_size = "auto"
        elif int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        else:
            effective_batch_size = int(batch_size)
        if stem is not None and model != "htdemucs_ft":
            raise ValueError("Single-stem acceleration only supports htdemucs_ft")
        if ane_time_encoder:
            if model != "htdemucs":
                raise ValueError("ANE waveform path only supports the default htdemucs model")
            if segment is not None and float(segment) != 7.8:
                raise ValueError("ANE waveform path requires 7.8-second segments")
            if not split:
                raise ValueError("ANE waveform path requires split=True")
            if int(effective_batch_size) <= 0:
                raise ValueError("ANE waveform path requires batch_size > 0")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("seed must be an integer or None.") from exc
        self.model_name = model
        self.shifts = int(shifts)
        self.overlap = float(overlap)
        self.split = split
        self.segment = float(segment) if segment is not None else None
        self.seed = seed
        self.batch_size = effective_batch_size
        self.jobs = jobs
        self.progress = progress
        self.callback = callback
        self.callback_arg = callback_arg
        self._ane_requested = bool(ane_time_encoder)
        self.compile = compile
        self._closed = False

        from .model_converter import get_mlx_model
        self._model = get_mlx_model(model)
        if hasattr(self._model, "eval"):
            self._model.eval()
        # 'fp32' (default) or 'fp16' attention; see mlx_transformer.FastMultiHeadAttention.
        from .mlx_transformer import resolve_attention_dtype, set_attention_dtype

        attention_dtype = resolve_attention_dtype(attention_precision)
        for sub in getattr(self._model, "models", [self._model]):
            if hasattr(sub, "eval"):
                sub.eval()
            if hasattr(sub, "named_modules"):
                set_attention_dtype(sub, attention_dtype)
            ct = getattr(sub, "crosstransformer", None)
            if ct is not None:
                for layer in getattr(ct, "layers", []) + getattr(ct, "layers_t", []):
                    for a in ("attn", "cross_attn"):
                        m = getattr(layer, a, None)
                        if m is not None and hasattr(m, "_ensure_fused"):
                            m._ensure_fused()
            # Materialize the weights on this thread. MLX evaluates lazily on the
            # stream of the thread that built the graph, so a model loaded here
            # and first run on another thread would otherwise fail with "There is
            # no Stream(...) in current thread".
            if hasattr(sub, "parameters"):
                mx.eval(sub.parameters())
        if stem is not None and stem not in self._model.sources:
            raise ValueError(f"Unknown stem {stem!r}; available: {', '.join(self._model.sources)}")
        self.stem = stem
        self._source_index = self._model.sources.index(stem) if stem is not None else None
        self._ane_worker = None
        if ane_time_encoder:
            from .ane import WaveformConv

            if len(self._model.models) != 1:
                raise RuntimeError("ANE waveform path requires a single HTDemucs model")
            self._ane_worker = WaveformConv()
            self._model.models[0]._ane_time_conv = self._ane_worker

    def close(self) -> None:
        self._closed = True
        if self._ane_worker is not None:
            self._ane_worker.close()
            if getattr(self._model.models[0], "_ane_time_conv", None) is self._ane_worker:
                del self._model.models[0]._ane_time_conv
            self._ane_worker = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Separator is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def samplerate(self) -> int:
        return int(self._model.samplerate)

    @property
    def audio_channels(self) -> int:
        return int(self._model.audio_channels)

    @property
    def model(self):
        return self._model

    def update_parameter(
        self,
        *,
        shifts: tp.Optional[int] = None,
        overlap: tp.Optional[float] = None,
        split: tp.Optional[bool] = None,
        segment: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        progress: tp.Optional[bool] = None,
        compile: tp.Optional[bool] = None,
    ) -> None:
        if shifts is not None:
            if int(shifts) < 0:
                raise ValueError("shifts must be >= 0.")
            self.shifts = int(shifts)
        if overlap is not None:
            overlap_f = float(overlap)
            if not (0.0 <= overlap_f < 1.0):
                raise ValueError("overlap must be in [0, 1).")
            self.overlap = overlap_f
        if split is not None:
            self.split = split
        if segment is not None:
            seg_f = float(segment)
            if seg_f <= 0:
                raise ValueError("segment must be > 0 when provided.")
            self.segment = seg_f
        if seed is not None:
            try:
                self.seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("seed must be an integer or None.") from exc
        if progress is not None:
            self.progress = progress
        if compile is not None:
            self.compile = bool(compile)

    def _prepare_wav_mx(self, wav):
        import mlx.core as mx

        if wav.ndim != 2:
            raise ValueError("Expected wav with shape (channels, time).")
        if int(wav.shape[0]) != self.audio_channels:
            if self.audio_channels == 1:
                wav = mx.mean(wav, axis=0, keepdims=True)
            elif int(wav.shape[0]) == 1 and self.audio_channels > 1:
                wav = mx.broadcast_to(wav, (self.audio_channels, int(wav.shape[1])))
            elif int(wav.shape[0]) > self.audio_channels:
                wav = wav[:self.audio_channels, :]
            else:
                raise ValueError(
                    f"Audio has {int(wav.shape[0])} channels "
                    f"but model expects {self.audio_channels}."
                )
        return wav

    def separate_tensor(
        self,
        wav,
        *,
        return_mx: bool = False,
    ) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
        if self._ane_requested and self._closed:
            raise RuntimeError("ANE Separator is closed")
        import mlx.core as mx

        from .apply_mlx import apply_model

        if not isinstance(wav, mx.array):
            # Any DLPack or buffer-protocol array is imported without a copy.
            wav = mx.asarray(wav)
        wav_mx = self._prepare_wav_mx(wav)
        mix = wav_mx[None, ...]
        estimates = apply_model(
            self._model,
            mix,
            shifts=self.shifts,
            split=self.split,
            overlap=self.overlap,
            segment=self.segment,
            progress=self.progress,
            batch_size=self.batch_size,
            seed=self.seed,
            source_index=self._source_index,
            compile=self.compile,
        )
        mx.eval(estimates)
        stems_mx = estimates[0]
        names = (self.stem,) if self.stem is not None else self._model.sources
        if return_mx:
            stems = {name: stems_mx[idx] for idx, name in enumerate(names)}
            return wav_mx, stems
        # NumPy output is this API's default for non-MLX callers.
        import numpy as np

        wav_np = np.asarray(wav_mx)
        stems_np = np.asarray(stems_mx)
        stems = {name: stems_np[idx] for idx, name in enumerate(names)}
        return wav_np, stems

    def separate_audio_file(
        self,
        path: tp.Union[str, Path],
        *,
        return_mx: bool = False,
    ) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
        from .audio import load_audio

        audio_mx, sr = load_audio(path, sr=self.samplerate, dtype="float32")
        return self.separate_tensor(audio_mx, return_mx=return_mx)

    def separate(
        self,
        audio_or_path: tp.Union[str, Path, tp.Any],
        *,
        output_dir: tp.Optional[tp.Union[str, Path]] = None,
        return_mx: bool = False,
        async_write: bool = True,
        filename_format: str = "{stem}.wav",
        clip: tp.Literal["rescale", "clamp", "tanh", "none"] = "rescale",
        bits_per_sample: tp.Literal[16, 24, 32] = 16,
        as_float: bool = False,
    ) -> tp.Union[tp.Tuple[tp.Any, tp.Dict[str, tp.Any]], tp.Dict[str, Path]]:
        """Unified separation entry point accepting file path or in-memory audio tensor.

        Args:
            audio_or_path: Path to audio file or in-memory tensor (mx.array or any
                DLPack/buffer-protocol array).
            output_dir: Optional destination directory to persist separated stems.
            return_mx: If True and output_dir is None, return MLX arrays instead of NumPy.
            async_write: If True and output_dir is provided, write stems asynchronously.
            filename_format: Format string for saved stems, e.g. "{stem}.wav" or
                "{track}_{stem}.wav".
            clip: Clipping mode ("rescale", "clamp", "tanh", "none").
            bits_per_sample: Bit depth for saved audio (16, 24, 32).
            as_float: Save as float32 audio.

        Returns:
            If output_dir is None: (mix, {stem_name: stem_audio})
            If output_dir is provided: {stem_name: saved_path}
        """
        from .audio import AsyncAudioWriter, save_audio

        want_mx = return_mx or (output_dir is not None)
        if isinstance(audio_or_path, (str, Path)):
            track_name = Path(audio_or_path).stem
            wav, stems = self.separate_audio_file(audio_or_path, return_mx=want_mx)
        else:
            track_name = "track"
            wav, stems = self.separate_tensor(audio_or_path, return_mx=want_mx)

        if output_dir is None:
            return wav, stems

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}

        if stems and isinstance(next(iter(stems.values())), mx.array):
            mx.eval(*stems.values())

        if async_write:
            with AsyncAudioWriter(
                clip=clip,
                bits_per_sample=bits_per_sample,
                as_float=as_float,
            ) as writer:
                for stem_name, stem_wav in stems.items():
                    filename = filename_format.format(stem=stem_name, track=track_name)
                    dest = out_dir / filename
                    writer.submit(stem_wav, dest, self.samplerate)
                    saved_paths[stem_name] = dest
        else:
            for stem_name, stem_wav in stems.items():
                filename = filename_format.format(stem=stem_name, track=track_name)
                dest = out_dir / filename
                save_audio(
                    stem_wav,
                    dest,
                    samplerate=self.samplerate,
                    clip=clip,
                    bits_per_sample=bits_per_sample,
                    as_float=as_float,
                )
                saved_paths[stem_name] = dest

        return saved_paths


def save_audio(*args, **kwargs):
    from .audio import save_audio as _save_audio
    return _save_audio(*args, **kwargs)


def list_models() -> tp.Dict[str, tp.List[str]]:
    return {
        "single": [k for k, v in MLX_MODEL_REGISTRY.items() if not v.get("is_bag")],
        "bag": [k for k, v in MLX_MODEL_REGISTRY.items() if v.get("is_bag")],
    }
