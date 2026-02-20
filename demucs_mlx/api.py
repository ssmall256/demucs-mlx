"""Public MLX API (Demucs-style)."""
from __future__ import annotations

import typing as tp
from pathlib import Path

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
        jobs: int = 0,
        progress: bool = False,
        batch_size: int = 8,
        callback: tp.Optional[tp.Callable[[dict], None]] = None,
        callback_arg: tp.Optional[dict] = None,
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
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0.")
        self.model_name = model
        self.shifts = int(shifts)
        self.overlap = float(overlap)
        self.split = split
        self.segment = float(segment) if segment is not None else None
        self.batch_size = int(batch_size)
        self.jobs = jobs
        self.progress = progress
        self.callback = callback
        self.callback_arg = callback_arg

        from .model_converter import get_mlx_model
        self._model = get_mlx_model(model)
        if hasattr(self._model, "eval"):
            self._model.eval()

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
        progress: tp.Optional[bool] = None,
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
        if progress is not None:
            self.progress = progress

    def _prepare_wav(self, wav):  # -> np.ndarray
        import numpy as np

        wav_np = np.asarray(wav)
        if wav_np.ndim != 2:
            raise ValueError("Expected wav with shape (channels, time).")
        if wav_np.shape[0] != self.audio_channels:
            if self.audio_channels == 1:
                wav_np = wav_np.mean(axis=0, keepdims=True)
            elif wav_np.shape[0] == 1 and self.audio_channels > 1:
                wav_np = np.tile(wav_np, (self.audio_channels, 1))
            elif wav_np.shape[0] > self.audio_channels:
                wav_np = wav_np[:self.audio_channels, :]
            else:
                raise ValueError(
                    f"Audio has {wav_np.shape[0]} channels but model expects {self.audio_channels}."
                )
        return wav_np

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
        import mlx.core as mx
        import numpy as np

        from .apply_mlx import apply_model

        if isinstance(wav, mx.array):
            wav_mx = self._prepare_wav_mx(wav)
            mix = wav_mx[None, ...]
        else:
            wav_np = self._prepare_wav(wav)
            wav_mx = mx.array(wav_np)
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
        )
        mx.eval(estimates)
        stems_mx = estimates[0]
        if return_mx:
            stems = {name: stems_mx[idx] for idx, name in enumerate(self._model.sources)}
            return wav_mx, stems
        wav_np = np.asarray(wav_mx)
        stems_np = np.asarray(stems_mx)
        stems = {name: stems_np[idx] for idx, name in enumerate(self._model.sources)}
        return wav_np, stems

    def separate_audio_file(
        self,
        path: tp.Union[str, Path],
        *,
        return_mx: bool = False,
    ) -> tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]:
        import mlx_audio_io as mac
        import numpy as np

        # mlx_audio_io.load returns (mx.array, sample_rate) in shape [frames, channels]
        audio_mx, sr = mac.load(str(path), dtype="float32")
        if sr != self.samplerate:
            # Resample to model sample rate via mlx-audio-io
            audio_mx, sr = mac.load(str(path), sr=self.samplerate, dtype="float32")
        # Convert to numpy and transpose to (channels, frames)
        wav = np.array(audio_mx, copy=False).T
        return self.separate_tensor(wav, return_mx=return_mx)


def save_audio(*args, **kwargs):
    from .audio import save_audio as _save_audio
    return _save_audio(*args, **kwargs)


def list_models() -> tp.Dict[str, tp.List[str]]:
    return {
        "single": [k for k, v in MLX_MODEL_REGISTRY.items() if not v.get("is_bag")],
        "bag": [k for k, v in MLX_MODEL_REGISTRY.items() if v.get("is_bag")],
    }
