import typing as tp
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_audio(path, *, sr: int, layout: str = "channels_first", dtype: str = "float32"):
    """Load through mlx-audio-io with a useful error for a broken native binding."""
    import mlx_audio_io as mac

    try:
        return mac.load(str(path), sr=sr, layout=layout, dtype=dtype)
    except TypeError as exc:
        if "Unable to convert function return value to a Python type" not in str(exc):
            raise
        try:
            from mlx_audio_io._native_loader import load_build_info

            build = load_build_info()
            pairing = (
                f" (built for MLX {build.get('build_mlx_version')} with "
                f"nanobind {build.get('build_nanobind_version')})"
            )
        except (AttributeError, ImportError, OSError, TypeError, ValueError):
            pairing = ""
        raise RuntimeError(
            "mlx-audio-io could not return an MLX array"
            f"{pairing}. Rebuild it with the nanobind version used by the "
            "installed MLX runtime."
        ) from exc


def prevent_clip(wav, mode='rescale'):
    """Prevent clipping in torch tensors."""
    import torch
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _prevent_clip_numpy(wav: np.ndarray, mode: str):
    """Prevent clipping in numpy arrays."""
    if mode is None or mode == 'none':
        return wav
    if not np.issubdtype(wav.dtype, np.floating):
        raise AssertionError("too late for clipping")
    if mode == 'rescale':
        max_val = float(np.max(np.abs(wav)))
        scale = max(1.01 * max_val, 1.0)
        wav = wav / scale
    elif mode == 'clamp':
        wav = np.clip(wav, -0.99, 0.99)
    elif mode == 'tanh':
        wav = np.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _prevent_clip_mlx(wav: mx.array, mode: str):
    """Prevent clipping using MLX ops (keeps data on GPU)."""
    if mode is None or mode == 'none':
        return wav
    if mode == 'rescale':
        max_val = mx.max(mx.abs(wav))
        scale = mx.maximum(1.01 * max_val, 1.0)
        wav = wav / scale
    elif mode == 'clamp':
        wav = mx.clip(wav, -0.99, 0.99)
    elif mode == 'tanh':
        wav = mx.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav,
               path: tp.Union[str, Path],
               samplerate: int,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False,
               layout: str = "channels_first"):
    """
    Save audio file using mlx_audio_io.
    Supports np.ndarray, mlx.core.array, and torch.Tensor.
    """
    import mlx_audio_io as mac
    path = Path(path)

    # Determine encoding
    if as_float:
        encoding = "float32"
    else:
        encoding = "pcm16" if bits_per_sample == 16 else "float32"

    save_layout = layout if getattr(wav, "ndim", 2) > 1 else "channels_last"

    # --- MLX HANDLING (Optimized) ---
    if isinstance(wav, mx.array):
        wav_mx = _prevent_clip_mlx(wav, mode=clip)
        mx.eval(wav_mx)
        mac.save(str(path), wav_mx, samplerate, layout=save_layout, encoding=encoding, clip=(clip != 'none'))
    # --- NUMPY HANDLING ---
    elif isinstance(wav, np.ndarray):
        wav_np = wav
        if np.issubdtype(wav_np.dtype, np.floating):
            wav_np = _prevent_clip_numpy(wav_np, mode=clip)
        mac.save(str(path), wav_np, samplerate, layout=save_layout, encoding=encoding, clip=False)
    # --- TORCH HANDLING (lazy import) ---
    else:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = prevent_clip(wav, mode=clip)
            wav_np = wav.detach().cpu().numpy()
            mac.save(str(path), wav_np, samplerate, layout=save_layout, encoding=encoding, clip=False)
        else:
            raise TypeError(f"Unsupported audio type: {type(wav)}")


class AsyncAudioWriter:
    """Non-blocking background thread pool for audio file serialization."""

    def __init__(
        self,
        maxsize: int = 4,
        workers: int = 2,
        *,
        clip: tp.Literal["rescale", "clamp", "tanh", "none"] = "rescale",
        bits_per_sample: tp.Literal[16, 24, 32] = 16,
        as_float: bool = False,
    ):
        import queue
        import threading

        if workers <= 0:
            raise ValueError("workers must be > 0")
        self._queue: queue.Queue[tp.Optional[tuple]] = queue.Queue(maxsize=maxsize)
        self._error: tp.Optional[BaseException] = None
        self._workers = int(workers)
        self._clip = clip
        self._bits_per_sample = int(bits_per_sample)
        self._as_float = bool(as_float)
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name=f"demucs-writer-{i}")
            for i in range(self._workers)
        ]
        for thread in self._threads:
            thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._queue.task_done()
                    break
                wav, path, samplerate = item
                save_audio(
                    wav,
                    path,
                    samplerate=samplerate,
                    clip=self._clip,
                    bits_per_sample=self._bits_per_sample,
                    as_float=self._as_float,
                )
            except BaseException as exc:
                self._error = exc
            finally:
                if item is not None:
                    self._queue.task_done()

    def submit(self, wav: tp.Any, path: tp.Union[str, Path], samplerate: int) -> None:
        if self._error is not None:
            raise self._error
        if isinstance(wav, mx.array):
            mx.eval(wav)
        self._queue.put((wav, path, samplerate))

    def close(self) -> None:
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

