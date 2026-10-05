import typing as tp
from pathlib import Path

import mlx.core as mx


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


def prevent_clip(wav, mode="rescale"):
    """Prevent clipping in torch tensors."""
    import torch

    if mode is None or mode == "none":
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == "rescale":
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == "clamp":
        wav = wav.clamp(-0.99, 0.99)
    elif mode == "tanh":
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _prevent_clip_mlx(wav: mx.array, mode: str):
    """Prevent clipping using MLX ops (keeps data on GPU)."""
    if mode is None or mode == "none":
        return wav
    if mode == "rescale":
        max_val = mx.max(mx.abs(wav))
        scale = mx.maximum(1.01 * max_val, 1.0)
        wav = wav / scale
    elif mode == "clamp":
        wav = mx.clip(wav, -0.99, 0.99)
    elif mode == "tanh":
        wav = mx.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def _save_prepared(
    wav, path, samplerate, *, layout, encoding, clip, check=None, prepared_save=None
):
    """Native encoding followed by atomic publication, including sync callers."""
    import os
    import uuid

    import mlx_audio_io as mac

    destination = Path(path)
    temporary = destination.with_name(
        "." + destination.stem + "." + uuid.uuid4().hex + ".tmp" + destination.suffix
    )
    try:
        if check is not None:
            check()
        if prepared_save is None:
            mac.save(str(temporary), wav, samplerate, layout=layout, encoding=encoding, clip=clip)
        else:
            prepared_save(str(temporary), wav, samplerate, encoding=encoding, clip=clip)
        if check is not None:
            check()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_audio(
    wav,
    path: tp.Union[str, Path],
    samplerate: int,
    clip: tp.Literal["rescale", "clamp", "tanh", "none"] = "rescale",
    bits_per_sample: tp.Literal[16, 24, 32] = 16,
    as_float: bool = False,
    layout: str = "channels_first",
):
    """
    Save audio file using mlx_audio_io.
    Accepts an mlx.core.array, a torch.Tensor, or any DLPack/buffer-protocol
    array; non-MLX input is imported without a copy.
    """
    path = Path(path)

    # Determine encoding
    if as_float or bits_per_sample == 32:
        encoding = "float32"
    elif bits_per_sample == 24:
        encoding = "pcm24"
    else:
        encoding = "pcm16"

    if not isinstance(wav, mx.array):
        if type(wav).__module__.split(".")[0] == "torch":
            wav = wav.detach().cpu()
        wav = mx.asarray(wav)
    if not mx.issubdtype(wav.dtype, mx.floating):
        raise TypeError(f"Expected floating-point audio, got {wav.dtype}")

    save_layout = layout if wav.ndim > 1 else "channels_last"
    wav_mx = _prevent_clip_mlx(wav, mode=clip)
    mx.eval(wav_mx)
    _save_prepared(
        wav_mx, path, samplerate, layout=save_layout, encoding=encoding, clip=(clip != "none")
    )


class _WriteGroup:
    """One complete output owner, retained until preparation and writers finish."""

    def __init__(self, writer, owner, owner_bytes, serial):
        import threading

        self.writer, self.owner, self.serial = writer, owner, serial
        self._lock = threading.Lock()
        self._references = 1
        self._lease = None if serial else writer._control.reserve(owner_bytes)
        if not serial and self._lease is None:
            raise ValueError("output owner exceeds the I/O budget; use serial export")

    def retain(self):
        with self._lock:
            self._references += 1

    def release(self):
        with self._lock:
            self._references -= 1
            if self._references:
                return
            self.owner = None
            lease, self._lease = self._lease, None
        if lease is not None:
            lease.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class AsyncAudioWriter:
    """Bounded native workers; MLX preparation stays on the caller.

    ``memory_budget_bytes`` limits queued prepared storage to at most 512 MiB
    and one-eighth of RAM. Zero writes serially. ``close`` drains active native
    calls and raises the first error. Caller views are copied when necessary
    so tiny slices cannot keep unaccounted base allocations in the queue.
    """

    def __init__(
        self,
        maxsize=4,
        workers=2,
        *,
        clip="rescale",
        bits_per_sample=16,
        as_float=False,
        memory_budget_bytes=None,
        _control=None,
    ):
        import queue
        import threading

        from .io_pipeline import IOBudget

        if workers <= 0 or maxsize <= 0:
            raise ValueError("workers and maxsize must be > 0")
        import mlx_audio_io as mac

        # Public prepared buffers acquire their allocation on the producer;
        # workers only encode an already owned Python buffer.
        if not hasattr(mac, "save_prepared") or not hasattr(mac, "prepare_save"):
            raise RuntimeError(
                "AsyncAudioWriter requires the mlx-audio-io prepared-buffer API "
                "(version 1.3.24 or newer); install the local sibling development build."
            )
        self._prepared_save = mac.save_prepared
        self._control = _control or IOBudget(memory_budget_bytes)
        self._queue = queue.Queue(maxsize=maxsize)
        self._closing = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self._state = threading.Condition()
        self._submissions = 0
        self._clip, self._bits_per_sample, self._as_float = clip, bits_per_sample, as_float
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name="demucs-writer-" + str(i))
            for i in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def peak_overlap_bytes(self):
        return self._control.peak_bytes

    def track(self, owner, owner_bytes, *, serial=False):
        if serial:
            self.drain()
        return _WriteGroup(self, owner, owner_bytes, serial)

    def _prepare(self, wav, *, owned):
        # Preserve the original clipping/dtype order. Unowned no-clip views need
        # an explicit snapshot: their small logical size can retain a huge base.
        if not isinstance(wav, mx.array):
            wav = mx.asarray(wav)
        prepared = _prevent_clip_mlx(wav, mode=self._clip)
        if prepared.dtype == mx.float16:
            prepared = prepared.astype(mx.float32)
        elif self._clip in (None, "none") and not owned:
            prepared = mx.asarray(prepared, copy=True)
        import mlx_audio_io as mac

        return mac.prepare_save(
            prepared, layout="channels_first" if prepared.ndim > 1 else "channels_last"
        )

    def _write(self, wav, path, samplerate):
        encoding = (
            "float32"
            if self._as_float or self._bits_per_sample == 32
            else ("pcm24" if self._bits_per_sample == 24 else "pcm16")
        )
        _save_prepared(
            wav,
            path,
            samplerate,
            layout=wav.layout,
            encoding=encoding,
            clip=self._clip not in (None, "none"),
            check=self._control.check,
            prepared_save=self._prepared_save,
        )

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            prepared, path, samplerate, lease, group = item
            del item
            try:
                self._control.check()
                self._write(prepared, path, samplerate)
            except BaseException as error:
                self._control.fail(error)
            finally:
                # A worker blocked on its next get must not retain the previous
                # view after returning its budget ticket. Drop it first.
                del prepared
                if lease is not None:
                    lease.release()
                if group is not None:
                    group.release()
                self._queue.task_done()

    def submit(self, wav, path, samplerate, *, _group=None):
        with self._state:
            if self._closing.is_set():
                raise RuntimeError("writer is closed")
            self._submissions += 1
        try:
            self._submit(wav, path, samplerate, _group=_group)
        finally:
            with self._state:
                self._submissions -= 1
                self._state.notify_all()

    def _submit(self, wav, path, samplerate, *, _group=None):
        import queue

        self._control.check()
        if not isinstance(wav, mx.array):
            wav = mx.asarray(wav)
        if not mx.issubdtype(wav.dtype, mx.floating):
            raise TypeError("Expected floating-point audio")
        # No-copy members are covered by their complete owner's group ticket.
        size = int(wav.size) * 4 if hasattr(wav, "size") else 0
        extra = (
            size
            if _group is None or self._clip not in (None, "none") or wav.dtype == mx.float16
            else 0
        )
        serial = (
            (_group is not None and _group.serial)
            or extra > self._control.limit
            or self._control.limit == 0
        )
        if serial:
            self.drain()
        lease = None if serial else self._control.reserve(extra)
        retained = False
        queued = False
        try:
            prepared = self._prepare(wav, owned=_group is not None)
            self._control.check()
            if serial:
                self._write(prepared, path, samplerate)
                return
            if _group is not None:
                _group.retain()
                retained = True
            item = (prepared, path, samplerate, lease, _group)
            while True:
                self._control.check()
                try:
                    self._queue.put(item, timeout=0.05)
                    queued = True  # The worker now owns the tickets.
                    self._control.check()
                    break
                except queue.Full:
                    continue
        except BaseException as error:
            if not queued:
                if lease is not None:
                    lease.release()
                if retained:
                    _group.release()
            self._control.fail(error)
            raise

    def drain(self):
        self._queue.join()
        self._control.check()

    def close(self, *, _error=None):
        if _error is not None:
            self._control.fail(_error)
        with self._close_lock:
            if not self._closed:
                with self._state:
                    self._closing.set()
                    while self._submissions:
                        self._state.wait(0.05)
                self._queue.join()
                # Accepted submitters and native calls are drained. Wake idle
                # workers immediately; a polling timeout adds up to 50 ms to
                # every file's public completion latency.
                for _ in self._threads:
                    self._queue.put(None)
                self._queue.join()
                for thread in self._threads:
                    thread.join()
                self._closed = True
        if _error is None:
            self._control.check()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(_error=exc_val)
