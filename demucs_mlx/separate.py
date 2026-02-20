"""MLX-only CLI for Demucs stem separation."""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .mlx_registry import MLX_MODEL_REGISTRY


class _AsyncWriter:
    def __init__(
        self,
        maxsize: int = 4,
        workers: int = 1,
        *,
        clip: str = "rescale",
        bits_per_sample: int = 16,
        as_float: bool = False,
    ):
        if workers <= 0:
            raise ValueError("workers must be > 0")
        self._queue: "queue.Queue[tp.Optional[tuple]]" = queue.Queue(maxsize=maxsize)
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
        from .audio import save_audio
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
            except BaseException as exc:  # propagate after join
                self._error = exc
            finally:
                if item is not None:
                    self._queue.task_done()

    def submit(self, wav: np.ndarray, path: Path, samplerate: int) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put((wav, path, samplerate))

    def close(self) -> None:
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._error is not None:
            raise self._error

def _list_models() -> int:
    for name in sorted(MLX_MODEL_REGISTRY.keys()):
        desc = MLX_MODEL_REGISTRY[name].get("description", "")
        if desc:
            print(f"{name}\t{desc}")
        else:
            print(name)
    return 0


def _load_audio(path: Path, model):
    import mlx_audio_io as mac

    # mlx_audio_io.load returns (mx.array, sample_rate) in shape [frames, channels]
    audio_mx, sr = mac.load(str(path), dtype="float32")
    if sr != model.samplerate:
        raise ValueError(
            f"Input sample rate {sr} does not match model sample rate {model.samplerate}. "
            "Resampling is currently not supported without torch."
        )
    # Convert to numpy and transpose to (channels, frames)
    wav = np.array(audio_mx, copy=False).T
    src_channels = wav.shape[0]
    tgt_channels = model.audio_channels
    if src_channels != tgt_channels:
        if tgt_channels == 1:
            wav = wav.mean(axis=0, keepdims=True)
        elif src_channels == 1 and tgt_channels > 1:
            wav = np.tile(wav, (tgt_channels, 1))
        elif src_channels > tgt_channels:
            wav = wav[:tgt_channels, :]
        else:
            raise ValueError(
                f"Audio has {src_channels} channels but model expects {tgt_channels}."
            )
    return wav

def _iter_prefetched_audio(
    tracks: tp.Sequence[str],
    model,
    *,
    prefetch: int,
) -> tp.Iterator[tuple[Path, np.ndarray]]:
    if prefetch <= 0:
        for track in tracks:
            path = Path(track)
            yield path, _load_audio(path, model)
        return

    q: queue.Queue = queue.Queue(
        maxsize=max(1, int(prefetch))
    )
    done = threading.Event()
    paths = [Path(track) for track in tracks]

    def _producer() -> None:
        try:
            for path in paths:
                if done.is_set():
                    break
                try:
                    wav = _load_audio(path, model)
                except BaseException as exc:
                    q.put((path, None, exc))
                    break
                q.put((path, wav, None))
        finally:
            q.put(None)

    thread = threading.Thread(target=_producer, daemon=True, name="demucs-audio-prefetch")
    thread.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            path, wav, exc = item
            if exc is not None:
                raise exc
            assert wav is not None
            yield path, wav
    finally:
        done.set()
        thread.join()

def _separate_one(
    path: Path,
    wav: np.ndarray,
    model,
    out_dir: Path,
    shifts: int,
    overlap: float,
    segment: tp.Optional[float],
    split: bool,
    batch_size: int,
    verbose: bool,
    writer: _AsyncWriter,
) -> None:
    import mlx.core as mx

    from .apply_mlx import apply_model

    total_steps = 4 + len(model.sources)
    stage = tqdm(total=total_steps, desc=path.name, unit="step", leave=False) if verbose else None
    try:
        if verbose:
            print(f"Loading audio: {path}")
        mix = mx.array(wav)[None, ...]
        if stage is not None:
            stage.update(1)

        if verbose:
            print("Running MLX separation...")
        estimates = apply_model(
            model,
            mix,
            shifts=shifts,
            split=split,
            overlap=overlap,
            segment=segment,
            batch_size=batch_size,
            progress=verbose,
        )
        mx.eval(estimates)
        if stage is not None:
            stage.update(1)
        track_out = out_dir / path.stem
        track_out.mkdir(parents=True, exist_ok=True)
        stem_paths = [track_out / f"{s}.wav" for s in model.sources]
        if stage is not None:
            stage.update(1)

        # Transfer once to host, then slice NumPy arrays for writer workers.
        stems = np.asarray(estimates[0])
        for stem_idx, stem_path in enumerate(stem_paths):
            stem = np.ascontiguousarray(stems[stem_idx], dtype=np.float32)
            writer.submit(stem, stem_path, samplerate=model.samplerate)
            if verbose:
                print(f"Wrote: {stem_path}")
        if stage is not None:
            stage.update(1 + len(stem_paths))
    finally:
        if stage is not None:
            stage.close()


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demucs-mlx",
        description="MLX-only Demucs stem separation",
    )
    parser.add_argument("tracks", nargs="*", help="Audio files to separate")
    parser.add_argument("-n", "--name", default="htdemucs", help="Model name")
    parser.add_argument("-o", "--out", default="separated", help="Output directory")
    parser.add_argument("--segment", type=float, default=None, help="Segment length in seconds")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    parser.add_argument("--shifts", type=int, default=1, help="Number of random shifts")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--write-workers", type=int, default=1,
                        help="Number of concurrent audio writer threads")
    parser.add_argument("--prefetch-tracks", type=int, default=2,
                        help="Number of prefetched decoded tracks")
    parser.add_argument("--no-split", action="store_true", help="Disable chunked inference")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models()

    if not args.tracks:
        parser.print_help(sys.stderr)
        return 2
    if args.shifts < 0:
        raise SystemExit("--shifts must be >= 0")
    if not (0.0 <= float(args.overlap) < 1.0):
        raise SystemExit("--overlap must be in [0, 1)")
    if args.segment is not None and float(args.segment) <= 0:
        raise SystemExit("--segment must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.write_workers <= 0:
        raise SystemExit("--write-workers must be > 0")
    if args.prefetch_tracks < 0:
        raise SystemExit("--prefetch-tracks must be >= 0")

    if args.name not in MLX_MODEL_REGISTRY:
        known = ", ".join(sorted(MLX_MODEL_REGISTRY.keys()))
        raise SystemExit(f"Unknown model '{args.name}'. Available: {known}")

    if args.verbose:
        print(f"Loading MLX model: {args.name}")
    from .model_converter import get_mlx_model
    model = get_mlx_model(args.name)
    if hasattr(model, "eval"):
        model.eval()

    out_dir = Path(args.out)
    writer = _AsyncWriter(maxsize=max(8, args.write_workers * 4), workers=args.write_workers)
    try:
        for path, wav in tqdm(
            _iter_prefetched_audio(args.tracks, model, prefetch=args.prefetch_tracks),
            total=len(args.tracks),
            desc="Tracks",
            unit="track",
        ):
            _separate_one(
                path,
                wav,
                model,
                out_dir,
                shifts=args.shifts,
                overlap=args.overlap,
                segment=args.segment,
                split=not args.no_split,
                batch_size=args.batch_size,
                verbose=args.verbose,
                writer=writer,
            )
    finally:
        writer.close()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
