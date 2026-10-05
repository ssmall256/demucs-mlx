"""MLX-only CLI for Demucs stem separation."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import typing as tp
from pathlib import Path

from tqdm import tqdm

from .audio import AsyncAudioWriter as _AsyncWriter
from .defaults import DEFAULT_BATCH_SIZE
from .mlx_registry import MLX_MODEL_REGISTRY


def _list_models() -> int:
    for name in sorted(MLX_MODEL_REGISTRY.keys()):
        desc = MLX_MODEL_REGISTRY[name].get("description", "")
        if desc:
            print(f"{name}\t{desc}")
        else:
            print(name)
    return 0


def _load_audio(path: Path, model):
    import mlx.core as mx

    from .audio import load_audio

    audio_mx, sr = load_audio(path, sr=model.samplerate, dtype="float32")
    wav = audio_mx
    src_channels = wav.shape[0]
    tgt_channels = model.audio_channels
    if src_channels != tgt_channels:
        if tgt_channels == 1:
            wav = mx.mean(wav, axis=0, keepdims=True)
        elif src_channels == 1 and tgt_channels > 1:
            wav = mx.broadcast_to(wav, (tgt_channels, wav.shape[1]))
        elif src_channels > tgt_channels:
            wav = wav[:tgt_channels, :]
        else:
            raise ValueError(f"Audio has {src_channels} channels but model expects {tgt_channels}.")
    return wav


def _iter_prefetched_audio(
    tracks, model, *, prefetch, _control=None, _estimates=None, _before_serial=None
):
    """Decode on the producer's MLX stream; never block shutdown on q.put."""
    from .io_pipeline import IOBudget, estimate_track

    control = _control or IOBudget()
    paths = [Path(track) for track in tracks]
    estimates = _estimates or [
        estimate_track(path, model, source_count=len(getattr(model, "sources", ["audio"])))
        for path in paths
    ]
    done = threading.Event()
    finished = threading.Event()
    q = queue.Queue(maxsize=max(1, prefetch))

    def put(item):
        while not done.is_set():
            control.check()
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def producer():
        import mlx.core as mx

        lease = None
        try:
            for path, estimate in zip(paths, estimates):
                while q.full() and not done.wait(0.05):
                    control.check()
                if done.is_set():
                    break
                if estimate is None or estimate.export_bytes > control.limit:
                    lease = None
                else:
                    lease = control.reserve(estimate.decoded_bytes, prefetch=True, stop=done)
                if done.is_set():
                    break
                if lease is None:
                    # A fence keeps later decodes from overlapping a serial or
                    # unknown-size track. Resume only after the consumer advances.
                    resume = threading.Event()
                    if not put((path, None, None, resume)):
                        break
                    while not resume.wait(0.05) and not done.is_set():
                        control.check()
                    continue
                wav = _load_audio(path, model)
                mx.eval(wav)
                if wav.nbytes > lease.size:
                    raise ValueError("Decoded allocation exceeds its metadata reservation")
                if not put((path, wav, lease, None)):
                    break
                lease = None
                del wav
        except BaseException as error:
            control.fail(error)
        finally:
            if lease is not None:
                lease.release()
            finished.set()

    if prefetch <= 0 or control.limit == 0:
        for path, estimate in zip(paths, estimates):
            control.check()
            if _before_serial is not None and (
                control.limit == 0
                or estimate is None
                or estimate.decoded_bytes + estimate.export_bytes > control.limit
            ):
                _before_serial()
            yield path, _load_audio(path, model)
        return
    thread = threading.Thread(target=producer, daemon=True, name="demucs-audio-prefetch")
    thread.start()
    try:
        while True:
            control.check()
            try:
                path, wav, lease, resume = q.get(timeout=0.05)
            except queue.Empty:
                if finished.is_set():
                    break
                continue
            if lease is not None:
                lease.release()  # This input becomes the active inference track.
            if wav is None:
                if _before_serial is not None:
                    _before_serial()
                wav = _load_audio(path, model)
            yield path, wav
            del wav
            if resume is not None:
                resume.set()
    finally:
        done.set()
        with control.condition:
            control.condition.notify_all()
        thread.join()  # Active native decode completes at its call boundary.
        while True:
            try:
                _, _, lease, _ = q.get_nowait()
                if lease is not None:
                    lease.release()
            except queue.Empty:
                break


def _separate_one(
    path: Path,
    wav,
    model,
    out_dir: Path,
    shifts: int,
    seed: tp.Optional[int],
    overlap: float,
    segment: tp.Optional[float],
    split: bool,
    batch_size: int,
    verbose: bool,
    writer: _AsyncWriter,
    stem: tp.Optional[str] = None,
    compile: tp.Optional[bool] = None,
    serial_export: bool = False,
) -> None:
    import mlx.core as mx

    from .apply_mlx import apply_model

    source_names = (stem,) if stem is not None else model.sources
    source_index = model.sources.index(stem) if stem is not None else None
    total_steps = 4 + len(source_names)
    stage = tqdm(total=total_steps, desc=path.name, unit="step", leave=False) if verbose else None
    try:
        if verbose:
            print(f"Loading audio: {path}")
        mix = wav[None, ...]
        if stage is not None:
            stage.update(1)

        if verbose:
            print("Running MLX separation...")
        estimates = apply_model(
            model,
            mix,
            shifts=shifts,
            seed=seed,
            split=split,
            overlap=overlap,
            segment=segment,
            batch_size=batch_size,
            progress=verbose,
            source_index=source_index,
            compile=compile,
            _check_cancel=writer._control.check,
        )
        mx.eval(estimates)
        if stage is not None:
            stage.update(1)
        track_out = out_dir / path.stem
        track_out.mkdir(parents=True, exist_ok=True)
        stem_paths = [track_out / f"{s}.wav" for s in source_names]
        if stage is not None:
            stage.update(1)

        # Pure MLX zero-copy stem handoff to async writer.
        # Materialize slice views on producer thread before worker dispatch.
        stems = [estimates[0][i] for i in range(len(source_names))]
        mx.eval(*stems)
        with writer.track(estimates, int(estimates.nbytes), serial=serial_export) as group:
            for stem_idx, stem_path in enumerate(stem_paths):
                writer.submit(stems[stem_idx], stem_path, samplerate=model.samplerate, _group=group)
                if verbose:
                    print(f"Queued: {stem_path}")
        if stage is not None:
            stage.update(1 + len(stem_paths))
    finally:
        if stage is not None:
            stage.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demucs-mlx",
        description="MLX-only Demucs stem separation",
    )
    parser.add_argument("tracks", nargs="*", help="Audio files to separate")
    parser.add_argument("-n", "--name", default="htdemucs", help="Model name")
    parser.add_argument(
        "--stem",
        default=None,
        help="For htdemucs_ft, compute only this stem (drums, bass, other, or vocals)",
    )
    parser.add_argument("-o", "--out", default="separated", help="Output directory")
    parser.add_argument("--segment", type=float, default=None, help="Segment length in seconds")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio")
    parser.add_argument("--shifts", type=int, default=1, help="Number of random shifts")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible shifts",
    )

    def _parse_batch_size(val: str):
        if str(val).lower() == "auto":
            return "auto"
        try:
            ival = int(val)
            if ival <= 0:
                raise argparse.ArgumentTypeError("--batch-size must be > 0")
            return ival
        except ValueError:
            raise argparse.ArgumentTypeError("--batch-size must be an integer or 'auto'")

    parser.add_argument(
        "-b",
        "--batch-size",
        type=_parse_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference (default: 'auto' based on hardware topology)",
    )
    parser.add_argument(
        "--write-workers", type=int, default=4, help="Number of concurrent audio writer threads"
    )
    parser.add_argument(
        "--prefetch-tracks", type=int, default=2, help="Number of prefetched decoded tracks"
    )
    parser.add_argument(
        "--io-memory-mib",
        type=int,
        default=None,
        help="Additional I/O overlap memory in MiB (default: min(512 MiB, RAM/8); 0 disables overlap)",
    )
    parser.add_argument("--no-split", action="store_true", help="Disable chunked inference")
    parser.add_argument(
        "--attention",
        choices=["fp32", "fp16"],
        default=None,
        help=(
            "Attention kernel precision: fp16 (default; projections stay fp32, ~3%% "
            "faster than fp32 and within 0.5 dB of it against upstream) or fp32. "
            "DEMUCS_MLX_ATTENTION_FP16=0 also selects fp32."
        ),
    )
    parser.add_argument(
        "--ane-time-encoder",
        action="store_true",
        help="run the first HTDemucs waveform convolution on the Neural Engine",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile repeated forward graph chunks "
            "(also controlled by DEMUCS_MLX_COMPILE_FORWARD=1)"
        ),
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Auto-tune batch size and stream policy based on Apple Silicon topology",
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    parser = _build_parser()

    args = parser.parse_args(argv)

    if args.list_models:
        return _list_models()

    if not args.tracks:
        parser.print_help(sys.stderr)
        return 2
    if args.auto_tune or args.batch_size is None or str(args.batch_size).lower() == "auto":
        from .hardware import optimal_batch_size

        args.batch_size = optimal_batch_size()
    elif int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be > 0")
    else:
        args.batch_size = int(args.batch_size)
    if args.shifts < 0:
        raise SystemExit("--shifts must be >= 0")
    if not (0.0 <= float(args.overlap) < 1.0):
        raise SystemExit("--overlap must be in [0, 1)")
    if args.segment is not None and float(args.segment) <= 0:
        raise SystemExit("--segment must be > 0")
    if args.write_workers <= 0:
        raise SystemExit("--write-workers must be > 0")
    if args.prefetch_tracks < 0:
        raise SystemExit("--prefetch-tracks must be >= 0")

    if args.name not in MLX_MODEL_REGISTRY:
        known = ", ".join(sorted(MLX_MODEL_REGISTRY.keys()))
        raise SystemExit(f"Unknown model '{args.name}'. Available: {known}")
    if args.stem is not None and args.name != "htdemucs_ft":
        raise SystemExit("--stem acceleration only supports htdemucs_ft")
    if args.ane_time_encoder:
        if args.name != "htdemucs":
            raise SystemExit("--ane-time-encoder only supports the default htdemucs model")
        if args.no_split or (args.segment is not None and args.segment != 7.8):
            raise SystemExit("--ane-time-encoder requires split 7.8-second segments")

    if args.verbose:
        print(f"Loading MLX model: {args.name}")
    from .model_converter import get_mlx_model

    from .io_pipeline import validate_destinations

    if args.io_memory_mib is not None and args.io_memory_mib < 0:
        raise SystemExit("--io-memory-mib must be >= 0")
    try:
        validate_destinations([Path(args.out) / Path(track).stem for track in args.tracks])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    model = get_mlx_model(args.name)
    if hasattr(model, "eval"):
        model.eval()
    from .mlx_transformer import resolve_attention_dtype, set_attention_dtype

    attention_dtype = resolve_attention_dtype(args.attention)
    for sub in getattr(model, "models", [model]):
        if hasattr(sub, "named_modules"):
            set_attention_dtype(sub, attention_dtype)
    if args.stem is not None and args.stem not in model.sources:
        raise SystemExit(f"Unknown stem {args.stem!r}; available: {', '.join(model.sources)}")
    ane_worker = None
    if args.ane_time_encoder:
        from .ane import WaveformConv

        if len(model.models) != 1:
            raise SystemExit("--ane-time-encoder requires a single HTDemucs model")
        ane_worker = WaveformConv()
        model.models[0]._ane_time_conv = ane_worker

    from .io_pipeline import IOBudget, estimate_track

    out_dir = Path(args.out)
    requested = None if args.io_memory_mib is None else args.io_memory_mib * 1024**2
    control = IOBudget(requested)
    count = 1 if args.stem else len(model.sources)
    # An unsplit, unshifted result may be a small center-trim view of a
    # full training-length output. Its root allocation is not exposed by the
    # Python array API; keep such tracks serial rather than charge view.nbytes.
    estimates = (
        [None] * len(args.tracks)
        if args.no_split and args.shifts == 0
        else [estimate_track(Path(track), model, source_count=count) for track in args.tracks]
    )
    control.export_headroom = max(
        (
            estimate.export_bytes
            for estimate in estimates
            if estimate is not None and estimate.export_bytes <= control.limit
        ),
        default=0,
    )
    writer = _AsyncWriter(
        maxsize=max(8, args.write_workers * 4), workers=args.write_workers, _control=control
    )
    audio = _iter_prefetched_audio(
        args.tracks,
        model,
        prefetch=args.prefetch_tracks,
        _control=control,
        _estimates=estimates,
        _before_serial=writer.drain,
    )
    error = None
    index = 0
    track_progress = tqdm(total=len(args.tracks), desc="Tracks", unit="track")
    try:
        # enumerate/tqdm iterators cache their previously yielded tuple, which
        # retains an input while the next serial decode allocates its buffer.
        for path, wav in audio:
            estimate = estimates[index]
            serial = (
                estimate is None or estimate.export_bytes + estimate.decoded_bytes > control.limit
            )
            if serial:
                writer.drain()
            _separate_one(
                path,
                wav,
                model,
                out_dir,
                shifts=args.shifts,
                seed=args.seed,
                overlap=args.overlap,
                segment=args.segment,
                split=not args.no_split,
                batch_size=args.batch_size,
                verbose=args.verbose,
                writer=writer,
                stem=args.stem,
                compile=args.compile,
                serial_export=serial,
            )
            del wav  # Retire the active input before advancing the decoder.
            index += 1
            track_progress.update(1)
    except BaseException as exc:
        error = exc
        control.fail(exc)
        raise
    finally:
        cleanup = [audio.close, lambda: writer.close(_error=error), track_progress.close]
        if ane_worker is not None:
            cleanup.append(ane_worker.close)
        for close in cleanup:
            try:
                close()
            except BaseException as exc:
                control.fail(exc)
        if error is None:
            control.check()
        if args.verbose and ane_worker is not None and control.error is None:
            print(
                "Neural Engine waveform convolution: "
                f"{ane_worker.predictions} predictions, "
                f"{ane_worker.busy_seconds:.3f}s execution, "
                f"{ane_worker.wait_seconds:.3f}s wait, "
                f"{ane_worker.transfer_seconds:.3f}s transfer"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
