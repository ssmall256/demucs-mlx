"""Ownership backpressure, first-error shutdown and native publication."""

import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as mac
import pytest

from demucs_mlx.audio import AsyncAudioWriter, save_audio
from demucs_mlx.io_pipeline import IOBudget, TrackEstimate
from demucs_mlx.separate import _iter_prefetched_audio


def test_budget_cancel_wakes_blocked_reserver():
    budget = IOBudget(64)
    first = budget.reserve(64)
    error = ValueError("first")
    caught = []

    def wait():
        try:
            budget.reserve(1)
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=wait)
    thread.start()
    budget.fail(error)
    budget.fail(RuntimeError("second"))
    thread.join(2)
    assert not thread.is_alive() and caught == [error]
    first.release()
    first.release()
    assert budget.reserved == 0 and budget.peak_bytes == 64


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_budget(limit):
    with pytest.raises(ValueError):
        IOBudget(limit)


def test_prefetch_full_queue_close_and_order(monkeypatch):
    import demucs_mlx.separate as separate

    monkeypatch.setattr(separate, "_load_audio", lambda path, model: mx.ones((2, 8)))
    budget = IOBudget(256)
    paths = [str(i) for i in range(8)]
    iterator = _iter_prefetched_audio(
        paths, None, prefetch=2, _control=budget, _estimates=[TrackEstimate(64, 0)] * len(paths)
    )
    assert next(iterator)[0] == Path("0")
    time.sleep(0.1)
    iterator.close()
    assert budget.reserved == 0
    assert not any(t.name == "demucs-audio-prefetch" for t in threading.enumerate())
    assert [
        p.name
        for p, _ in _iter_prefetched_audio(
            paths, None, prefetch=2, _control=budget, _estimates=[TrackEstimate(64, 0)] * len(paths)
        )
    ] == paths
    assert budget.reserved == 0 and budget.peak_bytes <= budget.limit


def test_prefetch_error_and_unknown_serial_fence(monkeypatch):
    import demucs_mlx.separate as separate

    loaded = []
    failure = OSError("decode")

    def load(path, model):
        loaded.append(path.name)
        if path.name == "bad":
            raise failure
        return mx.ones((2, 8))

    monkeypatch.setattr(separate, "_load_audio", load)
    budget = IOBudget(128)
    iterator = _iter_prefetched_audio(
        ["first", "bad", "last"], None, prefetch=2, _control=budget, _estimates=[None] * 3
    )
    assert next(iterator)[0].name == "first"
    time.sleep(0.1)
    assert loaded == ["first"]
    with pytest.raises(OSError, match="decode"):
        next(iterator)
    iterator.close()
    assert budget.reserved == 0


@pytest.mark.parametrize("clip", ["none", "rescale", "clamp", "tanh"])
@pytest.mark.parametrize("half", [False, True])
def test_prepared_writer_matches_sync_payload(tmp_path, clip, half):
    samples = mx.linspace(-1.2, 1.2, 260).reshape(2, 130)[:, ::-2]
    if half:
        samples = samples.astype(mx.float16)
    reference, result = tmp_path / "reference.wav", tmp_path / "result.wav"
    save_audio(samples, reference, 44100, clip=clip)
    with AsyncAudioWriter(clip=clip, workers=2, memory_budget_bytes=2048) as writer:
        writer.submit(samples, result, 44100)
    assert reference.read_bytes() == result.read_bytes()
    assert writer._control.reserved == 0
    assert not list(tmp_path.glob(".*.tmp.wav"))
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(samples, result, 44100)


def test_writer_owner_and_copy_reservations(tmp_path):
    root = mx.ones((4, 2, 64))
    control = IOBudget(2560)  # root 2048 + one prepared stem 512.
    with AsyncAudioWriter(workers=1, _control=control) as writer:
        with writer.track(root, root.nbytes) as group:
            for i in range(4):
                writer.submit(root[i], tmp_path / f"{i}.wav", 44100, _group=group)
        writer.drain()
        assert group.owner is None
        assert control.reserved == 0 and control.peak_bytes <= control.limit
    assert len(list(tmp_path.glob("*.wav"))) == 4


def test_writer_workers_do_not_build_mlx_graphs(tmp_path, monkeypatch):
    original = AsyncAudioWriter._prepare
    names = []

    def prepare(self, wav, **kwargs):
        names.append(threading.current_thread().name)
        return original(self, wav, **kwargs)

    monkeypatch.setattr(AsyncAudioWriter, "_prepare", prepare)
    with AsyncAudioWriter() as writer:
        writer.submit(mx.ones((2, 64)), tmp_path / "stem.wav", 44100)
    assert names == [threading.current_thread().name]


def test_atomic_write_failure_preserves_destination(tmp_path, monkeypatch):
    path = tmp_path / "stem.wav"
    path.write_bytes(b"old")
    error = OSError("disk")

    def fail(path, *args, **kwargs):
        Path(path).write_bytes(b"unfinished")
        raise error

    monkeypatch.setattr(mac._get_core_module(), "save", fail)
    writer = AsyncAudioWriter(workers=1, maxsize=1)
    writer.submit(mx.ones((2, 32)), path, 44100)
    with pytest.raises(OSError, match="disk"):
        writer.close()
    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp.wav"))
    assert writer._control.error is error and writer._control.reserved == 0
    assert not any(t.is_alive() for t in writer._threads)


def test_context_primary_error_and_zero_budget(tmp_path):
    writer = AsyncAudioWriter(memory_budget_bytes=0)
    failure = RuntimeError("inference")
    with pytest.raises(RuntimeError, match="inference"):
        with writer:
            writer.submit(mx.ones((2, 32)), tmp_path / "complete.wav", 44100)
            raise failure
    assert (tmp_path / "complete.wav").exists()
    assert writer.peak_overlap_bytes == 0
    assert writer._control.error is failure
    assert not any(t.is_alive() for t in writer._threads)


def test_collision_normalization_and_budget_headroom(tmp_path):
    from demucs_mlx.io_pipeline import validate_destinations

    with pytest.raises(ValueError, match="collide"):
        validate_destinations([tmp_path / "Track", tmp_path / "track"])
    budget = IOBudget(128)
    budget.export_headroom = 96
    assert budget.reserve(64, prefetch=True) is None
    ticket = budget.reserve(32, prefetch=True)
    export = budget.reserve(96)
    assert budget.peak_bytes == 128
    export.release()
    ticket.release()


def test_writer_failure_wakes_full_queue_submitter(tmp_path, monkeypatch):
    first_started, release = threading.Event(), threading.Event()
    error = OSError("first-write")

    def fail(*args, **kwargs):
        first_started.set()
        assert release.wait(3)
        raise error

    monkeypatch.setattr(mac._get_core_module(), "save", fail)
    samples = mx.ones((2, 16))
    mx.eval(samples)
    writer = AsyncAudioWriter(workers=1, maxsize=1)
    writer.submit(samples, tmp_path / "first.wav", 44100)
    assert first_started.wait(3)
    writer.submit(samples, tmp_path / "second.wav", 44100)
    caught = []

    def submit():
        try:
            writer.submit(samples, tmp_path / "third.wav", 44100)
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=submit)
    thread.start()
    time.sleep(0.1)
    release.set()
    thread.join(3)
    assert not thread.is_alive() and caught == [error]
    with pytest.raises(OSError, match="first-write"):
        writer.close()
    assert writer._control.reserved == 0
    assert not list(tmp_path.glob("*.wav"))


def test_inference_cancellation_does_not_dispatch_forward(monkeypatch):
    from types import SimpleNamespace

    import demucs_mlx.apply_mlx as apply

    dispatches, checks = [], []
    failure = RuntimeError("cancel inference")

    def check():
        checks.append(1)
        if len(checks) == 2:
            raise failure

    monkeypatch.setattr(apply, "_forward", lambda *args, **kwargs: dispatches.append(1))
    model = SimpleNamespace(samplerate=44100, sources=["s"], audio_channels=2)
    with pytest.raises(RuntimeError, match="cancel inference"):
        apply.apply_model(model, mx.ones((1, 2, 44100)), _check_cancel=check, seed=481)
    assert checks == [1, 1] and not dispatches


@pytest.mark.parametrize("prefetch", [0, 2])
def test_serial_fence_drains_before_decode(monkeypatch, prefetch):
    import demucs_mlx.separate as separate

    calls = []
    monkeypatch.setattr(
        separate, "_load_audio", lambda *args: calls.append("decode") or mx.ones((2, 8))
    )
    iterator = _iter_prefetched_audio(
        ["oversized"],
        None,
        prefetch=prefetch,
        _control=IOBudget(128),
        _estimates=[None],
        _before_serial=lambda: calls.append("drain"),
    )
    assert next(iterator)[0] == Path("oversized")
    iterator.close()
    assert calls == ["drain", "decode"]


def test_sdk_collision_preflight_and_single_stem(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from demucs_mlx.api import Separator

    separator = Separator.__new__(Separator)
    separator._model = SimpleNamespace(sources=["drums", "bass"], samplerate=44100)
    separator.stem = None
    called = []
    monkeypatch.setattr(
        separator,
        "separate_tensor",
        lambda *a, **kw: (None, {"drums": mx.zeros((2, 8))}) if called.append(1) is None else None,
    )
    with pytest.raises(ValueError, match="collide"):
        separator.separate(None, output_dir=tmp_path, filename_format="stem.wav")
    assert not called
    separator.stem = "drums"
    result = separator.separate(None, output_dir=tmp_path, filename_format="stem.wav")
    assert set(result) == {"drums"} and (tmp_path / "stem.wav").exists()


def test_cli_collision_preflight_precedes_model_loading(tmp_path, monkeypatch):
    import demucs_mlx.model_converter as converter
    import demucs_mlx.separate as separate

    def unexpected(*args, **kwargs):
        raise AssertionError("Model loading precedes collision check")

    monkeypatch.setattr(converter, "get_mlx_model", unexpected)
    with pytest.raises(SystemExit, match="collide"):
        separate.main([str(tmp_path / "a/track.m4a"), str(tmp_path / "b/track.wav")])


def test_cli_unknown_root_serial_and_retires_input(tmp_path, monkeypatch):
    import weakref
    from types import SimpleNamespace

    import demucs_mlx.model_converter as converter
    import demucs_mlx.separate as separate

    model = SimpleNamespace(sources=["stem"], samplerate=44100, audio_channels=2)
    monkeypatch.setattr(converter, "get_mlx_model", lambda *a: model)
    roots = []

    def load(*args):
        if roots:
            assert roots[-1]() is None, "Finished input survives into the next decode"
        value = SimpleAudio()
        roots.append(weakref.ref(value))
        return value

    class SimpleAudio:
        pass

    monkeypatch.setattr(separate, "_load_audio", load)

    def infer(*args, **kwargs):
        assert kwargs["serial_export"]

    monkeypatch.setattr(separate, "_separate_one", infer)
    assert (
        separate.main(
            [
                "one.wav",
                "two.wav",
                "--out",
                str(tmp_path),
                "--no-split",
                "--shifts",
                "0",
                "--prefetch-tracks",
                "0",
            ]
        )
        == 0
    )
    assert len(roots) == 2 and all(root() is None for root in roots)


def test_cli_preserves_inference_error_when_ane_close_fails(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import demucs_mlx.model_converter as converter
    import demucs_mlx.separate as separate

    model = SimpleNamespace(sources=["stem"], samplerate=44100, audio_channels=2)
    model.models = [model]
    monkeypatch.setattr(converter, "get_mlx_model", lambda *a: model)
    closed = []

    class Worker:
        def close(self):
            closed.append(1)
            raise RuntimeError("secondary close error")

    monkeypatch.setitem(sys.modules, "demucs_mlx.ane", SimpleNamespace(WaveformConv=Worker))
    monkeypatch.setattr(separate, "_load_audio", lambda *a: None)
    error = OSError("primary inference error")

    def infer(*args, **kwargs):
        raise error

    monkeypatch.setattr(separate, "_separate_one", infer)
    with pytest.raises(OSError, match="primary inference error") as caught:
        separate.main(
            ["track.wav", "--out", str(tmp_path), "--prefetch-tracks", "0", "--ane-time-encoder"]
        )
    assert caught.value is error and closed == [1]


def test_idle_worker_releases_view_before_returning_ticket(tmp_path, monkeypatch):
    import weakref

    views = []

    class View:
        pass

    def prepare(*args, **kwargs):
        view = View()
        views.append(weakref.ref(view))
        return view

    monkeypatch.setattr(AsyncAudioWriter, "_prepare", prepare)
    monkeypatch.setattr(AsyncAudioWriter, "_write", lambda *args: None)
    writer = AsyncAudioWriter(workers=1)
    try:
        writer.submit(mx.ones((2, 8)), tmp_path / "stem.wav", 44100)
        writer.drain()
        assert views[0]() is None and writer._control.reserved == 0
    finally:
        writer.close()


def test_inference_cancels_between_batches(monkeypatch):
    from types import SimpleNamespace

    import demucs_mlx.apply_mlx as apply

    checks, dispatched = [], []
    error = RuntimeError("cancel second batch")

    def check():
        checks.append(1)
        if len(checks) == 3:
            raise error

    def forward(model, value, **kwargs):
        dispatched.append(value.shape)
        return mx.zeros((value.shape[0], 1, 2, value.shape[-1]))

    monkeypatch.setattr(apply, "_forward", forward)
    model = SimpleNamespace(samplerate=100, segment=2, sources=["s"], audio_channels=2)
    with pytest.raises(RuntimeError, match="cancel second batch"):
        apply.apply_model(
            model, mx.ones((1, 2, 1000)), shifts=0, batch_size=1, _check_cancel=check, seed=481
        )
    assert len(dispatched) == 1


def test_native_worker_never_calls_python_mlx_eval(tmp_path, monkeypatch):
    caller = threading.get_ident()
    original = mx.eval

    def evaluate(*args, **kwargs):
        assert threading.get_ident() == caller
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", evaluate)
    with AsyncAudioWriter(workers=2) as writer:
        writer.submit(mx.ones((2, 64)), tmp_path / "stem.wav", 44100)
    assert (tmp_path / "stem.wav").exists()


def test_native_worker_receives_public_prepared_buffer(tmp_path, monkeypatch):
    caller = threading.get_ident()
    core = mac._get_core_module()
    original = core.save
    seen = []

    def write(path, samples, *args, **kwargs):
        assert threading.get_ident() != caller
        assert isinstance(samples, memoryview) and samples.readonly
        seen.append((samples.shape, samples.strides))
        return original(path, samples, *args, **kwargs)

    monkeypatch.setattr(core, "save", write)
    with AsyncAudioWriter(workers=1, clip="none") as writer:
        writer.submit(mx.ones((2, 65)), tmp_path / "stem.wav", 44100)
    assert seen == [((2, 65), (260, 4))]
