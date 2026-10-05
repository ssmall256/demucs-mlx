"""CPU-only ownership accounting and cancellation for overlapping file I/O."""

from __future__ import annotations

import threading
from dataclasses import dataclass


def maximum_io_budget() -> int:
    from .hardware import get_topology

    return min(512 * 1024**2, get_topology().ram_bytes // 8)


class IOBudget:
    def __init__(self, limit: int | None = None):
        maximum = maximum_io_budget()
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
        ):
            raise ValueError("memory_budget_bytes must be a nonnegative integer")
        self.limit = maximum if limit is None else min(limit, maximum)
        self.condition = threading.Condition()
        self.error: BaseException | None = None
        self.reserved = 0
        self.peak_bytes = 0
        # Protect room for an output owner plus one prepared stem. Prefetch may
        # use only the remaining room; output reservations consume that credit.
        self.export_headroom = 0

    def check(self):
        with self.condition:
            if self.error is not None:
                raise self.error

    def fail(self, error: BaseException):
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def reserve(self, size: int, *, prefetch=False, stop=None):
        if size < 0:
            raise ValueError("negative reservation")
        with self.condition:
            ceiling = self.limit - (self.export_headroom if prefetch else 0)
            if size > ceiling:
                return None
            while self.reserved + size > ceiling:
                self.check()
                if stop is not None and stop.is_set():
                    return None
                self.condition.wait(0.05)
            self.check()
            if stop is not None and stop.is_set():
                return None
            self.reserved += size
            self.peak_bytes = max(self.peak_bytes, self.reserved)
            return Reservation(self, size)

    def release(self, size):
        with self.condition:
            self.reserved -= size
            assert self.reserved >= 0
            self.condition.notify_all()


class Reservation:
    def __init__(self, budget, size):
        self.budget, self.size = budget, size
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            size, self.size = self.size, 0
        if size:
            self.budget.release(size)


@dataclass(frozen=True)
class TrackEstimate:
    decoded_bytes: int
    export_bytes: int


def estimate_track(path, model, *, clip="rescale", source_count=None):
    """Reserve whole source-channel storage, with one decode chunk of slack."""
    import math

    import mlx_audio_io as mac

    try:
        info = mac.info(str(path))
        if (
            info.frames <= 1
            or info.channels <= 0
            or not math.isfinite(info.sample_rate)
            or info.sample_rate <= 0
        ):
            return None
        frames = math.ceil(info.frames * model.samplerate / info.sample_rate) + 65536
        channels = max(info.channels, model.audio_channels)
        decoded = frames * channels * 4
        count = len(model.sources) if source_count is None else source_count
        output = frames * model.audio_channels * count * 4
        prepared = 0 if clip in (None, "none") else frames * model.audio_channels * 4
        return TrackEstimate(decoded, output + prepared)
    except (OSError, ValueError, RuntimeError, AttributeError, OverflowError):
        # Native formats with no trustworthy metadata stay on the serial path.
        return None


def validate_destinations(paths):
    import unicodedata

    names = [unicodedata.normalize("NFC", str(path.resolve())).casefold() for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Input tracks or stem names collide in the output directory")
