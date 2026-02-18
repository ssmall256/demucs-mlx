# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""MLX utility functions for demucs_mlx."""

from collections import defaultdict
from concurrent.futures import CancelledError
from contextlib import contextmanager
import math
import os
import tempfile
import typing as tp

import mlx.core as mx


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output array of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    pad_amount = tgt_length - length
    if pad_amount > 0:
        a = mx.pad(a, [(0, 0)] * (len(shape)) + [(0, pad_amount)])

    # Extract frames manually
    frames = []
    for i in range(n_frames):
        start = i * stride
        end = start + kernel_size
        if end <= tgt_length:
            frames.append(a[..., start:end])
        else:
            # Pad the last frame if needed
            frame = a[..., start:]
            pad_needed = kernel_size - frame.shape[-1]
            frame = mx.pad(frame, [(0, 0)] * (len(shape)) + [(0, pad_needed)])
            frames.append(frame)

    return mx.stack(frames, axis=-2)


def center_trim(tensor, reference: tp.Union[mx.array, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, mx.array):
        ref_size = reference.shape[-1]
    else:
        ref_size = reference
    delta = tensor.shape[-1] - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        start = delta // 2
        end = tensor.shape[-1] - (delta - delta // 2)
        tensor = tensor[..., start:end]
    return tensor


def pull_metric(history: tp.List[dict], name: str):
    out = []
    for metrics in history:
        metric = metrics
        for part in name.split("."):
            metric = metric[part]
        out.append(metric)
    return out


def EMA(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


def sizeof_fmt(num: float, suffix: str = 'B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def random_subset(dataset, max_samples: int, seed: int = 42):
    if max_samples >= len(dataset):
        return dataset
    # For MLX, we'll keep this simple for now
    # This is typically used for training, which we're not focusing on yet
    return dataset


class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, _dict, *args, **kwargs):
            self.func = func
            self._dict = _dict
            self.args = args
            self.kwargs = kwargs

        def result(self):
            if self._dict["run"]:
                return self.func(*self.args, **self.kwargs)
            else:
                raise CancelledError()

    def __init__(self, workers=0):
        self._dict = {"run": True}

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, self._dict, *args, **kwargs)

    def shutdown(self, *_, **__):
        self._dict["run"] = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return
