"""Regression checks for the shared inference batch-size default."""
from __future__ import annotations

import inspect

from demucs_mlx.api import Separator
from demucs_mlx.apply_mlx import apply_model
from demucs_mlx.defaults import DEFAULT_BATCH_SIZE
from demucs_mlx.separate import _build_parser


class _FakeModel:
    samplerate = 44_100
    audio_channels = 2
    sources = ["mix"]

    def eval(self) -> None:
        return None


def main() -> None:
    assert DEFAULT_BATCH_SIZE == 2
    assert inspect.signature(Separator).parameters["batch_size"].default == DEFAULT_BATCH_SIZE
    assert inspect.signature(apply_model).parameters["batch_size"].default == DEFAULT_BATCH_SIZE

    parser = _build_parser()
    assert parser.parse_args(["track.wav"]).batch_size == DEFAULT_BATCH_SIZE
    assert parser.parse_args(["track.wav", "--batch-size", "3"]).batch_size == 3

    from demucs_mlx import model_converter

    original_get_mlx_model = model_converter.get_mlx_model
    model_converter.get_mlx_model = lambda _name: _FakeModel()
    try:
        assert Separator().batch_size == DEFAULT_BATCH_SIZE
        assert Separator(batch_size=3).batch_size == 3
        try:
            Separator(batch_size=0)
        except ValueError:
            pass
        else:
            raise AssertionError("batch_size=0 must raise ValueError")
    finally:
        model_converter.get_mlx_model = original_get_mlx_model

    print("test_default_batch_size.py: OK")


if __name__ == "__main__":
    main()
