"""Regression checks against upstream Demucs.

The chunking tests run without torch. The parity tests build small,
randomly initialised upstream models, convert them, and compare one forward
pass, so they exercise each architecture's code paths without downloads; they
are skipped when torch and demucs are not installed.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from demucs_mlx.apply_mlx import apply_model


def _snr_db(reference: mx.array, estimate: mx.array) -> float:
    error = mx.sum((reference - estimate) ** 2)
    return float((10 * mx.log10(mx.sum(reference**2) / mx.maximum(error, 1e-30))).item())


def _mix(seconds: float, samplerate: int = 44_100) -> mx.array:
    frames = int(seconds * samplerate)
    mix = mx.random.uniform(-0.5, 0.5, (1, 2, frames), key=mx.random.key(5))
    mx.eval(mix)
    return mix


class _PaddingIdentity:
    """Identity model whose valid length exceeds its input, like HDemucs/Demucs."""

    samplerate = 44_100
    segment = 1.0
    sources = ["mix"]

    def __init__(self, extra: int):
        self.extra = extra
        self.input_lengths: list[int] = []

    def valid_length(self, length: int) -> int:
        return int(length) + self.extra

    def __call__(self, x: mx.array) -> mx.array:
        self.input_lengths.append(int(x.shape[-1]))
        return x[:, None, :, :]


@pytest.mark.parametrize("extra", [0, 37, 1786])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("seconds", [0.6, 3.3])
def test_split_reconstructs_when_model_pads_beyond_the_segment(extra, batch_size, seconds):
    """Each chunk must be trimmed to its own length before overlap-add."""
    mix = _mix(seconds)
    model = _PaddingIdentity(extra)
    out = apply_model(
        model, mix, shifts=0, split=True, overlap=0.25, batch_size=batch_size, compile=False
    )
    mx.eval(out)
    assert out.shape == (1, 1, 2, mix.shape[-1])
    assert mx.max(mx.abs(out[:, 0] - mix)).item() < 1e-5


def test_chunks_run_at_their_own_valid_length():
    """Upstream runs a short tail chunk at valid_length(its length), not the segment's."""
    samplerate = _PaddingIdentity.samplerate
    mix = _mix(2.3)
    model = _PaddingIdentity(10)
    apply_model(model, mix, shifts=0, split=True, overlap=0.25, batch_size=1, compile=False)
    segment = int(samplerate * model.segment)
    stride = int(0.75 * segment)
    expected = [
        model.valid_length(min(segment, mix.shape[-1] - offset))
        for offset in range(0, mix.shape[-1], stride)
    ]
    assert model.input_lengths == expected


# --------------------------------------------------------------------------
# Upstream parity with small random models
# --------------------------------------------------------------------------

def _to_torch(array: mx.array):
    """MLX -> torch on the CPU through the buffer protocol."""
    import torch

    array = mx.contiguous(array.astype(mx.float32))
    mx.eval(array)
    return torch.frombuffer(bytearray(memoryview(array)), dtype=torch.float32).reshape(array.shape)


def _upstream():
    torch = pytest.importorskip("torch")
    pytest.importorskip("demucs")
    torch.manual_seed(0)
    return torch


def _forward_parity(torch_model, seconds: float, attention: str | None = None) -> dict[str, float]:
    torch = _upstream()
    from demucs_mlx.mlx_convert import convert_single_model
    from demucs_mlx.mlx_transformer import resolve_attention_dtype, set_attention_dtype

    torch_model.eval()
    if hasattr(torch_model, "valid_length"):
        length = torch_model.valid_length(int(seconds * torch_model.samplerate))
    else:
        length = int(seconds * torch_model.samplerate)
    mix = _mix(length / torch_model.samplerate)[..., :length]
    mx.eval(mix)
    with torch.no_grad():
        reference = mx.asarray(torch_model(_to_torch(mix)).contiguous())
    model = convert_single_model(torch_model)
    model.eval()
    set_attention_dtype(model, resolve_attention_dtype(attention))
    out = model(mix)
    mx.eval(out)
    return {name: _snr_db(reference[0, k], out[0, k]) for k, name in enumerate(torch_model.sources)}


SOURCES = ["drums", "bass", "other", "vocals"]


def _assert_parity(snrs: dict[str, float], floor: float) -> None:
    assert min(snrs.values()) > floor, snrs


def test_htdemucs_parity():
    """Covers the cross-transformer input ordering (each branch reads the other's layer input)."""
    _upstream()
    from demucs.htdemucs import HTDemucs

    # Layer scale off, or the 1e-4 init hides any transformer error.
    model = HTDemucs(SOURCES, channels=8, depth=4, segment=1, bottom_channels=16,
                     t_layers=4, t_heads=2, dconv_mode=3, t_layer_scale=False, dconv_init=1.0)
    _assert_parity(_forward_parity(model, 1.0), 75.0)


def test_htdemucs_fp16_attention_parity():
    """The opt-in FP16 attention trades some parity for speed."""
    _upstream()
    from demucs.htdemucs import HTDemucs

    model = HTDemucs(SOURCES, channels=8, depth=4, segment=1, bottom_channels=16,
                     t_layers=4, t_heads=2, dconv_mode=3, t_layer_scale=False, dconv_init=1.0)
    _assert_parity(_forward_parity(model, 1.0, attention="fp16"), 55.0)


def test_hdemucs_local_attention_and_lstm_parity():
    """Covers DConv LocalState attention and BLSTM (depth >= dconv_attn/lstm)."""
    _upstream()
    from demucs.hdemucs import HDemucs

    # dconv_init=1: a 1e-4 layer scale would hide any DConv error.
    model = HDemucs(SOURCES, channels=8, depth=6, segment=4, cac=True, norm_starts=4,
                    dconv_attn=4, dconv_lstm=4, dconv_init=1.0)
    _assert_parity(_forward_parity(model, 2.0), 70.0)


def test_hdemucs_magnitude_wiener_parity():
    """Covers cac=False: magnitude output, phase copy (atan2) and wiener layout."""
    _upstream()
    from demucs.hdemucs import HDemucs

    model = HDemucs(SOURCES, channels=8, depth=6, segment=4, cac=False, hybrid_old=True,
                    norm_starts=999)
    _assert_parity(_forward_parity(model, 2.0), 70.0)


def test_hdemucs_multi_freqs_parity():
    """Covers MultiWrap: frequency-axis padding and transposed-conv bias overlap."""
    _upstream()
    from demucs.hdemucs import HDemucs

    model = HDemucs(SOURCES, channels=8, depth=6, segment=4, cac=True, multi_freqs=[0.1, 0.3],
                    multi_freqs_depth=2, norm_starts=999)
    _assert_parity(_forward_parity(model, 2.0), 70.0)


def test_demucs_time_domain_parity():
    """Covers julius x2 resampling, LocalState and BLSTM in the time-domain Demucs."""
    _upstream()
    from demucs.demucs import Demucs

    model = Demucs(SOURCES, channels=8, depth=6, segment=4, resample=True, dconv_attn=4,
                   dconv_lstm=4, norm_starts=4, rewrite=False, dconv_init=1.0)
    _assert_parity(_forward_parity(model, 2.0), 90.0)


def test_resample_matches_julius():
    _upstream()
    julius = pytest.importorskip("julius")
    from demucs_mlx.mlx_demucs import _resample_2x, _resample_half

    x = _mix(0.37)
    tx = _to_torch(x)
    up = mx.asarray(julius.resample_frac(tx, 1, 2).contiguous())
    down = mx.asarray(julius.resample_frac(tx, 2, 1).contiguous())
    assert _snr_db(up, _resample_2x(x)) > 100
    assert _snr_db(down, _resample_half(x)) > 100


@pytest.mark.parametrize("nfreqs", [0, 2])
def test_local_state_matches_upstream(nfreqs):
    """LocalState contracts the softmax weights over keys: content @ weights."""
    torch = _upstream()
    from demucs.demucs import LocalState as TorchLocalState

    from demucs_mlx.mlx_demucs import LocalState

    channels, heads, frames = 8, 2, 37
    reference = TorchLocalState(channels, heads=heads, nfreqs=nfreqs, ndecay=4).eval()
    model = LocalState(channels, heads=heads, nfreqs=nfreqs, ndecay=4)
    names = ["content", "query", "key", "proj", "query_decay"] + (["query_freqs"] if nfreqs else [])
    for name in names:
        conv = getattr(reference, name)
        target = getattr(model, name).conv
        target.weight = mx.asarray(conv.weight.detach().contiguous()).transpose(0, 2, 1)
        target.bias = mx.asarray(conv.bias.detach().contiguous())
    x = _mix(frames / 44_100)[:, :, :frames]
    x = mx.concatenate([x] * (channels // 2), axis=1)
    with torch.no_grad():
        expected = mx.asarray(reference(_to_torch(x)).contiguous())
    assert _snr_db(expected - x, model(x) - x) > 100


def _thread_result(fn):
    import threading

    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.mark.parametrize("architecture", ["htdemucs", "hdemucs"])
def test_model_runs_on_a_thread_other_than_the_one_that_built_it(architecture):
    """Side streams and cached arrays must not be bound to the first thread."""
    if architecture == "htdemucs":
        from demucs_mlx.mlx_htdemucs import HTDemucsMLX

        model = HTDemucsMLX(SOURCES, channels=8, depth=4, segment=1, bottom_channels=16,
                            t_layers=2, t_heads=2)
    else:
        from demucs_mlx.mlx_hdemucs import HDemucsMLX

        model = HDemucsMLX(SOURCES, channels=8, depth=6, segment=4, cac=True, norm_starts=4,
                           dconv_attn=4, dconv_lstm=4)
    model.eval()
    mx.eval(model.parameters())
    mix = _mix(1.3)

    def separate():
        out = apply_model(model, mix, shifts=0, split=True, overlap=0.25, batch_size=2)
        mx.eval(out)
        return out

    expected = separate()
    # The first call per shape runs eagerly and later ones compiled, which
    # reassociates float adds; that, not the thread, bounds the agreement.
    assert _snr_db(expected, _thread_result(separate)) > 90
    assert _snr_db(expected, _thread_result(separate)) > 90
