"""Demucs spectral wrappers around ``spectral_ops``.

Re-exports the core ``SpectralTransform`` from ``spectral_ops`` and adds
``spectro`` / ``ispectro`` convenience functions that handle the
multi-dimensional tensor layouts used by Demucs models (3-D for STFT,
4-D and 5-D for iSTFT).
"""

from typing import Optional

import mlx.core as mx

from .spectral_ops import (
    SpectralTransform,
    WindowLike,
    get_transform_mlx,
    resolve_fft_params,
)

__all__ = [
    "SpectralTransform",
    "spectro",
    "ispectro",
]


def spectro(
    x: mx.array,
    *,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: WindowLike = None,
    window_fn: str = "hann",
    periodic: bool = True,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    return_complex: bool = True,
    pad: int = 0,
    torch_like: bool = False,
) -> mx.array:
    """Torch-compatible STFT with Demucs multi-dim support.

    Input shapes:
    - [T]         → output [F, N]
    - [B, T]      → output [B, F, N]
    - [B, C, T]   → output [B, C, F, N]   (Demucs layout)
    """
    if not onesided:
        raise NotImplementedError("Only onesided=True supported")
    if not return_complex:
        raise NotImplementedError("Only return_complex=True supported")

    eff_n_fft, hop, win = resolve_fft_params(
        int(n_fft), hop_length, win_length, int(pad),
    )

    # Torch reflect padding constraint
    if torch_like and center:
        pad_amt = eff_n_fft // 2
        if int(x.shape[-1]) <= pad_amt:
            raise RuntimeError(
                f"stft: reflect padding requires input length > "
                f"eff_n_fft//2 (len={int(x.shape[-1])}, pad={pad_amt})."
            )

    transform = get_transform_mlx(
        n_fft=eff_n_fft,
        hop_length=hop,
        win_length=win,
        window_fn=window_fn,
        periodic=periodic,
        center=center,
        normalized=normalized,
        window=window,
    )

    # --- Demucs 3-D layout: [B, C, T] → reshape to [B*C, T] ---
    if x.ndim == 3:
        B, C, T = x.shape
        x2 = mx.contiguous(x).reshape(B * C, T)
        spec2 = transform.stft(x2)
        # [B*C, F, N] → [B, C, F, N]
        return spec2.reshape(B, C, spec2.shape[1], spec2.shape[2])

    # 1-D or 2-D: handled by SpectralTransform.stft directly
    orig_1d = x.ndim == 1
    if orig_1d:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"spectro expects [T], [B,T], or [B,C,T], got {x.shape}")

    spec = transform.stft(x)
    return mx.squeeze(spec, axis=0) if orig_1d else spec


def ispectro(
    z: mx.array,
    *,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: WindowLike = None,
    window_fn: str = "hann",
    periodic: bool = True,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    return_complex: bool = False,
    length: Optional[int] = None,
    pad: int = 0,
    torch_like: bool = False,
    safety: str = "auto",
    allow_fused: bool = True,
) -> mx.array:
    """Torch-compatible iSTFT with Demucs multi-dim support.

    Input shapes:
    - [F, N]         → output [T]
    - [B, F, N]      → output [B, T]
    - [B, C, F, N]   → output [B, C, T]       (Demucs layout)
    - [B, S, C, F, N] → output [B, S, C, T]   (Demucs bag layout)
    """
    if hop_length is None:
        raise ValueError("hop_length required")
    if not onesided:
        raise NotImplementedError("Only onesided=True supported")
    if return_complex:
        raise NotImplementedError("Only real output supported")

    hop = int(hop_length)

    if n_fft is None:
        # Infer base n_fft from frequency bins for onesided rfft.
        Fbins = int(z.shape[-2])
        n_fft = (Fbins - 1) * 2

    eff_n_fft, hop, win = resolve_fft_params(
        int(n_fft), hop, win_length, int(pad),
    )

    transform = get_transform_mlx(
        n_fft=eff_n_fft,
        hop_length=hop,
        win_length=win,
        window_fn=window_fn,
        periodic=periodic,
        center=center,
        normalized=normalized,
        window=window,
    )

    istft_kw = dict(
        length=length,
        torch_like=bool(torch_like),
        allow_fused=bool(allow_fused),
        safety=safety,
    )

    # --- Demucs 5-D layout: [B, S, C, F, N] ---
    if z.ndim == 5:
        B, S, C, F, N = z.shape
        z2 = mx.contiguous(z).reshape(B * S * C, F, N)
        wav2 = transform.istft(z2, **istft_kw)
        return wav2.reshape(B, S, C, wav2.shape[1])

    # --- Demucs 4-D layout: [B, C, F, N] ---
    if z.ndim == 4:
        B, C, F, N = z.shape
        z2 = mx.contiguous(z).reshape(B * C, F, N)
        wav2 = transform.istft(z2, **istft_kw)
        return wav2.reshape(B, C, wav2.shape[1])

    # 2-D or 3-D: handled by SpectralTransform.istft
    orig_2d = z.ndim == 2
    if orig_2d:
        z = z[None, :, :]
    elif z.ndim != 3:
        raise ValueError(
            f"ispectro expects [F,N], [B,F,N], [B,C,F,N], "
            f"or [B,S,C,F,N], got {z.shape}"
        )

    wav = transform.istft(z, **istft_kw)
    return wav[0] if orig_2d else wav
