"""Demucs spectral wrappers around ``mlx_spectro``.

Re-exports the core ``SpectralTransform`` from ``mlx_spectro`` and adds
``spectro`` / ``ispectro`` convenience functions that handle the
multi-dimensional tensor layouts used by Demucs models (3-D for STFT,
4-D and 5-D for iSTFT).

``CachedSpectralPair`` caches the transform and lazily creates
``compiled_pair()`` instances for repeated chunk sizes, giving 1.3–1.7x
speedup over the eager path.
"""

from typing import Optional

import mlx.core as mx
from mlx_spectro import (
    SpectralTransform,
    WindowLike,
    get_transform_mlx,
    resolve_fft_params,
)

__all__ = [
    "CachedSpectralPair",
    "SpectralTransform",
    "spectro",
    "ispectro",
]


class CachedSpectralPair:
    """Cached STFT/iSTFT transform with compiled_pair() acceleration.

    Creates the ``SpectralTransform`` once and reuses it.  For repeated
    chunk sizes (the common case in Demucs inference), lazily builds and
    caches ``compiled_pair()`` graphs that eliminate Python dispatch overhead.

    Handles the Demucs multi-dimensional reshape internally:
      - stft:  [B, C, T]   → [B*C, T]  → stft → [B, C, F, N]
      - istft: [B, C, F, N] → [B*C, F, N] → istft → [B, C, T]
               [B, S, C, F, N] → [B*S*C, F, N] → istft → [B, S, C, T]
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: Optional[int] = None,
    ) -> None:
        eff_n_fft, hop, win = resolve_fft_params(n_fft, hop_length, None, 0)
        self._transform = get_transform_mlx(
            n_fft=eff_n_fft,
            hop_length=hop,
            win_length=win,
            window_fn="hann",
            periodic=True,
            center=True,
            normalized=False,
            window=None,
        )
        self._compiled_cache: dict[int, tuple] = {}  # length → (stft_fn, istft_fn)

    def _get_pair(self, length: int, batch: int) -> tuple:
        """Get or create a compiled_pair for the given signal length."""
        if length not in self._compiled_cache:
            pair = self._transform.compiled_pair(
                length=length, layout="bfn", warmup_batch=batch,
            )
            self._compiled_cache[length] = pair
        return self._compiled_cache[length]

    def stft(self, x: mx.array) -> mx.array:
        """STFT with Demucs multi-dim support.

        Input [B, C, T] → output [B, C, F, N].
        Input [B, T]    → output [B, F, N].
        """
        if x.ndim == 3:
            B, C, T = x.shape
            x2 = mx.contiguous(x).reshape(B * C, T)
            stft_fn, _ = self._get_pair(T, B * C)
            spec2 = stft_fn(x2)
            return spec2.reshape(B, C, spec2.shape[1], spec2.shape[2])

        stft_fn, _ = self._get_pair(int(x.shape[-1]), int(x.shape[0]))
        return stft_fn(x)

    def istft(self, z: mx.array, length: int) -> mx.array:
        """iSTFT with Demucs multi-dim support.

        Input [B, S, C, F, N] → output [B, S, C, T].
        Input [B, C, F, N]    → output [B, C, T].
        Input [B, F, N]       → output [B, T].
        """
        if z.ndim == 5:
            B, S, C, F, N = z.shape
            z2 = mx.contiguous(z).reshape(B * S * C, F, N)
            _, istft_fn = self._get_pair(length, B * S * C)
            wav2 = istft_fn(z2)
            return wav2.reshape(B, S, C, wav2.shape[1])

        if z.ndim == 4:
            B, C, F, N = z.shape
            z2 = mx.contiguous(z).reshape(B * C, F, N)
            _, istft_fn = self._get_pair(length, B * C)
            wav2 = istft_fn(z2)
            return wav2.reshape(B, C, wav2.shape[1])

        _, istft_fn = self._get_pair(length, int(z.shape[0]))
        return istft_fn(z)

    def stft_eager(self, x: mx.array) -> mx.array:
        """STFT using cached transform without compiled graphs (fallback)."""
        if x.ndim == 3:
            B, C, T = x.shape
            x2 = mx.contiguous(x).reshape(B * C, T)
            spec2 = self._transform.stft(x2)
            return spec2.reshape(B, C, spec2.shape[1], spec2.shape[2])
        return self._transform.stft(x)

    def istft_eager(self, z: mx.array, length: int) -> mx.array:
        """iSTFT using cached transform without compiled graphs (fallback)."""
        if z.ndim == 5:
            B, S, C, F, N = z.shape
            z2 = mx.contiguous(z).reshape(B * S * C, F, N)
            wav2 = self._transform.istft(z2, length=length)
            return wav2.reshape(B, S, C, wav2.shape[1])
        if z.ndim == 4:
            B, C, F, N = z.shape
            z2 = mx.contiguous(z).reshape(B * C, F, N)
            wav2 = self._transform.istft(z2, length=length)
            return wav2.reshape(B, C, wav2.shape[1])
        return self._transform.istft(z, length=length)


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
