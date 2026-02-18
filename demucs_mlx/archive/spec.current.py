import time
from typing import Optional, Tuple, Literal, Dict, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

PadMode = Literal["reflect", "constant", "replicate"]

class SpectralTransform(nn.Module):
    """
    Production-Ready Optimized STFT/ISTFT for M-Series (MPS).
    
    Optimizations:
    - F.fold for OLA Signal (Faster than scatter_add)
    - ConvTranspose1d for OLA Envelope (Memory Efficient)
    - LRU Cache & Hardened Validation
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window_fn: str = "hann",
        center: bool = True,
        normalized: bool = False,
        pad_mode: PadMode = "reflect",
        cache_size: int = 8,
    ):
        super().__init__()
        if n_fft <= 0: raise ValueError("n_fft must be > 0")
        if hop_length <= 0: raise ValueError("hop_length must be > 0")

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        
        # [Feedback 1] Guard win_length
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) must be <= n_fft ({self.n_fft})")

        self.center = bool(center)
        self.normalized = bool(normalized)
        self.pad_mode = pad_mode

        # Primary window buffer
        window = self._create_window(window_fn, self.win_length, self.n_fft, dtype=torch.float32)
        self.register_buffer("window", window)

        # LRU Cache
        self._window_cache: OrderedDict[Tuple[str, torch.dtype], torch.Tensor] = OrderedDict()
        self._max_cache_size = cache_size

        # OLA Envelope State
        self.register_buffer("ola_denom", None, persistent=False)
        self.register_buffer("_cached_kernel", None, persistent=False)
        self._ola_cache_key: Optional[Tuple[int, int, int, int, torch.dtype]] = None

    def forward(self, x: torch.Tensor, inverse: bool = False, length: Optional[int] = None) -> torch.Tensor:
        if inverse:
            return self.istft(x, length=length)
        return self.stft(x)

    @staticmethod
    def _pad_widths(n_fft: int, win_length: int) -> Tuple[int, int]:
        left = (n_fft - win_length) // 2
        right = (n_fft - win_length + 1) // 2
        return int(left), int(right)

    def _create_window(self, window_fn: str, win_length: int, n_fft: int, *, dtype: torch.dtype) -> torch.Tensor:
        if window_fn == "hann":
            window = torch.hann_window(win_length, periodic=True, dtype=dtype, device="cpu")
        elif window_fn == "hamming":
            window = torch.hamming_window(win_length, periodic=True, dtype=dtype, device="cpu")
        else:
            raise ValueError(f"Unsupported window: {window_fn}")

        left, right = self._pad_widths(n_fft, win_length)
        if left > 0 or right > 0:
            window = F.pad(window, (left, right), mode="constant", value=0.0)
        return window

    def _get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.window.device == device and self.window.dtype == dtype:
            return self.window
        
        # Safe cache key
        key = (str(device), dtype)

        if key in self._window_cache:
            self._window_cache.move_to_end(key)
            return self._window_cache[key]

        w = self.window.to(device=device, dtype=dtype)
        self._window_cache[key] = w
        
        if len(self._window_cache) > self._max_cache_size:
            self._window_cache.popitem(last=False)
            
        return w

    def _update_ola_cache(self, frames: int, device: torch.device, dtype: torch.dtype) -> int:
        frames = int(frames)
        out_len = self.hop_length * (frames - 1) + self.n_fft
        
        dev_id = device.index if device.index is not None else -1
        key = (frames, out_len, device.type, dev_id, dtype)

        if self.ola_denom is not None and self._ola_cache_key == key:
            return out_len

        w = self._get_window(device, dtype)
        
        if self._cached_kernel is None or self._cached_kernel.dtype != dtype or self._cached_kernel.device != device:
            self._cached_kernel = (w * w).view(1, 1, self.n_fft)

        # [Feedback 5 (Partial)] Allocating small ones tensor is cheap, but could be cached if desired.
        # Keeping it simple for now as it's O(1) memory relative to audio length.
        ones = torch.ones((1, 1, frames), dtype=dtype, device=device)
        
        # [Feedback 2] Safe Squeeze
        # Avoid .squeeze() which might kill dimensions if out_len == 1
        denom = F.conv_transpose1d(
            ones, 
            self._cached_kernel, 
            stride=self.hop_length
        ).squeeze(0).squeeze(0)

        self.ola_denom = denom
        self._ola_cache_key = key
        return out_len

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_complex(): raise TypeError("stft expects real input")
        if x.numel() == 0: raise ValueError("Input signal has zero length")
        
        *other, T = x.shape
        x2 = x.reshape(-1, T)
        
        if not x2.is_contiguous(): x2 = x2.contiguous()

        if self.center:
            pad = self.n_fft // 2
            if self.pad_mode == "reflect" and T < pad:
                 x2 = F.pad(x2, (pad, pad), mode="constant", value=0.0)
            else:
                 x2 = F.pad(x2, (pad, pad), mode=self.pad_mode)

        frames = x2.unfold(-1, self.n_fft, self.hop_length)
        
        w = self._get_window(x2.device, x2.dtype)
        frames = (frames * w.view(1, 1, self.n_fft))
        if not frames.is_contiguous(): frames = frames.contiguous()

        norm = "ortho" if self.normalized else None
        Z = torch.fft.rfft(frames, n=self.n_fft, dim=-1, norm=norm)
        
        Z = Z.transpose(1, 2)
        return Z.view(*other, Z.shape[-2], Z.shape[-1])

    def istft(self, z: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if not z.is_complex():
            if z.shape[-1] == 2: z = torch.complex(z[..., 0], z[..., 1])
            else: raise ValueError("istft expects complex input")

        *other, freqs, frames = z.shape
        
        # [Feedback 4] Sanity Check Freqs
        expected_freqs = self.n_fft // 2 + 1
        if freqs != expected_freqs:
            raise ValueError(f"freqs dim ({freqs}) != expected ({expected_freqs}) for n_fft={self.n_fft}")

        z2 = z.reshape(-1, freqs, frames)

        norm = "ortho" if self.normalized else None
        time_frames = torch.fft.irfft(z2.transpose(1, 2).contiguous(), n=self.n_fft, dim=-1, norm=norm)

        w = self._get_window(z2.device, time_frames.dtype)
        time_frames = time_frames * w.view(1, 1, self.n_fft)

        out_len = self._update_ola_cache(frames, z2.device, time_frames.dtype)

        tf = time_frames.transpose(1, 2).contiguous()
        out = F.fold(
            tf,
            output_size=(1, out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(2).squeeze(1)

        # [Feedback 3] Dtype-aware epsilon
        denom = self.ola_denom
        eps = torch.finfo(out.dtype).eps
        out = out / denom.clamp_min(eps).unsqueeze(0)

        if self.center:
            pad = self.n_fft // 2
            if length is None: out = out[..., pad:-pad]
            else: out = out[..., pad : pad + length]
        elif length is not None:
            out = out[..., :length]

        return out.view(*other, out.shape[-1])


# ==============================================================================
# Backwards Compatibility Wrappers
# ==============================================================================
# The original demucs codebase used simple spectro()/ispectro() functions.
# These wrappers provide the same API using the optimized SpectralTransform.

_global_transforms = {}


def _get_transform(n_fft, hop_length, normalized, pad):
    """Get or create a cached SpectralTransform instance."""
    win_length = n_fft // (1 + pad) if pad > 0 else n_fft
    key = (n_fft, hop_length, win_length, normalized, pad)
    if key not in _global_transforms:
        _global_transforms[key] = SpectralTransform(
            n_fft=n_fft * (1 + pad),
            hop_length=hop_length,
            win_length=win_length,
            normalized=normalized,
            center=True,
            pad_mode="reflect",
        )
    return _global_transforms[key]


def spectro(x, n_fft=512, hop_length=None, pad=0):
    """
    Compute STFT using optimized SpectralTransform.

    Backwards compatible wrapper for the original spectro() function.
    Now uses the optimized SpectralTransform implementation.

    Args:
        x: Input waveform tensor [..., time]
        n_fft: FFT size
        hop_length: Hop length (default: n_fft // 4)
        pad: Padding factor for FFT size

    Returns:
        Complex spectrogram [..., freqs, frames]
    """
    hop_length = hop_length or n_fft // 4
    normalized = True

    transform = _get_transform(n_fft, hop_length, normalized, pad)

    # Handle MPS/XPU devices (move to CPU if needed for compatibility)
    device = x.device
    is_mps_xpu = device.type in ['mps', 'xpu']
    if is_mps_xpu:
        x = x.cpu()

    z = transform.stft(x)

    if is_mps_xpu:
        z = z.to(device)

    return z


def ispectro(z, hop_length=None, length=None, pad=0):
    """
    Compute inverse STFT using optimized SpectralTransform.

    Backwards compatible wrapper for the original ispectro() function.
    Now uses the optimized SpectralTransform implementation.

    Args:
        z: Complex spectrogram [..., freqs, frames]
        hop_length: Hop length (default: inferred from freqs)
        length: Output length in samples (default: inferred)
        pad: Padding factor that was used in forward STFT

    Returns:
        Reconstructed waveform [..., time]
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    hop_length = hop_length or n_fft // 4
    normalized = True

    transform = _get_transform(n_fft, hop_length, normalized, pad)

    # Handle MPS/XPU devices
    device = z.device
    is_mps_xpu = device.type in ['mps', 'xpu']
    if is_mps_xpu:
        z = z.cpu()

    x = transform.istft(z, length=length)

    if is_mps_xpu:
        x = x.to(device)

    return x