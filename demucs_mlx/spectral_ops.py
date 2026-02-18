"""Reusable MLX spectral ops (STFT/iSTFT building blocks).

Public surface is intentionally small and stable:
`SpectralTransform`, `WindowLike`, `make_window`,
`resolve_fft_params`, `get_transform_mlx`, and `spec_mlx_device_key`.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, Union

import mlx.core as mx

__all__ = [
    "SpectralTransform",
    "ISTFTBackendPolicy",
    "STFTOutputLayout",
    "WindowLike",
    "make_window",
    "resolve_fft_params",
    "get_transform_mlx",
    "spec_mlx_device_key",
    "get_cache_debug_stats",
    "reset_cache_debug_stats",
]

import hashlib
import json
import math
import os
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ISTFTBackendPolicy = Literal["auto", "mlx_fft", "metal", "torch_fallback"]
STFTOutputLayout = Literal["bfn", "bnf"]

_CACHE_STATS_ENABLED = (
    os.environ.get("SPEC_MLX_CACHE_STATS", "0").lower()
    not in ("0", "false", "no", "off")
)
_CACHE_STATS_TRACE = _CACHE_STATS_ENABLED and (
    os.environ.get("SPEC_MLX_CACHE_STATS_TRACE", "0").lower() in ("1", "true", "yes", "on")
)
try:
    _CACHE_STATS_TRACE_MAX_RAW = int(os.environ.get("SPEC_MLX_CACHE_STATS_TRACE_MAX", "128"))
except Exception:
    _CACHE_STATS_TRACE_MAX_RAW = 128
_CACHE_STATS_TRACE_MAX = max(16, _CACHE_STATS_TRACE_MAX_RAW)
_CACHE_STATS_LOCK = threading.Lock()
_CACHE_STATS_COUNTS: Counter[str] = Counter()
_CACHE_STATS_LAST_KEY: Dict[str, str] = {}
_CACHE_STATS_RECENT: deque = deque(maxlen=_CACHE_STATS_TRACE_MAX)


def _summarize_cache_key(key: Any) -> str:
    if isinstance(key, tuple):
        head = ", ".join(type(v).__name__ for v in key[:4])
        return f"tuple(len={len(key)}, types=[{head}])"
    if isinstance(key, dict):
        keys = list(key.keys())[:6]
        return f"dict(keys={keys})"
    if hasattr(key, "shape"):
        shape = getattr(key, "shape", None)
        return f"{type(key).__name__}(shape={tuple(int(d) for d in shape)})"
    return type(key).__name__


if _CACHE_STATS_ENABLED:
    def _record_cache_event(name: str, *, key: Any = None, detail: Optional[str] = None) -> None:
        with _CACHE_STATS_LOCK:
            _CACHE_STATS_COUNTS[name] += 1
            if key is not None:
                _CACHE_STATS_LAST_KEY[name] = _summarize_cache_key(key)
            if _CACHE_STATS_TRACE:
                item = {"event": str(name)}
                if key is not None:
                    item["key"] = _summarize_cache_key(key)
                if detail is not None:
                    item["detail"] = str(detail)
                _CACHE_STATS_RECENT.append(item)
else:
    def _record_cache_event(name: str, *, key: Any = None, detail: Optional[str] = None) -> None:
        return None


def reset_cache_debug_stats() -> None:
    with _CACHE_STATS_LOCK:
        _CACHE_STATS_COUNTS.clear()
        _CACHE_STATS_LAST_KEY.clear()
        _CACHE_STATS_RECENT.clear()


def get_cache_debug_stats(*, reset: bool = False) -> dict:
    """Return lightweight cache/debug counters for runtime tuning.

    Includes cache hit/miss/eviction counters plus representative key-shape hints.
    """
    with _CACHE_STATS_LOCK:
        payload = {
            "enabled": bool(_CACHE_STATS_ENABLED),
            "counts": dict(_CACHE_STATS_COUNTS),
            "last_key_shapes": dict(_CACHE_STATS_LAST_KEY),
            "recent_events": list(_CACHE_STATS_RECENT),
        }
    payload["kernel_cache"] = _KernelCache.debug_snapshot()
    payload["transform_cache"] = dict(_get_transform_cached.cache_info()._asdict())
    if reset:
        reset_cache_debug_stats()
    return payload

def _default_cache_path() -> Path:
    # Allow override
    override = os.environ.get("SPEC_MLX_AUTOTUNE_CACHE_PATH")
    if override:
        return Path(override).expanduser()

    home = Path.home()

    # macOS
    if sys.platform == "darwin":
        base = home / "Library" / "Caches"
    # Windows
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
    # Linux/others
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(home / ".cache")))

    return base / "mlx-spectro" / "spec_mlx_autotune.json"


def _encode_key(device_key: str, kernel_name: str, n_fft: int, hop: int) -> str:
    # Stable, JSON-friendly
    return f"{device_key}||{kernel_name}||{int(n_fft)}||{int(hop)}"


def _decode_key(k: str) -> Optional[Tuple[str, str, int, int]]:
    try:
        device_key, kernel_name, n_fft_s, hop_s = k.split("||", 3)
        return device_key, kernel_name, int(n_fft_s), int(hop_s)
    except Exception:
        return None


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, sort_keys=True)

    tmp.write_text(data, encoding="utf-8")
    # Atomic on POSIX + Windows when replacing in same directory
    tmp.replace(path)


# ==============================================================================
# Autotuning configuration
# ==============================================================================
# Set SPEC_MLX_AUTOTUNE=0 to disable autotuning (fallback to 256).
_SPEC_MLX_AUTOTUNE = (
    os.environ.get("SPEC_MLX_AUTOTUNE", "1").lower()
    not in ("0", "false", "no", "off")
)


def _parse_tgx_int(raw: str) -> Optional[int]:
    try:
        v = int(str(raw).strip())
    except Exception:
        return None
    return v if v > 0 else None


def _resolve_manual_tgx_override(kernel_name: str) -> Optional[int]:
    """Return explicit threadgroup override from SPEC_MLX_TGX, if present.

    Supported forms:
    - "256" (global override)
    - "kernel_name:256,other_kernel:512,*:128" (per-kernel with optional wildcard)
    """
    raw = os.environ.get("SPEC_MLX_TGX", "").strip()
    if not raw:
        return None

    # Simple global override, e.g. SPEC_MLX_TGX=256
    if ":" not in raw and "," not in raw:
        return _parse_tgx_int(raw)

    wildcard: Optional[int] = None
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            tgx = _parse_tgx_int(token)
            if tgx is not None:
                wildcard = tgx
            continue
        k, v = token.split(":", 1)
        k = k.strip()
        tgx = _parse_tgx_int(v)
        if tgx is None:
            continue
        if k == kernel_name:
            return tgx
        if k == "*":
            wildcard = tgx
    return wildcard

@lru_cache(maxsize=1)
def spec_mlx_device_key() -> str:
    """Best-effort device identifier for caching autotune results.

    We keep this extremely defensive because MLX APIs can differ by version.
    """
    # Prefer MLX metal device name if available.
    try:
        metal = getattr(mx, "metal", None)
        if metal is not None:
            name_fn = getattr(metal, "device_name", None)
            if callable(name_fn):
                name = name_fn()
                if name:
                    return str(name)
    except Exception:
        pass

    # Fallback to default_device / current_device if available.
    for attr in ("default_device", "current_device", "device"):
        try:
            fn = getattr(mx, attr, None)
            if callable(fn):
                dev = fn()
                if dev is not None:
                    return str(dev)
        except Exception:
            pass

    # Last resort: backend hint or platform.
    return os.environ.get("MLX_BACKEND", "metal")


# ==============================================================================
# Metal Kernels
# ==============================================================================

# Shared k-bounds macro, prepended to each kernel source.
# Computes the range of overlapping frames [k_min, k_max] for output sample `st`.
# HOP, FRAME are compile-time template constants.
# n_frames and out_len are runtime values loaded from a `params` int32 buffer.
_METAL_K_BOUNDS = """
int k_max = st / HOP;
if (k_max >= n_frames) k_max = n_frames - 1;
int k_min = 0;
{ int target = st - FRAME; if (target >= 0) k_min = (target / HOP) + 1; }
"""

# --- 1) Metal Kernel: Gather-style OLA ---
# Type-polymorphic template; float32 accumulation, cast to T on output.
# Dynamic lengths (n_frames, out_len) are passed via `params` buffer to avoid
# kernel recompilation for every unique audio length.
_METAL_OLA_TEMPLATE = """
int n_frames = params[0];
int out_len = params[1];
int st = (int)thread_position_in_grid.x;
int sb = (int)thread_position_in_grid.y;
if (st >= out_len) return;
""" + _METAL_K_BOUNDS + """
float acc = 0.0f;
int base_offset = sb * n_frames * FRAME;

#pragma unroll 4
for (int k = k_max; k >= k_min; --k) {
    int off = st - k * HOP;
    acc += (float)frames[base_offset + k * FRAME + off] * (float)window[off];
}

out[sb * out_len + st] = (T)acc;
"""

# --- 2) Metal Kernel: Gather-style OLA + normalization ---
# Fused overlap-add with Torch-style masked divide by window² envelope.
_METAL_OLA_NORM_TEMPLATE = """
int n_frames = params[0];
int out_len = params[1];
int st = (int)thread_position_in_grid.x;
int sb = (int)thread_position_in_grid.y;
if (st >= out_len) return;
""" + _METAL_K_BOUNDS + """
float acc = 0.0f;
float den = 0.0f;
int base_offset = sb * n_frames * FRAME;

#pragma unroll 4
for (int k = k_max; k >= k_min; --k) {
    int off = st - k * HOP;
    acc += (float)frames[base_offset + k * FRAME + off] * (float)window[off];
    den += (float)window_sq[off];
}

// Torch-style masked normalization (no epsilon clamp):
// keep zero where overlap-add envelope is effectively zero.
out[sb * out_len + st] = (den > 1.0e-11f) ? (T)(acc / den) : (T)(0.0f);
"""


# --- 3) Metal Kernel: OLA Envelope ---
# Computes window² overlap-add envelope (1-D, no batch dimension).
_METAL_OLA_ENVELOPE_TEMPLATE = """
int n_frames = params[0];
int out_len = params[1];
int st = (int)thread_position_in_grid.x;
if (st >= out_len) return;
""" + _METAL_K_BOUNDS + """
float den = 0.0f;

#pragma unroll 4
for (int k = k_min; k <= k_max; ++k) {
    int off = st - k * HOP;
    den += (float)window_sq[off];
}

out[st] = (T)den;
"""

# --- 4) Metal Kernel: Fused windowed frame extraction ---
# Combines reflect-pad + as_strided + window multiply into a single pass.
# Reads directly from the unpadded signal with reflect-boundary indexing,
# multiplies by the analysis window, and writes windowed frames.
# Eliminates the padded-signal intermediate allocation entirely.
#
# Two variants:
#   Simple — Grid: (n_fft, n_frames, B).  One thread per (fft_bin, frame, batch).
#            Each thread reads one sample from global memory independently.
#   Tiled  — One threadgroup per tile of consecutive frames.  Loads the shared
#            signal chunk into threadgroup memory once, then each thread reads
#            from fast shared memory.  ~NFFT/HOP data reuse ratio (typically 4×).
#            Faster when the workload is bandwidth-bound (large B × sig_len).

_METAL_FUSED_FRAME_EXTRACT_TEMPLATE = """
int sig_len = params[0];
int n_frames = params[1];
int f_idx = (int)thread_position_in_grid.x;
int n_idx = (int)thread_position_in_grid.y;
int b_idx = (int)thread_position_in_grid.z;
if (f_idx >= NFFT || n_idx >= n_frames) return;

// Position in the (virtual) padded signal
int src_pos = n_idx * HOP + f_idx;

// Reflect-pad boundary: map to original signal index
int orig_idx;
if (src_pos < PAD) {
    orig_idx = PAD - src_pos;
} else if (src_pos < PAD + sig_len) {
    orig_idx = src_pos - PAD;
} else {
    orig_idx = sig_len - 2 - (src_pos - PAD - sig_len);
}

float val = (float)signal[b_idx * sig_len + orig_idx] * (float)win[f_idx];
out[b_idx * n_frames * NFFT + n_idx * NFFT + f_idx] = (T)val;
"""

_METAL_TILED_FRAME_EXTRACT_TEMPLATE = """
// Threadgroup shared memory for the signal chunk covering this tile of frames.
// CHUNK_LEN = (TILE_FRAMES - 1) * HOP + NFFT, sized at compile time.
threadgroup float shared_buf[CHUNK_LEN];

int sig_len = params[0];
int n_frames = params[1];

int local_x = (int)thread_position_in_threadgroup.x;
int local_y = (int)thread_position_in_threadgroup.y;
int tg_x_idx = (int)threadgroup_position_in_grid.x;
int b_idx = (int)threadgroup_position_in_grid.z;

// Frame range for this tile
int frame_start = tg_x_idx * TILE_FRAMES;
int frame_end = frame_start + TILE_FRAMES;
if (frame_end > n_frames) frame_end = n_frames;
int n_tile_frames = frame_end - frame_start;

// Virtual-padded range this tile needs
int sig_start = frame_start * HOP;
int sig_end_val = (frame_end - 1) * HOP + NFFT;
int chunk_len = sig_end_val - sig_start;

// Cooperative load: all threads load signal into shared memory
int n_threads = TG_X * TG_Y;
int my_id = local_y * TG_X + local_x;
for (int i = my_id; i < chunk_len; i += n_threads) {
    int src_pos = sig_start + i;
    int orig_idx;
    if (src_pos < PAD) {
        orig_idx = PAD - src_pos;
    } else if (src_pos < PAD + sig_len) {
        orig_idx = src_pos - PAD;
    } else {
        orig_idx = sig_len - 2 - (src_pos - PAD - sig_len);
    }
    shared_buf[i] = (float)signal[b_idx * sig_len + orig_idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Each thread handles one frame in the tile, looping over fft bins
int tile_frame = local_y;
if (tile_frame < n_tile_frames) {
    int n_idx = frame_start + tile_frame;
    for (int f = local_x; f < NFFT; f += TG_X) {
        int local_pos = tile_frame * HOP + f;
        float val = shared_buf[local_pos] * (float)win[f];
        out[b_idx * n_frames * NFFT + n_idx * NFFT + f] = (T)val;
    }
}
"""

# Minimum output bytes (B * n_frames * n_fft * 4) before using the tiled kernel.
# Below this threshold the workload is latency-bound and tiling adds no benefit.
# Benchmarked on M4 Max: tiled wins consistently above ~100 MB, is neutral at
# ~60 MB, and can regress below that.  100 MB is a safe crossover point.
_TILED_FRAME_EXTRACT_BYTE_THRESHOLD = 100_000_000  # ~100 MB


class _FrameExtractCache:
    """Singleton for the fused frame-extraction Metal kernels (simple + tiled)."""
    _lock = threading.Lock()
    _simple: object = None
    _tiled: object = None

    @classmethod
    def get_simple(cls):
        if cls._simple is not None:
            return cls._simple
        with cls._lock:
            if cls._simple is not None:
                return cls._simple
            try:
                cls._simple = mx.fast.metal_kernel(
                    name="fused_frame_extract",
                    input_names=["signal", "win", "params"],
                    output_names=["out"],
                    source=_METAL_FUSED_FRAME_EXTRACT_TEMPLATE,
                )
            except Exception:
                cls._simple = False
            return cls._simple

    @classmethod
    def get_tiled(cls):
        if cls._tiled is not None:
            return cls._tiled
        with cls._lock:
            if cls._tiled is not None:
                return cls._tiled
            try:
                cls._tiled = mx.fast.metal_kernel(
                    name="tiled_frame_extract",
                    input_names=["signal", "win", "params"],
                    output_names=["out"],
                    source=_METAL_TILED_FRAME_EXTRACT_TEMPLATE,
                )
            except Exception:
                cls._tiled = False
            return cls._tiled

    @classmethod
    def get(cls):
        """Backward-compat: return the simple kernel."""
        return cls.get_simple()

    @classmethod
    def tile_params(cls, n_fft: int, hop_length: int) -> Optional[tuple]:
        """Compute tiling parameters that fit in 32 KB threadgroup memory.

        Returns (tile_frames, tg_x, tg_y, chunk_len) or None if tiling
        is not possible for this config.
        """
        max_shared_floats = 32768 // 4  # 8192 floats in 32 KB
        tile_frames = (max_shared_floats - n_fft) // hop_length + 1
        if tile_frames < 2:
            return None  # no reuse benefit with 1 frame per tile
        tg_x = min(256, n_fft)
        tg_y = tile_frames
        # Metal: max 1024 threads per threadgroup
        if tg_x * tg_y > 1024:
            tg_y = 1024 // tg_x
            tile_frames = tg_y
        if tile_frames < 2:
            return None
        chunk_len = (tile_frames - 1) * hop_length + n_fft
        return (tile_frames, tg_x, tg_y, chunk_len)


# Template-tuple tracking for params_cache.* debug events.
# Preserves backward compatibility with test assertions that check these counters.
_last_tmpl: Dict[str, tuple] = {}

def _record_tmpl_event(kernel_tag: str, tmpl: list) -> None:
    """Emit params_cache.hit / params_cache.miss based on template-tuple reuse."""
    key = tuple(tmpl)
    prev = _last_tmpl.get(kernel_tag)
    if prev == key:
        _record_cache_event("params_cache.hit", key=key, detail=kernel_tag)
    else:
        _record_cache_event("params_cache.miss", key=key, detail=kernel_tag)
        _last_tmpl[kernel_tag] = key


# ==============================================================================
# Kernel Cache Manager
# ==============================================================================

class _KernelCache:
    """Singleton to manage Metal kernel compilation and caching."""

    _lock = threading.Lock()

    _ola_kernel = None
    _ola_norm_kernel = None
    _envelope_kernel = None

    # Cache of autotuned threadgroup sizes:
    # key: (device_key, kernel_name, n_fft, hop) -> int threadgroup_x
    _tgx_cache: Dict[Tuple[str, str, int, int], int] = {}
    _tgx_cache_loaded: bool = False
    _tgx_cache_dirty: bool = False

    @classmethod
    def _max_entries(cls) -> int:
        raw = os.environ.get("SPEC_MLX_AUTOTUNE_MAX_ENTRIES", "").strip()
        try:
            v = int(raw) if raw else 500
        except Exception:
            v = 500
        return max(32, v)

    @classmethod
    def _trim_cache_locked(cls) -> None:
        # Keep cache bounded to avoid unbounded growth across many experiments.
        max_entries = cls._max_entries()
        extra = len(cls._tgx_cache) - max_entries
        if extra <= 0:
            return
        _record_cache_event("kernel_cache.tgx.evict", detail=f"count={int(extra)}")
        for k in list(cls._tgx_cache.keys())[:extra]:
            cls._tgx_cache.pop(k, None)

    @classmethod
    def _persist_enabled(cls) -> bool:
        return (
            os.environ.get("SPEC_MLX_AUTOTUNE_PERSIST", "1").lower()
            not in ("0", "false", "no", "off")
        )

    @classmethod
    def _load_tgx_cache_from_disk(cls) -> None:
        # Caller is expected to hold cls._lock.
        if cls._tgx_cache_loaded:
            return
        cls._tgx_cache_loaded = True

        if not cls._persist_enabled():
            _record_cache_event("kernel_cache.tgx.load.skipped_persist_off")
            return

        path = _default_cache_path()
        if not path.exists():
            _record_cache_event("kernel_cache.tgx.load.miss_file", detail=str(path))
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 1:
                _record_cache_event("kernel_cache.tgx.load.version_mismatch")
                return
            raw = payload.get("tgx_cache", {})
            if not isinstance(raw, dict):
                _record_cache_event("kernel_cache.tgx.load.bad_payload")
                return

            for k_str, v in raw.items():
                decoded = _decode_key(k_str)
                if decoded is None:
                    continue
                if not isinstance(v, int):
                    continue
                cls._tgx_cache[decoded] = int(v)
            cls._trim_cache_locked()
            _record_cache_event("kernel_cache.tgx.load.ok", detail=f"entries={len(cls._tgx_cache)}")
        except Exception:
            # Cache corruption or permission errors should never break execution.
            _record_cache_event("kernel_cache.tgx.load.error")
            return

    @classmethod
    def _save_tgx_cache_to_disk(cls) -> None:
        # Caller is expected to hold cls._lock.
        if not cls._persist_enabled():
            _record_cache_event("kernel_cache.tgx.save.skipped_persist_off")
            return
        if not cls._tgx_cache_dirty:
            return

        path = _default_cache_path()
        try:
            obj = {
                "version": 1,
                "tgx_cache": {
                    _encode_key(d, k, n, h): int(v)
                    for (d, k, n, h), v in cls._tgx_cache.items()
                },
            }
            _atomic_write_json(path, obj)
            cls._tgx_cache_dirty = False
            _record_cache_event("kernel_cache.tgx.save.ok", detail=f"entries={len(cls._tgx_cache)}")
        except Exception:
            # I/O errors should not break execution.
            _record_cache_event("kernel_cache.tgx.save.error")
            return

    @classmethod
    def get_threadgroup_x(
        cls, device_key: str, kernel_name: str, n_fft: int, hop: int,
    ) -> Optional[int]:
        key = (device_key, kernel_name, int(n_fft), int(hop))
        # Fast path: once loaded, lock-free reads avoid per-call lock overhead
        # on hot steady-state inference paths.
        if cls._tgx_cache_loaded:
            val = cls._tgx_cache.get(key)
            _record_cache_event(
                "kernel_cache.tgx.hit" if val is not None else "kernel_cache.tgx.miss",
                key=key,
            )
            return val
        with cls._lock:
            cls._load_tgx_cache_from_disk()
            val = cls._tgx_cache.get(key)
            _record_cache_event(
                "kernel_cache.tgx.hit" if val is not None else "kernel_cache.tgx.miss",
                key=key,
            )
            return val

    @classmethod
    def set_threadgroup_x(
        cls, device_key: str, kernel_name: str,
        n_fft: int, hop: int, tgx: int,
    ) -> None:
        key = (device_key, kernel_name, int(n_fft), int(hop))
        with cls._lock:
            cls._load_tgx_cache_from_disk()
            cls._tgx_cache[key] = int(tgx)
            cls._trim_cache_locked()
            cls._tgx_cache_dirty = True
        _record_cache_event("kernel_cache.tgx.store", key=key)

    @classmethod
    def autotune_threadgroup_x(
        cls,
        *,
        kernel,
        kernel_name: str,
        n_fft: int,
        hop: int,
        grid: tuple,
        inputs: list,
        output_shape: tuple,
        output_dtype,
        template: list = None,
        default_tgx: int = 256,
        # UPDATED: Steps of 32 cover SIMD boundaries better than just powers of 2.
        # This checks 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, ...
        candidates: tuple = (
            32, 64, 96, 128, 160, 192, 224, 256,
            320, 384, 448, 512, 768, 1024
        ),
        warmup: int = 2,
        iters: int = 8,
    ) -> int:
        """Autotune threadgroup.x for a specific kernel + (n_fft, hop) on the current device.

        Runs only on first use; results are cached in-memory and can be persisted to disk.
        Controlled by SPEC_MLX_AUTOTUNE (set to 0/false to disable; falls back to default_tgx).
        Persistence controlled by SPEC_MLX_AUTOTUNE_PERSIST (default enabled).
        SPEC_MLX_TGX can force explicit tgx (global or per-kernel mapping).
        SPEC_MLX_AUTOTUNE_MAX_ENTRIES bounds persistent cache growth.
        """
        manual_tgx = _resolve_manual_tgx_override(kernel_name)
        if manual_tgx is not None:
            return int(manual_tgx)

        grid_x = int(grid[0]) if len(grid) > 0 else 0

        def _clamp_tgx_for_grid(tgx: int) -> int:
            tgx = int(tgx)
            if tgx <= 0:
                tgx = 32
            # Enforce Apple/Metal constraints and SIMD alignment.
            tgx = min(max(tgx, 32), 1024)
            tgx = (tgx // 32) * 32
            if tgx < 32:
                tgx = 32
            max_tgx = 1024
            if grid_x > 0:
                max_tgx = max(32, min(1024, int(grid_x)))
                max_tgx = (max_tgx // 32) * 32
                if max_tgx < 32:
                    max_tgx = 32
            return min(tgx, max_tgx)

        clamped_default_tgx = _clamp_tgx_for_grid(int(default_tgx))

        # Very short outputs are dominated by launch overhead; avoid persisting
        # tiny-shape tuning results into the global (n_fft, hop) cache.
        if grid_x > 0 and grid_x < int(default_tgx):
            _record_cache_event("kernel_cache.tgx.short_grid_default", detail=f"grid_x={grid_x}")
            return int(clamped_default_tgx)

        if not _SPEC_MLX_AUTOTUNE:
            return int(clamped_default_tgx)

        device_key = spec_mlx_device_key()
        cached = cls.get_threadgroup_x(device_key, kernel_name, n_fft, hop)
        if cached is not None:
            return int(_clamp_tgx_for_grid(int(cached)))

        # Helper: one timing run for a given tgx
        def _time_one(tgx: int) -> float:
            call_kwargs = dict(
                inputs=inputs,
                output_shapes=[output_shape],
                output_dtypes=[output_dtype],
                grid=grid,
                threadgroup=(int(tgx), 1, 1),
            )
            if template is not None:
                call_kwargs["template"] = template

            for _ in range(warmup):
                y = kernel(**call_kwargs)[0]
                mx.eval(y)

            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                y = kernel(**call_kwargs)[0]
                mx.eval(y)
            mx.synchronize()
            t1 = time.perf_counter()
            return (t1 - t0) / max(iters, 1)

        benchmark_candidates = sorted(
            {
                _clamp_tgx_for_grid(int(tgx))
                for tgx in candidates
                if int(tgx) > 0
            }
        )
        benchmark_candidates = [tgx for tgx in benchmark_candidates if tgx >= 32]
        if not benchmark_candidates:
            benchmark_candidates = [int(clamped_default_tgx)]

        best_tgx = int(clamped_default_tgx)
        best_time = None

        # Benchmark candidate tg sizes; skip invalid ones gracefully.
        for tgx in benchmark_candidates:
            try:
                dt = _time_one(tgx)
            except Exception:
                continue
            
            if best_time is None or dt < best_time:
                best_time = dt
                best_tgx = tgx

        # Persist result.
        cls.set_threadgroup_x(device_key, kernel_name, n_fft, hop, best_tgx)
        with cls._lock:
            cls._save_tgx_cache_to_disk()
        return int(best_tgx)

    @classmethod
    def get_ola(cls, dtype=None):
        """Return the OLA Metal kernel (compile once, dtype-agnostic)."""
        if cls._ola_kernel is not None:
            _record_cache_event("kernel_cache.ola.hit")
            return cls._ola_kernel
        _record_cache_event("kernel_cache.ola.miss")
        with cls._lock:
            if cls._ola_kernel is not None:
                return cls._ola_kernel
            try:
                cls._ola_kernel = mx.fast.metal_kernel(
                    name="ola_windowed_optimized",
                    input_names=["frames", "window", "params"],
                    output_names=["out"],
                    source=_METAL_OLA_TEMPLATE,
                )
                _record_cache_event("kernel_cache.ola.compile_ok")
            except Exception:
                cls._ola_kernel = False
                _record_cache_event("kernel_cache.ola.compile_fail")
            return cls._ola_kernel

    @classmethod
    def get_ola_norm(cls, dtype=None):
        """Return the OLA+norm Metal kernel (compile once, dtype-agnostic)."""
        if cls._ola_norm_kernel is not None:
            _record_cache_event("kernel_cache.ola_norm.hit")
            return cls._ola_norm_kernel
        _record_cache_event("kernel_cache.ola_norm.miss")
        with cls._lock:
            if cls._ola_norm_kernel is not None:
                return cls._ola_norm_kernel
            try:
                cls._ola_norm_kernel = mx.fast.metal_kernel(
                    name="ola_norm_windowed_div_envelope",
                    input_names=["frames", "window", "window_sq", "params"],
                    output_names=["out"],
                    source=_METAL_OLA_NORM_TEMPLATE,
                )
                _record_cache_event("kernel_cache.ola_norm.compile_ok")
            except Exception:
                cls._ola_norm_kernel = False
                _record_cache_event("kernel_cache.ola_norm.compile_fail")
            return cls._ola_norm_kernel

    @classmethod
    def get_envelope(cls, dtype=None):
        """Return the envelope Metal kernel (compile once, dtype-agnostic)."""
        if cls._envelope_kernel is not None:
            _record_cache_event("kernel_cache.envelope.hit")
            return cls._envelope_kernel
        _record_cache_event("kernel_cache.envelope.miss")
        with cls._lock:
            if cls._envelope_kernel is not None:
                return cls._envelope_kernel
            try:
                cls._envelope_kernel = mx.fast.metal_kernel(
                    name="ola_envelope",
                    input_names=["window_sq", "params"],
                    output_names=["out"],
                    source=_METAL_OLA_ENVELOPE_TEMPLATE,
                )
                _record_cache_event("kernel_cache.envelope.compile_ok")
            except Exception:
                cls._envelope_kernel = False
                _record_cache_event("kernel_cache.envelope.compile_fail")
            return cls._envelope_kernel

    @classmethod
    def debug_snapshot(cls) -> dict:
        def _is_compiled(v):
            return v is not None and v is not False

        with cls._lock:
            return {
                "ola_kernel_compiled": _is_compiled(cls._ola_kernel),
                "ola_norm_kernel_compiled": _is_compiled(cls._ola_norm_kernel),
                "envelope_kernel_compiled": _is_compiled(cls._envelope_kernel),
                "tgx_entries": int(len(cls._tgx_cache)),
                "tgx_loaded": bool(cls._tgx_cache_loaded),
                "tgx_dirty": bool(cls._tgx_cache_dirty),
            }

def _run_metal_ola(
    frames: mx.array,
    window: mx.array,
    hop: int,
    out_len: int,
    *,
    require_metal: bool = False,
) -> mx.array:
    """Overlap-add using optimized branchless Metal kernel."""
    def _scatter_plan(nframe: int, frame: int, out_len_i: int) -> tuple[mx.array, mx.array]:
        out_len_i = int(out_len_i)
        frame_starts = mx.arange(int(nframe), dtype=mx.int32) * int(hop)
        sample_idx = mx.arange(int(frame), dtype=mx.int32)
        indices = frame_starts[:, None] + sample_idx[None, :]
        flat_indices = indices.reshape(-1)
        valid_mask = flat_indices < out_len_i
        if out_len_i <= 0:
            # Keep shapes consistent for degenerate requests.
            return mx.zeros_like(flat_indices), valid_mask
        clipped = mx.clip(flat_indices, 0, out_len_i - 1)
        return clipped, valid_mask

    def _run_fallback(frames_local: mx.array, window_local: mx.array, out_len_i: int) -> mx.array:
        B, nframe, frame = frames_local.shape
        clipped, valid_mask = _scatter_plan(int(nframe), int(frame), int(out_len_i))

        values = (frames_local * window_local[None, None, :]).reshape(int(B), -1)
        values = values * valid_mask.astype(values.dtype)[None, :]

        if int(out_len_i) <= 0:
            return mx.zeros((int(B), 0), dtype=frames_local.dtype)

        batch_offsets = (mx.arange(int(B), dtype=mx.int32) * int(out_len_i))[:, None]
        scatter_indices = (batch_offsets + clipped[None, :]).reshape(-1)

        out = mx.zeros((int(B) * int(out_len_i),), dtype=frames_local.dtype)
        out = out.at[scatter_indices].add(values.reshape(-1))
        return out.reshape(int(B), int(out_len_i))

    if frames.dtype != window.dtype:
        window = window.astype(frames.dtype)

    B, nframe, frame = frames.shape
    kernel = _KernelCache.get_ola()
    if kernel is False:
        _record_cache_event("backend.ola.fallback", detail="metal_unavailable")
        if require_metal:
            raise RuntimeError(
                "backend_policy='metal' requires Metal OLA kernel, but it is unavailable."
            )
        return _run_fallback(frames, window, int(out_len))
    frames = mx.contiguous(frames)
    window = mx.contiguous(window)

    params = mx.array([int(nframe), int(out_len)], dtype=mx.int32)
    tmpl = [("T", frames.dtype), ("HOP", hop), ("FRAME", frame)]
    _record_tmpl_event("ola", tmpl)

    tgx = _KernelCache.autotune_threadgroup_x(
        kernel=kernel,
        kernel_name=f"ola_windowed_optimized_{frames.dtype}",
        n_fft=frame,
        hop=hop,
        grid=(out_len, B, 1),
        inputs=[frames, window, params],
        template=tmpl,
        output_shape=(B, out_len),
        output_dtype=frames.dtype,
        default_tgx=256,
    )

    outputs = kernel(
        inputs=[frames, window, params],
        template=tmpl,
        output_shapes=[(B, out_len)],
        output_dtypes=[frames.dtype],
        grid=(out_len, B, 1),
        threadgroup=(tgx, 1, 1),
        init_value=0,
    )
    _record_cache_event("backend.ola.metal")
    return outputs[0]

def _run_metal_ola_norm(
    frames: mx.array,
    window: mx.array,
    window_sq: mx.array,
    hop: int,
    out_len: int,
    *,
    require_metal: bool = False,
) -> mx.array:
    """Overlap-add with fused normalization (Torch-like masked divide).

    Note: any NOLA / envelope-min checks are handled outside the fused kernel.
    """
    def _compute_envelope(window_sq_local: mx.array, nframe: int, out_len_i: int) -> mx.array:
        frame = int(window_sq_local.shape[0])
        frame_starts = mx.arange(int(nframe), dtype=mx.int32) * int(hop)
        sample_idx = mx.arange(frame, dtype=mx.int32)
        indices = frame_starts[:, None] + sample_idx[None, :]
        flat_indices = indices.reshape(-1)
        valid_mask = flat_indices < int(out_len_i)
        if int(out_len_i) <= 0:
            return mx.zeros((0,), dtype=window_sq_local.dtype)
        clipped = mx.clip(flat_indices, 0, int(out_len_i) - 1)

        values = mx.tile(window_sq_local, (int(nframe),))
        values = values * valid_mask.astype(values.dtype)

        denom = mx.zeros((int(out_len_i),), dtype=values.dtype)
        denom = denom.at[clipped].add(values)
        return denom

    def _run_fallback(
        frames_local: mx.array,
        window_local: mx.array,
        window_sq_local: mx.array,
        nframe: int,
        out_len_i: int,
    ) -> mx.array:
        out_sum = _run_metal_ola(frames_local, window_local, int(hop), int(out_len_i))
        denom = _compute_envelope(window_sq_local, int(nframe), int(out_len_i))

        if out_sum.dtype != mx.float32:
            out_sum_f32 = out_sum.astype(mx.float32)
            denom_f32 = denom.astype(mx.float32)
            out_f32 = mx.where(
                mx.abs(denom_f32)[None, :] > 1.0e-11,
                out_sum_f32 / denom_f32[None, :],
                mx.zeros_like(out_sum_f32),
            )
            return out_f32.astype(out_sum.dtype)

        return mx.where(
            mx.abs(denom)[None, :] > 1.0e-11,
            out_sum / denom[None, :],
            mx.zeros_like(out_sum),
        )

    if frames.dtype != window.dtype:
        window = window.astype(frames.dtype)
    if frames.dtype != window_sq.dtype:
        window_sq = window_sq.astype(frames.dtype)

    B, nframe, frame = frames.shape
    kernel = _KernelCache.get_ola_norm()
    if kernel is False:
        _record_cache_event("backend.ola_norm.fallback", detail="metal_unavailable")
        if require_metal:
            raise RuntimeError(
                "backend_policy='metal' requires fused Metal OLA norm kernel, "
                "but it is unavailable."
            )
        return _run_fallback(frames, window, window_sq, int(nframe), int(out_len))

    frames = mx.contiguous(frames)
    window = mx.contiguous(window)
    window_sq = mx.contiguous(window_sq)

    params = mx.array([int(nframe), int(out_len)], dtype=mx.int32)
    tmpl = [("T", frames.dtype), ("HOP", hop), ("FRAME", frame)]
    _record_tmpl_event("ola_norm", tmpl)

    tgx = _KernelCache.autotune_threadgroup_x(
        kernel=kernel,
        kernel_name=f"ola_norm_windowed_div_envelope_{frames.dtype}",
        n_fft=frame,
        hop=hop,
        grid=(out_len, B, 1),
        inputs=[frames, window, window_sq, params],
        template=tmpl,
        output_shape=(B, out_len),
        output_dtype=frames.dtype,
        default_tgx=256,
    )

    outputs = kernel(
        inputs=[frames, window, window_sq, params],
        template=tmpl,
        output_shapes=[(B, out_len)],
        output_dtypes=[frames.dtype],
        grid=(out_len, B, 1),
        threadgroup=(tgx, 1, 1),
        init_value=0,
    )
    _record_cache_event("backend.ola_norm.metal")
    return outputs[0]
# ==============================================================================
# Helpers & Transforms
# ==============================================================================

WindowLike = Union[str, mx.array, None]

def _window_cache_signature(
    *,
    provided_window: WindowLike,
    resolved_window: mx.array,
    window_fn: str,
    win_length: int,
    n_fft: int,
    periodic: bool,
) -> tuple:
    """Return a stable key component identifying window shape/content."""
    if isinstance(provided_window, mx.array):
        arr = np.ascontiguousarray(np.asarray(resolved_window, dtype=np.float32))
        digest = hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
        return ("array", int(arr.shape[0]), digest)
    return ("generated", str(window_fn), int(win_length), int(n_fft), bool(periodic))

# ==============================================================================
# Fused iSTFT safety (Torch-like NOLA check, cached)
# ==============================================================================

# Cached device key for OLA safety checks (expensive to compute, rarely changes)
# Bounded safety check cache: stores (ok, min_abs_val, checked_length) tuples.
# Keyed by (device_key, n_fft, hop, win_length, center, window_signature,
# n_frames, length_mode, length_key).
# We keep this bounded to avoid unbounded growth in long-running processes.

_OLA_SAFETY_CACHE_MAX = 256
_OLA_SAFETY_CACHE: "OrderedDict[tuple, tuple[bool, float, int]]" = OrderedDict()

_OLA_SAFETY_LOCK = threading.Lock()
_OLA_SAFETY_LENGTH_STRATEGY = (
    os.environ.get("SPEC_MLX_OLA_SAFETY_LENGTH_STRATEGY", "bucket")
    .strip().lower()
)
if _OLA_SAFETY_LENGTH_STRATEGY not in ("bucket", "exact"):
    _OLA_SAFETY_LENGTH_STRATEGY = "bucket"
try:
    _OLA_SAFETY_LENGTH_BUCKET = max(
        1, int(os.environ.get("SPEC_MLX_OLA_SAFETY_LENGTH_BUCKET", "2048"))
    )
except Exception:
    _OLA_SAFETY_LENGTH_BUCKET = 2048

def _ola_safety_cache_get(key: tuple) -> Optional[tuple[bool, float, int]]:
    with _OLA_SAFETY_LOCK:
        val = _OLA_SAFETY_CACHE.get(key)
        if val is None:
            _record_cache_event("ola_safety_cache.miss", key=key)
            return None
        # Refresh LRU order
        _OLA_SAFETY_CACHE.move_to_end(key)
        _record_cache_event("ola_safety_cache.hit", key=key)
        return val

def _ola_safety_cache_set(key: tuple, value: tuple[bool, float, int]) -> None:
    with _OLA_SAFETY_LOCK:
        _OLA_SAFETY_CACHE[key] = value
        _OLA_SAFETY_CACHE.move_to_end(key)
        _record_cache_event("ola_safety_cache.store", key=key)
        if len(_OLA_SAFETY_CACHE) > _OLA_SAFETY_CACHE_MAX:
            _OLA_SAFETY_CACHE.popitem(last=False)
            _record_cache_event("ola_safety_cache.evict")


def _resolve_ola_safety_length_key_and_check_length(
    length: Optional[int],
    *,
    safety: str,
) -> tuple[str, int, Optional[int], Optional[int]]:
    """Resolve cache key + effective check length for NOLA safety checks.

    Returns ``(length_mode, length_key, check_length, requested_length)``.
    ``length_key`` controls cache-key bucketing; ``check_length`` controls the
    actual envelope span that will be validated.
    """
    if length is None:
        return "none", -1, None, None

    requested = max(0, int(length))
    if safety != "auto" or _OLA_SAFETY_LENGTH_STRATEGY == "exact":
        return "exact", requested, requested, requested

    bucket = int(_OLA_SAFETY_LENGTH_BUCKET)
    bucketed = ((requested + bucket - 1) // bucket) * bucket if requested > 0 else 0
    # Keep the check length exact for Torch-compatible behavior. Bucket only the
    # cache key to reduce key cardinality.
    return "bucket", bucketed, requested, requested

def _ola_envelope_min_check_cached(
    transform: "SpectralTransform",
    *,
    n_frames: int,
    length: Optional[int],
    torch_like: bool,
    safety: str,
    require_metal: bool = False,
) -> None:
    """Torch-style envelope-min check (NOLA) with optimized caching."""
    safety = str(safety)
    if safety not in ("off", "auto", "always"):
        raise ValueError("safety must be one of {'off','auto','always'}")
    if safety == "off":
        return

    # Use cached device key instead of computing every time.
    (length_mode, length_key, check_length, requested_length) = (
        _resolve_ola_safety_length_key_and_check_length(
            length,
            safety=safety,
        )
    )
    
    cache_key = (
        spec_mlx_device_key(),      # ← Cached at module level
        transform.n_fft,
        transform.hop_length,
        transform.win_length,
        transform.center,
        transform._window_cache_sig,
        int(n_frames),
        length_mode,
        length_key,
    )

    # Fast path: check cache for 'auto' mode
    if safety == "auto":
        cached = _ola_safety_cache_get(cache_key)
        if cached is not None:
            ok, min_abs_val, cached_checked_length = cached

            # Monotonic reuse rules for exact Torch-compatible semantics:
            # - If cached check passed at Lc, any shorter Lr <= Lc also passes.
            # - If cached check failed at Lc, any longer Lr >= Lc also fails.
            reusable = True
            if requested_length is not None and cached_checked_length >= 0:
                if ok:
                    reusable = int(requested_length) <= int(cached_checked_length)
                else:
                    reusable = int(requested_length) >= int(cached_checked_length)

            if reusable:
                if (not ok) and torch_like:
                    raise RuntimeError(
                        f"istft: window overlap-add envelope is too small "
                        f"(cached: min={min_abs_val:.3e}); "
                        "this matches Torch's NOLA safety intent."
                    )
                return

    # Compute envelope
    denom, _ = transform._get_ola_envelope(int(n_frames), require_metal=require_metal)

    if transform.center:
        pad = transform.n_fft // 2
        if check_length is None:
            denom = denom[pad:-pad]
        else:
            end = min(int(denom.shape[0]), pad + int(check_length))
            denom = denom[pad:end]
    elif check_length is not None:
        take = min(int(check_length), int(denom.shape[0]))
        denom = denom[:take]

    # Check minimum; guard empty windows to match torch-side behavior.
    if int(denom.shape[0]) == 0:
        min_abs_val = 0.0
    else:
        min_abs = mx.min(mx.abs(denom))
        mx.eval(min_abs)
        min_abs_val = float(min_abs.item())
    ok = (min_abs_val >= 1e-11)
    
    # Cache result
    if safety == "auto":
        checked_len_value = int(check_length) if check_length is not None else -1
        _ola_safety_cache_set(cache_key, (ok, min_abs_val, checked_len_value))
    
    if (not ok) and torch_like:
        raise RuntimeError(
            "istft: window overlap-add envelope is too small "
            f"(min={min_abs_val:.3e}); this matches Torch's NOLA safety intent. "
            "Try increasing win_length, using a COLA-compliant window/hop, "
            "or disable safety checks (safety='off') if you know what you're doing."
        )

@mx.compile
def _torch_like_reflect_pad_1d_compiled(x: mx.array, pad: int) -> mx.array:
    """Compiled reflect-pad using pure MLX ops."""
    if pad <= 0:
        return x
    if x.shape[-1] <= pad:
        raise ValueError(
            f"torch-like reflect padding requires input_length > pad "
            f"(got length={x.shape[-1]}, pad={pad})"
        )
    left = x[..., 1 : pad + 1][..., ::-1]
    right = x[..., -pad - 1 : -1][..., ::-1]
    return mx.concatenate([left, x, right], axis=-1)


def _torch_like_reflect_pad_1d(x: mx.array, pad: int) -> mx.array:
    return _torch_like_reflect_pad_1d_compiled(x, pad)


def make_window(
    window: WindowLike,
    window_fn: str,
    win_length: int,
    n_fft: int,
    periodic: bool,
) -> mx.array:
    """Create or validate a 1D window with strict shape and type checks."""
    if isinstance(window, mx.array):
        # Optimization: Avoid unnecessary copy if already float32
        w = window if window.dtype == mx.float32 else window.astype(mx.float32)
        if w.ndim != 1:
            raise ValueError(f"window must be 1D, got shape {w.shape}")
        if w.shape[0] == n_fft:
            return w
        if w.shape[0] != win_length:
            raise ValueError(
                f"window length must be win_length ({win_length}) "
                f"or n_fft ({n_fft}), got {w.shape[0]}"
            )
        
        left = (n_fft - win_length) // 2
        right = (n_fft - win_length + 1) // 2
        return mx.pad(w, [(left, right)], mode="constant")

    if win_length <= 0:
        raise ValueError("win_length must be positive")

    denom = win_length if periodic else (win_length - 1 if win_length > 1 else 1)
    idx = mx.arange(win_length, dtype=mx.float32)

    if window_fn == "hann":
        w = 0.5 - 0.5 * mx.cos(2.0 * math.pi * idx / denom)
    elif window_fn == "hamming":
        w = 0.54 - 0.46 * mx.cos(2.0 * math.pi * idx / denom)
    elif window_fn in ("rect", "boxcar", "ones"):
        w = mx.ones((win_length,), dtype=mx.float32)
    else:
        raise ValueError(f"Unknown window_fn: {window_fn}")

    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = (n_fft - win_length + 1) // 2
        w = mx.pad(w, [(left, right)], mode="constant")
    elif win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

    return w

def resolve_fft_params(
    n_fft: int,
    hop_length: Optional[int],
    win_length: Optional[int],
    pad: int,
) -> tuple[int, int, int]:
    """Resolve effective FFT parameters.

    Returns (effective_n_fft, hop_length, win_length).
    """
    n_fft = int(n_fft)
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")

    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be >= 0")

    if hop_length is None:
        hop_length = n_fft // 4
    hop_length = int(hop_length)
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    # Match PyTorch semantics: `pad` is an *additive* number of samples padded on
    # both sides within each analysis frame. Effective FFT size grows by 2*pad.
    effective_n_fft = n_fft + 2 * pad

    # Match PyTorch default: if win_length is not provided, use n_fft.
    if win_length is None:
        win_length = n_fft
    win_length = int(win_length)
    
    if win_length <= 0:
        raise ValueError("win_length must be positive")
    if win_length > effective_n_fft:
        raise ValueError(
            f"win_length ({win_length}) must be <= "
            f"effective_n_fft ({effective_n_fft})"
        )

    return effective_n_fft, hop_length, win_length


_ISTFT_BACKEND_POLICIES = ("auto", "mlx_fft", "metal", "torch_fallback")
_STFT_OUTPUT_LAYOUTS = ("bfn", "bnf")


def _resolve_backend_policy(
    backend_policy: Optional[str],
    *,
    default_policy: str = "auto",
) -> str:
    policy = default_policy if backend_policy is None else str(backend_policy)
    if policy not in _ISTFT_BACKEND_POLICIES:
        raise ValueError(
            "backend_policy must be one of "
            f"{_ISTFT_BACKEND_POLICIES}"
        )
    return policy


def _resolve_stft_output_layout(
    output_layout: Optional[str],
    *,
    default_layout: str = "bfn",
) -> str:
    layout = default_layout if output_layout is None else str(output_layout)
    if layout not in _STFT_OUTPUT_LAYOUTS:
        raise ValueError(
            "output_layout must be one of "
            f"{_STFT_OUTPUT_LAYOUTS}"
        )
    return layout


_DEFAULT_ISTFT_BACKEND_POLICY = "auto"

# ==============================================================================
# SpectralTransform
# ==============================================================================

@dataclass(frozen=True)
class _TransformKey:
    n_fft: int
    hop_length: int
    win_length: int
    window_fn: str
    periodic: bool
    center: bool
    normalized: bool
    istft_backend_policy: str

_MLX_OLA_FUSE_NORM = (
    os.environ.get("MLX_OLA_FUSE_NORM", "1").lower()
    not in ("0", "false", "no", "off")
)

class SpectralTransform:
    """
    High-performance STFT/iSTFT engine for MLX.
    Uses fused kernels and cached configurations for maximum throughput.
    """
    __slots__ = (
        'n_fft', 'hop_length', 'win_length', 'window', 'window_fn',
        '_window_sq', 'center', 'normalized', 'periodic',
        'istft_backend_policy',
        '_window_cache_sig',
        '_norm_factor', '_inv_norm_factor',
        '_cache_key', 'ola_denom', 'ola_denom_inv',
        '_compiled_stft_fns', '_compiled_istft_fns',
        '_torch_window_cache', '_window_runtime_cache',
    )

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window_fn: str = "hann",
        *,
        window: WindowLike = None,
        periodic: bool = True,
        center: bool = True,
        normalized: bool = False,
        istft_backend_policy: Optional[str] = None,
    ):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else self.n_fft
        
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) must be <= n_fft ({self.n_fft})")

        self.center = bool(center)
        self.normalized = bool(normalized)
        self.periodic = bool(periodic)
        self.window_fn = str(window_fn)
        self.istft_backend_policy = _resolve_backend_policy(
            istft_backend_policy,
            default_policy=_DEFAULT_ISTFT_BACKEND_POLICY,
        )

        self.window = make_window(
            window=window,
            window_fn=self.window_fn,
            win_length=self.win_length,
            n_fft=self.n_fft,
            periodic=self.periodic,
        )
        self.window = mx.contiguous(self.window)
        self._window_cache_sig = _window_cache_signature(
            provided_window=window,
            resolved_window=self.window,
            window_fn=self.window_fn,
            win_length=self.win_length,
            n_fft=self.n_fft,
            periodic=self.periodic,
        )

        # Pre-computed normalization factors (avoid per-call sqrt)
        self._inv_norm_factor = 1.0 / math.sqrt(self.n_fft) if self.normalized else 1.0
        self._norm_factor = math.sqrt(self.n_fft) if self.normalized else 1.0

        # Pre-computed squared window for fused NOLA
        self._window_sq = mx.contiguous((self.window ** 2).astype(mx.float32))
        self._window_runtime_cache: Dict[str, Tuple[mx.array, mx.array]] = {
            str(self.window.dtype): (self.window, self._window_sq)
        }

        # Legacy caches
        self.ola_denom = None
        self.ola_denom_inv = None
        self._cache_key = None
        self._compiled_stft_fns = {}
        self._compiled_istft_fns = {}
        self._torch_window_cache: Dict[str, Any] = {}

    def _window_pair_for_dtype(self, dtype: Any) -> Tuple[mx.array, mx.array]:
        key = str(dtype)
        cached = self._window_runtime_cache.get(key)
        if cached is not None:
            return cached

        window = self.window if self.window.dtype == dtype else self.window.astype(dtype)
        window_sq = (
            self._window_sq if self._window_sq.dtype == dtype
            else self._window_sq.astype(dtype)
        )
        window = mx.contiguous(window)
        window_sq = mx.contiguous(window_sq)
        pair = (window, window_sq)
        self._window_runtime_cache[key] = pair
        return pair

    def warmup(self, batch: int = 1, length: int = 4096) -> None:
        """Force compilation of kernels and execution paths."""
        x = mx.zeros((batch, length), dtype=mx.float32)
        Z = self.stft(x, output_layout="bnf")
        y = self.istft(Z, length=length, input_layout="bnf")
        mx.eval(Z, y)

    def prewarm_kernels(self, batch: int = 1, length: Optional[int] = None) -> None:
        """Precompile stft + fused and legacy istft kernels for this transform."""
        if length is None:
            length = max(int(self.n_fft) * 2, 4096)
        x = mx.zeros((int(batch), int(length)), dtype=mx.float32)
        Z = self.stft(x, output_layout="bnf")
        y_fused = self.istft(
            Z,
            length=int(length),
            torch_like=False,
            allow_fused=True,
            safety="off",
            input_layout="bnf",
        )
        y_legacy = self.istft(
            Z,
            length=int(length),
            torch_like=False,
            allow_fused=False,
            safety="off",
            input_layout="bnf",
        )
        mx.eval(Z, y_fused, y_legacy)

    def prewarm_compiled(
        self,
        *,
        batch: int = 1,
        length: Optional[int] = None,
        validate: bool = False,
        torch_like: bool = False,
        allow_fused: bool = True,
        safety: str = "off",
        backend_policy: Optional[str] = None,
    ) -> None:
        """Precompile cached compiled STFT/iSTFT callables for steady-shape loops."""
        if length is None:
            length = max(int(self.n_fft) * 2, 4096)
        x = mx.zeros((int(batch), int(length)), dtype=mx.float32)
        z = self.stft_compiled(x, output_layout="bnf")
        y = self.istft_compiled(
            z,
            length=int(length),
            validate=validate,
            torch_like=torch_like,
            allow_fused=allow_fused,
            safety=safety,
            long_mode_strategy="native",
            backend_policy=backend_policy,
            input_layout="bnf",
        )
        mx.eval(z, y)

    def get_compiled_stft(self, *, output_layout: str = "bfn"):
        """Return a cached compiled STFT callable for steady-shape workloads."""
        resolved_layout = _resolve_stft_output_layout(output_layout)
        cached = self._compiled_stft_fns.get(resolved_layout)
        if cached is not None:
            _record_cache_event("compiled_stft_cache.hit", key=resolved_layout)
            return cached
        _record_cache_event("compiled_stft_cache.miss", key=resolved_layout)

        @mx.compile
        def _compiled(x: mx.array) -> mx.array:
            return self.stft(x, output_layout=resolved_layout)

        self._compiled_stft_fns[resolved_layout] = _compiled
        return _compiled

    def get_compiled_istft(
        self,
        *,
        length: Optional[int] = None,
        validate: bool = False,
        torch_like: bool = False,
        allow_fused: bool = True,
        safety: str = "auto",
        long_mode_strategy: str = "native",
        backend_policy: Optional[str] = None,
        input_layout: str = "bfn",
    ):
        """Return a cached compiled iSTFT callable for fixed runtime options.

        For best throughput, reuse the returned callable with stable input shapes.
        """
        if long_mode_strategy != "native":
            raise ValueError(
                "Compiled iSTFT supports long_mode_strategy='native' only. "
                "Use eager istft for numpy_fallback/torch_fallback behavior."
            )
        resolved_backend = _resolve_backend_policy(
            backend_policy,
            default_policy=self.istft_backend_policy,
        )
        if resolved_backend == "torch_fallback":
            raise ValueError(
                "Compiled iSTFT does not support backend_policy='torch_fallback'. "
                "Use eager istft for torch fallback behavior."
            )
        resolved_input_layout = _resolve_stft_output_layout(input_layout)

        key = (
            int(length) if length is not None else -1,
            bool(validate),
            bool(torch_like),
            bool(allow_fused),
            str(safety),
            str(long_mode_strategy),
            str(resolved_backend),
            str(resolved_input_layout),
        )
        cached = self._compiled_istft_fns.get(key)
        if cached is not None:
            _record_cache_event("compiled_istft_cache.hit", key=key)
            return cached
        _record_cache_event("compiled_istft_cache.miss", key=key)

        @mx.compile
        def _compiled(z: mx.array) -> mx.array:
            return self.istft(
                z,
                length=length,
                validate=validate,
                torch_like=torch_like,
                allow_fused=allow_fused,
                safety=safety,
                long_mode_strategy=long_mode_strategy,
                backend_policy=resolved_backend,
                input_layout=resolved_input_layout,
            )

        self._compiled_istft_fns[key] = _compiled
        return _compiled

    def stft_compiled(self, x: mx.array, *, output_layout: str = "bfn") -> mx.array:
        """Execute compiled STFT using cached compiled graph."""
        return self.get_compiled_stft(output_layout=output_layout)(x)

    def istft_compiled(
        self,
        z: mx.array,
        *,
        length: Optional[int] = None,
        validate: bool = False,
        torch_like: bool = False,
        allow_fused: bool = True,
        safety: str = "auto",
        long_mode_strategy: str = "native",
        backend_policy: Optional[str] = None,
        input_layout: str = "bfn",
    ) -> mx.array:
        """Execute compiled iSTFT using cached compiled graph for fixed options."""
        fn = self.get_compiled_istft(
            length=length,
            validate=validate,
            torch_like=torch_like,
            allow_fused=allow_fused,
            safety=safety,
            long_mode_strategy=long_mode_strategy,
            backend_policy=backend_policy,
            input_layout=input_layout,
        )
        return fn(z)

    def compiled_pair(
        self,
        *,
        length: int,
        layout: str = "bnf",
        warmup_batch: Optional[int] = None,
    ) -> tuple:
        """Return ``(stft_fn, istft_fn)`` compiled for a fixed configuration.

        10–20% faster than the eager ``stft()``/``istft()`` methods in
        steady-state loops by eliminating per-call Python dispatch overhead.

        Args:
            length: Signal length in samples.  Fixed for this pair.
            layout: ``"bnf"`` (default, fastest) or ``"bfn"``.
            warmup_batch: If provided, run one warmup pass with this batch
                size so that kernel compilation and safety checks are paid
                upfront rather than on the first real call.

        Returns:
            ``(stft_fn, istft_fn)`` — call as ``z = stft_fn(x)``,
            ``y = istft_fn(z)``.

        Example::

            t = SpectralTransform(n_fft=1024, hop_length=256)
            stft, istft = t.compiled_pair(length=44100, warmup_batch=2)

            for chunk in stream:
                z = stft(chunk)
                z = process(z)
                y = istft(z)
                mx.eval(y)
        """
        resolved_layout = _resolve_stft_output_layout(layout)

        # Prime the NOLA safety cache with one eager call so the compiled
        # path never hits an mx.eval barrier inside the safety check.
        warmup_len = int(length)
        warmup_b = int(warmup_batch) if warmup_batch is not None else 1
        x_warm = mx.zeros((warmup_b, warmup_len), dtype=mx.float32)
        z_warm = self.stft(x_warm, output_layout=resolved_layout)
        y_warm = self.istft(
            z_warm, length=warmup_len,
            input_layout=resolved_layout, safety="auto",
        )
        mx.eval(z_warm, y_warm)

        stft_fn = self.get_compiled_stft(output_layout=resolved_layout)
        istft_fn = self.get_compiled_istft(
            length=warmup_len,
            input_layout=resolved_layout,
            safety="auto",
        )

        # If warmup_batch was given, also prewarm the compiled graphs.
        if warmup_batch is not None:
            z_c = stft_fn(x_warm)
            y_c = istft_fn(z_c)
            mx.eval(z_c, y_c)

        return stft_fn, istft_fn

    def _get_ola_envelope(
        self, n_frames: int, *, require_metal: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Legacy path envelope calculation (used only if fused norm is disabled)."""
        out_len = self.hop_length * (n_frames - 1) + self.n_fft
        key = (self.n_fft, self.hop_length, out_len, n_frames)
        
        # Re-add memoization check
        if (
            getattr(self, "_cache_key", None) == key
            and self.ola_denom is not None
            and self.ola_denom_inv is not None
        ):
            if require_metal and (_KernelCache.get_envelope() is False):
                raise RuntimeError(
                    "backend_policy='metal' requires Metal envelope kernel, but it is unavailable."
                )
            _record_cache_event("ola_envelope_cache.hit", key=key)
            return self.ola_denom, self.ola_denom_inv
        _record_cache_event("ola_envelope_cache.miss", key=key)

        _, window_sq = self._window_pair_for_dtype(mx.float32)

        kernel = _KernelCache.get_envelope()
        if kernel is False:
            _record_cache_event("backend.envelope.fallback", detail="metal_unavailable")
            if require_metal:
                raise RuntimeError(
                    "backend_policy='metal' requires Metal envelope kernel, but it is unavailable."
                )
            frame = int(window_sq.shape[0])
            frame_starts = mx.arange(int(n_frames), dtype=mx.int32) * int(self.hop_length)
            sample_idx = mx.arange(frame, dtype=mx.int32)
            indices = frame_starts[:, None] + sample_idx[None, :]
            flat_indices = indices.reshape(-1)
            valid_mask = flat_indices < int(out_len)
            clipped = mx.clip(flat_indices, 0, int(out_len) - 1)

            values = mx.tile(window_sq, (int(n_frames),))
            values = values * valid_mask.astype(values.dtype)
            denom = mx.zeros((int(out_len),), dtype=window_sq.dtype)
            denom = denom.at[clipped].add(values)

            if denom.dtype != mx.float32:
                denom = denom.astype(mx.float32)
            denom_inv = mx.where(mx.abs(denom) > 1.0e-11, 1.0 / denom, mx.zeros_like(denom))
            self.ola_denom = denom
            self.ola_denom_inv = denom_inv
            self._cache_key = key
            return denom, denom_inv
            
        params = mx.array([int(n_frames), int(out_len)], dtype=mx.int32)
        tmpl = [
            ("T", window_sq.dtype), ("HOP", self.hop_length),
            ("FRAME", self.n_fft),
        ]
        _record_tmpl_event("envelope", tmpl)

        tgx = _KernelCache.autotune_threadgroup_x(
            kernel=kernel,
            kernel_name=f"ola_envelope_optimized_{window_sq.dtype}",
            n_fft=self.n_fft,
            hop=self.hop_length,
            grid=(out_len, 1, 1),
            inputs=[window_sq, params],
            template=tmpl,
            output_shape=(out_len,),
            output_dtype=window_sq.dtype,
            default_tgx=256,
        )

        outputs = kernel(
            inputs=[window_sq, params],
            template=tmpl,
            grid=(out_len, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(out_len,)],
            output_dtypes=[window_sq.dtype],
            init_value=0,
        )
        _record_cache_event("backend.envelope.metal")

        denom = outputs[0]
        if denom.dtype != mx.float32:
            denom = denom.astype(mx.float32)
        # Match fused kernel semantics: masked divide when envelope is tiny.
        denom_inv = mx.where(mx.abs(denom) > 1.0e-11, 1.0 / denom, mx.zeros_like(denom))
        self.ola_denom = denom
        self.ola_denom_inv = denom_inv
        self._cache_key = key
        return denom, denom_inv

    def stft(self, x: mx.array, *, output_layout: str = "bfn") -> mx.array:
        """
        Forward STFT.
        Input: [T] or [B, T]
        Output: [B, F, N] (Complex, `output_layout="bfn"`), or
                [B, N, F] (Complex, `output_layout="bnf"`)
        """
        resolved_layout = _resolve_stft_output_layout(output_layout)
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"stft expects 1D or 2D input, got {x.shape}")

        B, sig_len = x.shape

        # ── Fast path: fused frame extraction ────────────────────────
        # When center=True and the signal is long enough, a Metal kernel
        # combines reflect-pad + strided windowing, eliminating the
        # padded-signal intermediate buffer.
        #
        # Two kernel variants:
        #   Tiled  — uses threadgroup shared memory for ~NFFT/HOP data reuse.
        #            ~1.3× faster when bandwidth-bound (large B × sig_len).
        #   Simple — one thread per output element, no shared memory.
        #            Used for small workloads where dispatch latency dominates.
        if self.center and sig_len >= self.n_fft:
            pad = self.n_fft // 2
            padded_len = sig_len + 2 * pad
            n_frames = (padded_len - self.n_fft) // self.hop_length + 1
            out_bytes = B * n_frames * self.n_fft * 4

            # params buffer: dynamic lengths as runtime inputs (not template constants)
            # to avoid kernel recompilation for every unique audio length.
            fe_params = mx.array([int(sig_len), int(n_frames)], dtype=mx.int32)

            # Try tiled kernel for large workloads
            tiled_ok = False
            if out_bytes >= _TILED_FRAME_EXTRACT_BYTE_THRESHOLD:
                tiled_kernel = _FrameExtractCache.get_tiled()
                tp = _FrameExtractCache.tile_params(self.n_fft, self.hop_length)
                if tiled_kernel and tp is not None:
                    tile_frames, tg_x, tg_y, chunk_len = tp
                    x = mx.contiguous(x)
                    n_tile_groups = math.ceil(n_frames / tile_frames)
                    tmpl = [
                        ("T", x.dtype), ("NFFT", self.n_fft),
                        ("HOP", self.hop_length), ("PAD", pad),
                        ("TILE_FRAMES", tile_frames), ("TG_X", tg_x),
                        ("TG_Y", tg_y), ("CHUNK_LEN", chunk_len),
                    ]
                    outputs = tiled_kernel(
                        inputs=[x, self.window, fe_params],
                        template=tmpl,
                        output_shapes=[(B, n_frames, self.n_fft)],
                        output_dtypes=[x.dtype],
                        grid=(n_tile_groups * tg_x, tg_y, B),
                        threadgroup=(tg_x, tg_y, 1),
                    )
                    tiled_ok = True

            # Fall back to simple kernel
            if not tiled_ok:
                kernel = _FrameExtractCache.get_simple()
                if kernel and kernel is not False:
                    x = mx.contiguous(x)
                    tmpl = [
                        ("T", x.dtype), ("NFFT", self.n_fft),
                        ("HOP", self.hop_length), ("PAD", pad),
                    ]
                    outputs = kernel(
                        inputs=[x, self.window, fe_params],
                        template=tmpl,
                        output_shapes=[(B, n_frames, self.n_fft)],
                        output_dtypes=[x.dtype],
                        grid=(self.n_fft, n_frames, B),
                        threadgroup=(min(256, self.n_fft), 1, 1),
                    )
                else:
                    outputs = None

            if outputs is not None:
                spec = mx.fft.rfft(outputs[0], axis=-1)
                if self.normalized:
                    spec = spec * self._inv_norm_factor
                if resolved_layout == "bnf":
                    return spec
                return spec.transpose(0, 2, 1)

        # ── Fallback path ────────────────────────────────────────────
        # Ensure contiguous memory before striding
        x = mx.contiguous(x)

        if self.center:
            pad = self.n_fft // 2
            x = _torch_like_reflect_pad_1d_compiled(x, pad)

        B, T_pad = x.shape
        if T_pad < self.n_fft:
            need = self.n_fft - T_pad
            x = mx.pad(x, [(0, 0), (0, need)], mode="constant")
            B, T_pad = x.shape

        n_frames = (T_pad - self.n_fft) // self.hop_length + 1

        # Strided view (zero-copy)
        frames = mx.as_strided(
            x,
            shape=(B, n_frames, self.n_fft),
            strides=(T_pad, self.hop_length, 1),
        )

        # Apply window
        frames = frames * self.window

        # FFT (Real -> Complex)
        spec = mx.fft.rfft(frames, axis=-1)

        if self.normalized:
            spec = spec * self._inv_norm_factor

        # MLX rFFT emits [B, N, F]. Keep native layout for bnf, transpose for bfn.
        if resolved_layout == "bnf":
            return spec
        return spec.transpose(0, 2, 1)

    def istft(
        self,
        z: mx.array,
        length: Optional[int] = None,
        validate: bool = False,
        *,
        torch_like: bool = False,
        allow_fused: bool = True,
        safety: str = "auto",
        long_mode_strategy: str = "native",
        backend_policy: Optional[str] = None,
        input_layout: str = "bfn",
    ) -> mx.array:
        """
        Inverse STFT.
        Input: [B, F, N] (Complex, ``input_layout="bfn"``) or
               [B, N, F] (Complex, ``input_layout="bnf"``)
        Output: [B, T] (Real)

        long_mode_strategy:
          - "native": always use MLX FFT for iFFT (fastest path).
          - "numpy_fallback": for center=True and extended-length requests
            (length > trimmed_length), use NumPy iFFT to reduce cross-framework
            numerical drift in long-mode tails.
          - "torch_fallback": for targeted center=True extended-length requests,
            run torch.istft for strict parity while keeping native MLX path for
            all other cases.
        backend_policy:
          - "auto": use legacy routing rules from long_mode_strategy.
          - "mlx_fft": force MLX iFFT path (no NumPy/Torch fallback route).
          - "metal": same as mlx_fft, but require Metal OLA/envelope kernels.
          - "torch_fallback": force torch.istft path for this call.
        """
        if long_mode_strategy not in {"native", "numpy_fallback", "torch_fallback"}:
            raise ValueError(
                "long_mode_strategy must be one of "
                "{'native', 'numpy_fallback', 'torch_fallback'}"
            )
        resolved_backend = _resolve_backend_policy(
            backend_policy,
            default_policy=self.istft_backend_policy,
        )
        resolved_input_layout = _resolve_stft_output_layout(input_layout)
        if resolved_backend != "auto" and long_mode_strategy != "native":
            raise ValueError(
                "When backend_policy is not 'auto', long_mode_strategy must be 'native'."
            )
        _record_cache_event(f"backend_policy.call.{resolved_backend}")

        orig_2d_input = (z.ndim == 2)
        if orig_2d_input:
            z = z[None, ...]
        elif z.ndim != 3:
            raise ValueError(
                "istft expects a 2D/3D complex spectrogram in [F, N]/[B, F, N] "
                f"(input_layout='bfn') or [N, F]/[B, N, F] (input_layout='bnf'); got {z.shape}"
            )

        z_bfn = None
        z_bnf = None
        onesided_bins = int(self.n_fft // 2 + 1)
        twosided_bins = int(self.n_fft)
        if resolved_input_layout == "bfn":
            B, freq_bins, n_frames = z.shape
            if int(freq_bins) not in (onesided_bins, twosided_bins):
                raise ValueError(
                    "istft input_layout='bfn' expects frequency bins in axis=1 with "
                    f"size {onesided_bins} (onesided) or {twosided_bins} (dualsided); "
                    f"got axis=1 size {int(freq_bins)}. "
                    "If your input is [B, N, F], pass input_layout='bnf'."
                )
            z_bfn = z
        else:
            B, n_frames, freq_bins = z.shape
            if int(freq_bins) not in (onesided_bins, twosided_bins):
                raise ValueError(
                    "istft input_layout='bnf' expects frequency bins in axis=2 with "
                    f"size {onesided_bins} (onesided) or {twosided_bins} (dualsided); "
                    f"got axis=2 size {int(freq_bins)}. "
                    "If your input is [B, F, N], pass input_layout='bfn'."
                )
            z_bnf = z

        trimmed_length = self.hop_length * (int(n_frames) - 1) if self.center else (
            self.hop_length * (int(n_frames) - 1) + self.n_fft
        )
        is_long_request = (length is not None) and (int(length) > int(trimmed_length))

        if resolved_backend == "torch_fallback":
            if z_bfn is None:
                z_bfn = z.transpose(0, 2, 1)
            out_torch = self._istft_torch_fallback(z_bfn, length=length)
            if out_torch is None:
                raise RuntimeError(
                    "backend_policy='torch_fallback' requested, but torch fallback is unavailable."
                )
            _record_cache_event("backend_policy.route.torch_fallback_forced")
            if orig_2d_input:
                out_torch = out_torch[0]
            return out_torch

        use_torch_fallback = (
            (resolved_backend == "auto")
            and
            bool(self.center)
            and bool(is_long_request)
            and (long_mode_strategy == "torch_fallback")
            and (int(self.hop_length) * 2 == int(self.n_fft))
        )
        if use_torch_fallback:
            if z_bfn is None:
                z_bfn = z.transpose(0, 2, 1)
            out_torch = self._istft_torch_fallback(z_bfn, length=length)
            if out_torch is not None:
                _record_cache_event("backend_policy.route.torch_fallback_auto")
                if orig_2d_input:
                    out_torch = out_torch[0]
                return out_torch

        # Keep native [B, N, F] layout for irfft whenever possible.
        if z_bnf is None:
            z_bnf = z.transpose(0, 2, 1)

        use_numpy_fallback = (
            (resolved_backend == "auto")
            and bool(self.center)
            and bool(is_long_request)
            and (long_mode_strategy == "numpy_fallback")
        )
        if use_numpy_fallback:
            _record_cache_event("backend_policy.route.numpy_fallback")
            try:
                z_np = np.asarray(z_bnf)
                # NumPy's IRFFT tracks torch.fft.irfft closely for these inputs and
                # avoids amplified tail drift in center+long reconstruction.
                time_frames_np = np.fft.irfft(
                    z_np, n=self.n_fft, axis=-1,
                ).astype(np.float32, copy=False)
                time_frames = mx.array(time_frames_np, dtype=mx.float32)
            except Exception as err:
                warnings.warn(
                    "numpy irfft fallback failed; continuing with native MLX irfft path "
                    f"(fallback error: {err})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                time_frames = mx.fft.irfft(z_bnf, n=self.n_fft, axis=-1)
        else:
            # iFFT (Complex -> Real)
            _record_cache_event("backend_policy.route.mlx_irfft")
            time_frames = mx.fft.irfft(z_bnf, n=self.n_fft, axis=-1)
        
        if self.normalized:
            time_frames = time_frames * self._norm_factor
        window_runtime, window_sq_runtime = self._window_pair_for_dtype(time_frames.dtype)

        out_len = self.hop_length * (n_frames - 1) + self.n_fft

        fuse_norm = _MLX_OLA_FUSE_NORM and allow_fused
        require_metal = (resolved_backend == "metal")

        # Optional Torch-style NOLA safety check (cached) even when fused.
        if fuse_norm:
            _ola_envelope_min_check_cached(
                self,
                n_frames=int(n_frames),
                length=length,
                torch_like=bool(torch_like),
                safety=safety,
                require_metal=require_metal,
            )

        if fuse_norm and not validate:
            out = _run_metal_ola_norm(
                time_frames,
                window_runtime,
                window_sq_runtime,
                self.hop_length,
                out_len,
                require_metal=require_metal,
            )
        else:
            out_sum = _run_metal_ola(
                time_frames,
                window_runtime,
                self.hop_length,
                out_len,
                require_metal=require_metal,
            )
            denom, denom_inv = self._get_ola_envelope(n_frames, require_metal=require_metal)

            # Torch-like math: masked divide by envelope (no epsilon clamp).
            if out_sum.dtype != mx.float32:
                out_sum = out_sum.astype(mx.float32)
            out_legacy = out_sum * denom_inv

            if validate:
                out_fused = _run_metal_ola_norm(
                    time_frames,
                    window_runtime,
                    window_sq_runtime,
                    self.hop_length,
                    out_len,
                    require_metal=require_metal,
                )
                mx.eval(out_fused, out_legacy)
                if not mx.allclose(out_fused, out_legacy, atol=1e-5):
                    diff = mx.max(mx.abs(out_fused - out_legacy))
                    mx.eval(diff)
                    raise RuntimeError(f"istft mismatch: max diff {float(diff.item()):.3e}")
                out = out_fused
            else:
                out = out_legacy
        # Trimming / Torch-like length handling
        if self.center:
            pad = self.n_fft // 2
            if length is None:
                out = out[:, pad:-pad] if int(out.shape[1]) > (2 * pad) else out[:, :0]
            else:
                start = pad
                target = int(length)
                end = min(int(out.shape[1]), start + target)
                out = out[:, start:end]
                if int(out.shape[1]) < target:
                    out = mx.concatenate(
                        [
                            out,
                            mx.zeros(
                                (int(out.shape[0]), target - int(out.shape[1])),
                                dtype=out.dtype,
                            ),
                        ],
                        axis=1,
                    )
        elif length is not None:
            target = int(length)
            if int(out.shape[1]) >= target:
                out = out[:, :target]
            else:
                out = mx.concatenate(
                    [
                        out,
                        mx.zeros(
                            (int(out.shape[0]), target - int(out.shape[1])),
                            dtype=out.dtype,
                        ),
                    ],
                    axis=1,
                )

        if orig_2d_input:
            out = out[0]

        return out

    def _istft_torch_fallback(
        self, z_bfn: mx.array, *, length: Optional[int],
    ) -> Optional[mx.array]:
        """Run torch.istft for strict parity in selected long-mode cases."""
        try:
            import torch
        except Exception as err:
            _record_cache_event("backend.torch_fallback.import_error")
            warnings.warn(
                "torch fallback unavailable; continuing with native MLX iFFT path "
                f"(import error: {err})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        try:
            spec_np = np.asarray(z_bfn)
            if spec_np.ndim != 3:
                raise ValueError(f"expected [B, F, N] spec for torch fallback, got {spec_np.shape}")

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            spec_t = torch.from_numpy(spec_np)
            if spec_t.dtype != torch.complex64:
                spec_t = spec_t.to(dtype=torch.complex64)
            spec_t = spec_t.to(device=device)

            window_t = self._torch_window_cache.get(device.type)
            if window_t is None:
                window_np = np.asarray(self.window, dtype=np.float32)
                window_t = torch.from_numpy(window_np).to(device=device, dtype=torch.float32)
                self._torch_window_cache[device.type] = window_t

            freq_bins = int(spec_t.shape[1])
            onesided = freq_bins != int(self.n_fft)
            with warnings.catch_warnings():
                # Torch MPS may emit a noisy internal resize warning here that does not
                # affect correctness for this fallback path.
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"An output with one or more elements was resized"
                        r" since it had shape \[\].*"
                    ),
                    category=UserWarning,
                )
                out_t = torch.istft(
                    spec_t,
                    n_fft=int(self.n_fft),
                    hop_length=int(self.hop_length),
                    win_length=int(self.win_length),
                    window=window_t,
                    center=bool(self.center),
                    normalized=bool(self.normalized),
                    onesided=bool(onesided),
                    return_complex=False,
                    length=(int(length) if length is not None else None),
                )
            _record_cache_event("backend.torch_fallback.success")
            return mx.array(out_t.detach().to("cpu").numpy(), dtype=mx.float32)
        except Exception as err:
            _record_cache_event("backend.torch_fallback.error")
            warnings.warn(
                "torch fallback failed; continuing with native MLX iFFT path "
                f"(fallback error: {err})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


# ==============================================================================
# Global Cache & Factory
# ==============================================================================

@lru_cache(maxsize=64)
def _get_transform_cached(key: _TransformKey) -> SpectralTransform:
    return SpectralTransform(
        n_fft=key.n_fft,
        hop_length=key.hop_length,
        win_length=key.win_length,
        window_fn=key.window_fn,
        periodic=key.periodic,
        center=key.center,
        normalized=key.normalized,
        istft_backend_policy=key.istft_backend_policy,
    )

def get_transform_mlx(
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn: str,
    periodic: bool,
    center: bool,
    normalized: bool,
    window: WindowLike,
    istft_backend_policy: Optional[str] = None,
) -> SpectralTransform:
    """Return a cached or bespoke SpectralTransform for the given config."""
    resolved_backend = _resolve_backend_policy(
        istft_backend_policy,
        default_policy=_DEFAULT_ISTFT_BACKEND_POLICY,
    )
    if isinstance(window, mx.array):
        return SpectralTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window_fn,
            window=window,
            periodic=periodic,
            center=center,
            normalized=normalized,
            istft_backend_policy=resolved_backend,
        )

    key = _TransformKey(
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window_fn=str(window_fn),
        periodic=bool(periodic),
        center=bool(center),
        normalized=bool(normalized),
        istft_backend_policy=str(resolved_backend),
    )
    return _get_transform_cached(key)
