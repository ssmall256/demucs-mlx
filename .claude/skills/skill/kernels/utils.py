"""
Small utilities for MLX Metal-kernel scripts.

This module is intentionally defensive: it avoids hard dependency on particular
MLX versions by using getattr checks where possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json
import threading
from typing import Any, Optional, Tuple

try:
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover
    mx = None  # type: ignore


# ------------------------------
# Device / capability helpers
# ------------------------------

def metal_available() -> bool:
    if mx is None:
        return False
    metal = getattr(mx, "metal", None)
    return bool(metal and getattr(metal, "is_available", lambda: False)())


def device_key() -> str:
    """
    Best-effort device identifier used in on-disk caches.

    We keep this very robust so the skill runs across MLX versions.
    """
    if mx is None:
        return "no-mlx"
    metal = getattr(mx, "metal", None)
    # Some MLX builds expose device_info() / device_name()
    for attr in ("device_info", "device_name", "get_device_name"):
        if metal is not None and hasattr(metal, attr):
            try:
                v = getattr(metal, attr)()
                if isinstance(v, dict):
                    # Prefer stable / meaningful keys
                    for k in ("name", "device_name", "gpu_name"):
                        if k in v and v[k]:
                            return str(v[k])
                    return json.dumps(v, sort_keys=True)
                if v:
                    return str(v)
            except Exception:
                pass
    # Fallback: default device string
    for attr in ("default_device", "Device"):
        if hasattr(mx, attr):
            try:
                return str(getattr(mx, attr)())
            except Exception:
                pass
    return "unknown-device"


def supports_bfloat16() -> bool:
    """
    bfloat16 support is generally M3+ on Apple GPUs.
    We probe via dtype availability + a tiny kernel compile attempt if possible.
    """
    if mx is None:
        return False
    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is None:
        return False
    # If Metal isn't available, treat as unsupported for GPU kernels.
    if not metal_available():
        return False
    # If MLX exposes a direct capability flag, prefer it.
    metal = getattr(mx, "metal", None)
    for attr in ("supports_bfloat16", "supports_bf16"):
        if metal is not None and hasattr(metal, attr):
            try:
                return bool(getattr(metal, attr)())
            except Exception:
                pass
    # Conservative default: claim False unless explicitly known.
    return False


# ------------------------------
# Cached scalar buffers
# ------------------------------

_SCALAR_LOCK = threading.Lock()
_SCALAR_F32_CACHE: dict[float, Any] = {}
_SCALAR_I32_CACHE: dict[int, Any] = {}

def scalar_f32(value: float):
    """
    Return a cached mx.array([value], float32).
    Helpful to avoid allocating tiny buffers inside hot call paths.
    """
    if mx is None:
        raise RuntimeError("mlx is not available")
    with _SCALAR_LOCK:
        arr = _SCALAR_F32_CACHE.get(value)
        if arr is None:
            arr = mx.array([value], dtype=mx.float32)
            _SCALAR_F32_CACHE[value] = arr
        return arr

def scalar_i32(value: int):
    if mx is None:
        raise RuntimeError("mlx is not available")
    with _SCALAR_LOCK:
        arr = _SCALAR_I32_CACHE.get(value)
        if arr is None:
            arr = mx.array([value], dtype=mx.int32)
            _SCALAR_I32_CACHE[value] = arr
        return arr
