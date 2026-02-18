"""
Autotune + caching for MLX metal kernels.

Goals:
- Keep it simple and copy-paste friendly.
- Avoid "bench lies" from lazy eval by always synchronizing around timing.
- Provide both in-memory and on-disk caching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover
    mx = None  # type: ignore

from .utils import device_key, metal_available

_CACHE_LOCK = threading.Lock()
_MEM_CACHE: Dict[str, int] = {}

def _cache_path() -> Path:
    base = os.environ.get("MLX_METAL_KERNELS_CACHE_DIR") or str(Path.home() / ".cache" / "mlx-metal-kernels-skill")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / "autotune.json"

def _load_disk() -> Dict[str, int]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _save_disk(d: Dict[str, int]) -> None:
    path = _cache_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(path)

def _make_key(kernel_name: str, shape_sig: str, dtype_sig: str) -> str:
    return f"{device_key()}::{kernel_name}::{shape_sig}::{dtype_sig}"

def autotune_enabled() -> bool:
    """
    Enable by setting MLX_METAL_AUTOTUNE=1 (or true/yes).
    """
    v = os.environ.get("MLX_METAL_AUTOTUNE", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def pick_threadgroup(
    *,
    kernel_name: str,
    shape_sig: str,
    dtype_sig: str,
    candidates: Iterable[int],
    run: Callable[[int], Any],
    warmup: int = 2,
    iters: int = 20,
    default: int = 256,
) -> int:
    """
    Pick the fastest threadgroup-x from candidates.

    Parameters
    ----------
    kernel_name, shape_sig, dtype_sig:
        Used only for caching.
    candidates:
        Iterable of tgx values (must be <= 1024 on Metal).
    run:
        Callable taking tgx and executing one forward pass (must return an mx.array or similar).
    """
    if mx is None or not metal_available():
        return default

    key = _make_key(kernel_name, shape_sig, dtype_sig)

    with _CACHE_LOCK:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]

        disk = _load_disk()
        if key in disk:
            _MEM_CACHE[key] = int(disk[key])
            return _MEM_CACHE[key]

    if not autotune_enabled():
        with _CACHE_LOCK:
            _MEM_CACHE[key] = default
        return default

    # Filter candidates
    cand = [int(c) for c in candidates if int(c) > 0 and int(c) <= 1024]
    if not cand:
        cand = [default]

    # Warmup
    for _ in range(max(0, warmup)):
        y = run(cand[0])
        try:
            mx.eval(y)
            mx.synchronize()
        except Exception:
            pass

    best_tgx = cand[0]
    best_ms = float("inf")

    for tgx in cand:
        # One-time compile path is included; that's OK because we cache results.
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            y = run(tgx)
        try:
            mx.eval(y)
            mx.synchronize()
        except Exception:
            pass
        end = time.perf_counter()
        ms = (end - start) * 1000.0 / max(1, iters)
        if ms < best_ms:
            best_ms = ms
            best_tgx = tgx

    with _CACHE_LOCK:
        _MEM_CACHE[key] = int(best_tgx)
        disk = _load_disk()
        disk[key] = int(best_tgx)
        _save_disk(disk)

    return int(best_tgx)
