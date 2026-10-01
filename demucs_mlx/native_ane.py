"""Zero-GIL native Core ML dispatch bridge for Demucs on Apple Silicon."""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_LIB_HANDLE: Optional[ctypes.CDLL] = None
_LIB_INITIALIZED_MODEL: Optional[str] = None


def get_native_ane_lib() -> Optional[ctypes.CDLL]:
    """Compile or load the cached native Core ML dispatch dynamic library."""
    global _LIB_HANDLE
    if _LIB_HANDLE is not None:
        return _LIB_HANDLE

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None

    csrc_path = Path(__file__).parent / "csrc" / "demucs_ane.m"
    if not csrc_path.is_file():
        return None

    cache_dir = Path.home() / ".cache" / "demucs-mlx" / "ane"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dylib_path = cache_dir / "libdemucs_ane.dylib"

    # Recompile if dylib does not exist or source is newer
    needs_compile = not dylib_path.is_file() or (
        dylib_path.stat().st_mtime < csrc_path.stat().st_mtime
    )

    if needs_compile:
        try:
            cmd = [
                "clang",
                "-O3",
                "-fobjc-arc",
                "-shared",
                "-fPIC",
                "-framework",
                "Foundation",
                "-framework",
                "CoreML",
                str(csrc_path),
                "-o",
                str(dylib_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as exc:
            # Fall back to PyObjC if clang compilation fails
            return None

    try:
        lib = ctypes.CDLL(str(dylib_path))
        lib.init_ane_conv.argtypes = [ctypes.c_char_p]
        lib.init_ane_conv.restype = ctypes.c_int
        lib.predict_conv_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.predict_conv_batch.restype = ctypes.c_int
        _LIB_HANDLE = lib
        return _LIB_HANDLE
    except Exception:
        return None


def is_native_ane_available() -> bool:
    """Return True if native Core ML compilation and loading succeeds."""
    return get_native_ane_lib() is not None


def predict_waveform_conv_native(
    model_path: Path | str,
    input_data: np.ndarray,
    output_target: np.ndarray,
) -> bool:
    """
    Execute waveform convolution on the Neural Engine via zero-GIL native dispatch.
    Returns True if successful, False if fallback is required.
    """
    lib = get_native_ane_lib()
    if lib is None:
        return False

    global _LIB_INITIALIZED_MODEL
    path_str = str(model_path)
    if _LIB_INITIALIZED_MODEL != path_str:
        ret = lib.init_ane_conv(path_str.encode("utf-8"))
        if ret != 0:
            return False
        _LIB_INITIALIZED_MODEL = path_str

    if not input_data.flags.c_contiguous:
        input_data = np.ascontiguousarray(input_data)
    if not output_target.flags.c_contiguous:
        output_target = np.ascontiguousarray(output_target)

    count = int(input_data.shape[0])
    res = lib.predict_conv_batch(
        ctypes.c_void_p(input_data.ctypes.data),
        ctypes.c_void_p(output_target.ctypes.data),
        ctypes.c_int(count),
    )
    return res == 0
