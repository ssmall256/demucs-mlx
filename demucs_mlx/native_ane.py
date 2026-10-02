"""Zero-GIL native Core ML dispatch bridge for Demucs on Apple Silicon.

Buffers are MLX arrays handed to Core ML by address: MLX exposes its unified
memory through the buffer protocol, so the bridge reads the input and writes
the output in place with no copies and no NumPy.
"""
from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .ane_paths import ane_cache_dir

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

    cache_dir = ane_cache_dir()
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
        except Exception:
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


class MLXBuffer:
    """The raw memory of an evaluated, row-contiguous MLX array.

    Holds a buffer export for as long as it lives, so the address stays valid;
    release it (``with`` or ``release()``) before the array is used again.
    """

    def __init__(self, array: mx.array, *, dtype: mx.Dtype):
        if array.dtype != dtype:
            raise TypeError(f"expected {dtype}, got {array.dtype}")
        mx.eval(array)
        self._view = memoryview(array)
        if not self._view.c_contiguous:
            self._view.release()
            raise ValueError("array must be row-contiguous")
        self._bytes = (ctypes.c_char * self._view.nbytes).from_buffer(self._view)
        self.address = ctypes.addressof(self._bytes)

    def release(self) -> None:
        if self._bytes is not None:
            self._bytes = None
            self._view.release()

    def __enter__(self) -> "MLXBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def predict_waveform_conv_native(
    model_path: Path | str,
    input_data: mx.array,
    output_target: mx.array,
) -> bool:
    """
    Run the waveform convolution on the Neural Engine via zero-GIL native dispatch.

    ``input_data`` is float32 (N, 2, 343980); ``output_target`` is a float16
    (N, 48, 85995) array that is written in place. Returns True on success,
    False if the PyObjC fallback is required.
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

    count = int(input_data.shape[0])
    with MLXBuffer(input_data, dtype=mx.float32) as src, MLXBuffer(
        output_target, dtype=mx.float16
    ) as dst:
        res = lib.predict_conv_batch(
            ctypes.c_void_p(src.address),
            ctypes.c_void_p(dst.address),
            ctypes.c_int(count),
        )
    return res == 0
