"""Runtime hardware topology detection and auto-tuning for Apple Silicon."""
from __future__ import annotations

import functools
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AppleSiliconTopology:
    """Hardware topology and auto-tuned runtime policy for the host system."""

    chip_name: str
    gpu_cores: int
    cpu_cores: int
    ram_bytes: int
    ram_gb: float
    estimated_bandwidth_gbps: float
    optimal_batch_size: int
    recommended_stream_policy: str
    recommended_backend: str

    def summary(self) -> str:
        return (
            f"{self.chip_name} ({self.gpu_cores} GPU cores, {self.cpu_cores} CPU cores, "
            f"{self.ram_gb:.1f} GB unified RAM, ~{self.estimated_bandwidth_gbps:.0f} GB/s bandwidth) -> "
            f"optimal batch_size={self.optimal_batch_size}, stream_policy={self.recommended_stream_policy}"
        )


def _get_sysctl(key: str) -> Optional[str]:
    try:
        res = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def _detect_gpu_cores(chip_name: str) -> int:
    """Detect GPU core count via IORegistry IOAccelerator service."""
    try:
        res = subprocess.run(
            ["ioreg", "-r", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', res.stdout)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    # Heuristic fallback based on known chip defaults
    chip_lower = chip_name.lower()
    if "m4 max" in chip_lower:
        return 40
    if "m4 pro" in chip_lower:
        return 20
    if "m4" in chip_lower:
        return 10
    if "m3 max" in chip_lower:
        return 40
    if "m3 pro" in chip_lower:
        return 18
    if "m3" in chip_lower:
        return 10
    if "m2 ultra" in chip_lower:
        return 60
    if "m2 max" in chip_lower:
        return 30
    if "m2 pro" in chip_lower:
        return 16
    if "m2" in chip_lower:
        return 10
    if "m1 ultra" in chip_lower:
        return 48
    if "m1 max" in chip_lower:
        return 24
    if "m1 pro" in chip_lower:
        return 16
    if "m1" in chip_lower:
        return 8
    return 8


def _estimate_memory_bandwidth(chip_name: str, gpu_cores: int) -> float:
    """Estimate memory bandwidth in GB/s based on Apple Silicon generation and core count."""
    chip_lower = chip_name.lower()
    if "m4 max" in chip_lower:
        return 546.0 if gpu_cores >= 38 else 410.0
    if "m4 pro" in chip_lower:
        return 273.0
    if "m4" in chip_lower:
        return 120.0
    if "m3 max" in chip_lower:
        return 400.0 if gpu_cores >= 36 else 300.0
    if "m3 pro" in chip_lower:
        return 150.0
    if "m3" in chip_lower:
        return 100.0
    if "m2 ultra" in chip_lower:
        return 800.0
    if "m2 max" in chip_lower:
        return 400.0
    if "m2 pro" in chip_lower:
        return 200.0
    if "m2" in chip_lower:
        return 100.0
    if "m1 ultra" in chip_lower:
        return 800.0
    if "m1 max" in chip_lower:
        return 400.0
    if "m1 pro" in chip_lower:
        return 200.0
    if "m1" in chip_lower:
        return 68.25
    return 100.0


@functools.lru_cache(maxsize=1)
def get_topology() -> AppleSiliconTopology:
    """Detect Apple Silicon hardware topology and return cached auto-tuning policy."""
    is_darwin = sys.platform == "darwin"
    is_arm = platform.machine() == "arm64"

    if is_darwin and is_arm:
        brand = _get_sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        hw_ncpu = _get_sysctl("hw.ncpu")
        cpu_cores = int(hw_ncpu) if hw_ncpu and hw_ncpu.isdigit() else os.cpu_count() or 8
        memsize_str = _get_sysctl("hw.memsize")
        ram_bytes = int(memsize_str) if memsize_str and memsize_str.isdigit() else 16 * (1024**3)
        gpu_cores = _detect_gpu_cores(brand)
    else:
        brand = platform.processor() or "Generic CPU/GPU"
        cpu_cores = os.cpu_count() or 4
        ram_bytes = 16 * (1024**3)
        gpu_cores = 8

    ram_gb = ram_bytes / (1024**3)
    bandwidth = _estimate_memory_bandwidth(brand, gpu_cores)

    # Optimal batch size determination:
    # - M4 Max 40-core with 128GB: batch size 8 achieves peak saturation (88.8x RTFx).
    # - M4 Max 32-core / M4 Pro 20-core with 32-64GB: batch size 4 avoids allocator thrash and peaks at 74x / 49x.
    # - Base M-series or <= 16GB RAM: batch size 2 balances memory and latency.
    if gpu_cores >= 38 and ram_gb >= 64.0:
        opt_batch = 8
    elif gpu_cores >= 18 and ram_gb >= 32.0:
        opt_batch = 4
    elif gpu_cores >= 14 and ram_gb >= 16.0:
        opt_batch = 2
    else:
        opt_batch = 2

    # Stream policy determination:
    # Chips with >= 16 GPU cores have sufficient execution units to overlap Time and Spectral branches
    # on independent Metal command queues without compute starvation.
    stream_policy = "dual_stream" if gpu_cores >= 16 else "single_stream"
    backend = "mlx_gpu"

    return AppleSiliconTopology(
        chip_name=brand,
        gpu_cores=gpu_cores,
        cpu_cores=cpu_cores,
        ram_bytes=ram_bytes,
        ram_gb=ram_gb,
        estimated_bandwidth_gbps=bandwidth,
        optimal_batch_size=opt_batch,
        recommended_stream_policy=stream_policy,
        recommended_backend=backend,
    )


def optimal_batch_size() -> int:
    """Return the auto-tuned optimal batch size for the current machine."""
    return get_topology().optimal_batch_size


def optimal_stream_policy() -> str:
    """Return the auto-tuned optimal stream policy ('dual_stream' or 'single_stream')."""
    return get_topology().recommended_stream_policy
