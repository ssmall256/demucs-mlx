"""Matrix multiply using simdgroup_matrix 8x8 hardware MMA (M3+ only).

Demonstrates:
- simdgroup_matrix<T, 8, 8> type declaration
- simdgroup_load / simdgroup_store for 8x8 tiles
- simdgroup_multiply_accumulate for hardware MMA
- Mixed-precision: half inputs with float32 accumulator
- Tiled GEMM: one simdgroup per 8x8 output tile, K-dimension loop
- Graceful skip on M1/M2 (Apple GPU family < 9)

simdgroup_matrix operations are Apple's equivalent of NVIDIA tensor cores.
Each simdgroup (32 threads) collectively processes an 8x8 matrix tile.
For production GEMM, prefer mx.matmul (which already uses these internally
with multi-level tiling). Custom kernels are useful for fused operations.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def supports_simdgroup_matrix() -> bool:
    """Check if the current device supports simdgroup_matrix (M3+)."""
    if not mx.metal.is_available():
        return False
    info = mx.device_info()
    name = info.get("device_name", "")
    return any(chip in name for chip in ["M3", "M4", "M5", "M6"])


# ---------------------------------------------------------------------------
# Float32 GEMM kernel: one simdgroup per 8x8 output tile
# Grid: (32, M_tiles, N_tiles) — 32 threads per simdgroup
# ---------------------------------------------------------------------------
SIMDGROUP_MATMUL_F32_SOURCE = """
uint row_tile = threadgroup_position_in_grid.y;
uint col_tile = threadgroup_position_in_grid.z;

uint M = a_shape[0];
uint K = a_shape[1];
uint N = b_shape[1];

// Each simdgroup computes one 8x8 output tile
simdgroup_matrix<float, 8, 8> acc(0);

// Tile over K dimension in steps of 8
for (uint kt = 0; kt < K; kt += 8) {
    simdgroup_matrix<float, 8, 8> matA, matB;

    // Load 8x8 tile of A starting at (row_tile*8, kt)
    // Stride = K (number of columns in the full matrix)
    simdgroup_load(matA, (const device float*)(a + row_tile * 8 * K + kt), K);

    // Load 8x8 tile of B starting at (kt, col_tile*8)
    simdgroup_load(matB, (const device float*)(b + kt * N + col_tile * 8), N);

    // Accumulate: acc += A_tile * B_tile
    simdgroup_multiply_accumulate(acc, matA, matB, acc);
}

// Store result tile at (row_tile*8, col_tile*8)
simdgroup_store(acc, (device float*)(c + row_tile * 8 * N + col_tile * 8), N);
"""

# ---------------------------------------------------------------------------
# Mixed-precision kernel: half inputs, float32 accumulator, half output
# ---------------------------------------------------------------------------
SIMDGROUP_MATMUL_F16_SOURCE = """
uint row_tile = threadgroup_position_in_grid.y;
uint col_tile = threadgroup_position_in_grid.z;

uint M = a_shape[0];
uint K = a_shape[1];
uint N = b_shape[1];

// Float32 accumulator for numerical stability
simdgroup_matrix<float, 8, 8> acc(0);

for (uint kt = 0; kt < K; kt += 8) {
    // Half-precision input tiles
    simdgroup_matrix<half, 8, 8> matA, matB;

    simdgroup_load(matA, (const device half*)(a + row_tile * 8 * K + kt), K);
    simdgroup_load(matB, (const device half*)(b + kt * N + col_tile * 8), N);

    // Mixed-precision MMA: half * half + float -> float
    simdgroup_multiply_accumulate(acc, matA, matB, acc);
}

// Store float32 accumulator to output (output_dtypes=[mx.float32])
// Cast c to float* BEFORE adding offset — c is typed as device half*
// due to template T=half, but the output buffer is allocated as float32
simdgroup_store(acc, (device float*)c + row_tile * 8 * N + col_tile * 8, N);
"""


# Compile kernels once at module level
_simdgroup_matmul_f32_kernel = mx.fast.metal_kernel(
    name="simdgroup_matmul_f32",
    input_names=["a", "b"],
    output_names=["c"],
    source=SIMDGROUP_MATMUL_F32_SOURCE,
)

_simdgroup_matmul_f16_kernel = mx.fast.metal_kernel(
    name="simdgroup_matmul_f16",
    input_names=["a", "b"],
    output_names=["c"],
    source=SIMDGROUP_MATMUL_F16_SOURCE,
)


def _pad_to_8(x: mx.array, axis: int) -> mx.array:
    """Pad a dimension to a multiple of 8."""
    size = x.shape[axis]
    remainder = size % 8
    if remainder == 0:
        return x
    pad_size = 8 - remainder
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_size)
    return mx.pad(x, pad_width)


def simdgroup_matmul(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiply using simdgroup_matrix MMA (float32).

    Args:
        a: Shape (M, K).
        b: Shape (K, N).

    Returns:
        Result of shape (M, N).
    """
    M_orig, K = a.shape
    _, N_orig = b.shape

    # Pad to multiples of 8
    a_padded = _pad_to_8(_pad_to_8(a, 0), 1)
    b_padded = _pad_to_8(_pad_to_8(b, 0), 1)
    M_pad, K_pad = a_padded.shape
    _, N_pad = b_padded.shape

    M_tiles = M_pad // 8
    N_tiles = N_pad // 8

    out = _simdgroup_matmul_f32_kernel(
        inputs=[a_padded, b_padded],
        template=[("T", mx.float32)],
        grid=(32, M_tiles, N_tiles),
        threadgroup=(32, 1, 1),
        output_shapes=[(M_pad, N_pad)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]

    return out[:M_orig, :N_orig]


def simdgroup_matmul_f16(a: mx.array, b: mx.array) -> mx.array:
    """Matrix multiply with half inputs and float32 accumulator.

    Args:
        a: Shape (M, K), float16.
        b: Shape (K, N), float16.

    Returns:
        Result of shape (M, N), float32 (accumulated in fp32).
    """
    M_orig, K = a.shape
    _, N_orig = b.shape

    a_padded = _pad_to_8(_pad_to_8(a, 0), 1)
    b_padded = _pad_to_8(_pad_to_8(b, 0), 1)
    M_pad, K_pad = a_padded.shape
    _, N_pad = b_padded.shape

    M_tiles = M_pad // 8
    N_tiles = N_pad // 8

    # Output is float32 (from the float accumulator)
    out = _simdgroup_matmul_f16_kernel(
        inputs=[a_padded, b_padded],
        template=[("T", mx.float16)],
        grid=(32, M_tiles, N_tiles),
        threadgroup=(32, 1, 1),
        output_shapes=[(M_pad, N_pad)],
        output_dtypes=[mx.float32],
        init_value=0,
    )[0]

    return out[:M_orig, :N_orig]


def validate(dtype=mx.float32):
    """Validate simdgroup matmul against mx.matmul."""
    configs = [
        (8, 8, 8),
        (16, 32, 16),
        (64, 128, 64),
        (24, 16, 24),     # Non-multiple-of-8: tests padding
        (8, 24, 8),
    ]

    fn = simdgroup_matmul if dtype == mx.float32 else simdgroup_matmul_f16
    label = "f32" if dtype == mx.float32 else "f16 (f32 acc)"

    print(f"Validating simdgroup matmul — {label}")
    print("-" * 60)

    all_passed = True
    for M, K, N in configs:
        a = mx.random.normal((M, K)).astype(dtype)
        b = mx.random.normal((K, N)).astype(dtype)

        expected = (a.astype(mx.float32) @ b.astype(mx.float32))
        actual = fn(a, b)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        atol = 1e-2 if dtype == mx.float16 else 1e-4
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label_s = f"({M}x{K}) @ ({K}x{N})"
        print(f"  {label_s:>25s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    print("-" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== simdgroup_matrix Matmul Demo (M3+ Only) ===\n")

    if not supports_simdgroup_matrix():
        info = mx.device_info()
        print(f"Device: {info.get('device_name', 'unknown')}")
        print("simdgroup_matrix requires M3 or later. Skipping.")
        raise SystemExit(0)

    info = mx.device_info()
    print(f"Device: {info.get('device_name', 'unknown')}")
    print()

    # Quick demo
    a = mx.random.normal((16, 32))
    b = mx.random.normal((32, 16))
    c = simdgroup_matmul(a, b)
    mx.eval(c)
    print(f"A: {a.shape}, B: {b.shape} -> C: {c.shape}")
    print(f"Max diff vs mx.matmul: {mx.max(mx.abs(c - (a @ b))).item():.2e}")
    print()

    # Validate
    ok1 = validate(dtype=mx.float32)
    print()
    ok2 = validate(dtype=mx.float16)

    if not (ok1 and ok2):
        raise SystemExit(1)
