# simdgroup_matrix Operations (M3+ Only)

## What Are simdgroup_matrix Operations?

Apple GPU family 9+ (M3, M4) provides hardware 8x8 matrix multiply-accumulate (MMA) operations through the `simdgroup_matrix` API. A single simdgroup (32 threads) collectively holds and processes an 8x8 matrix tile, with each thread storing 2 elements (8×8 / 32 = 2).

This is Apple's equivalent of NVIDIA's tensor cores / `wmma` instructions. It accelerates compute-bound operations like matrix multiplication by performing 8x8 FMAs in hardware.

**Three main operations:**
- `simdgroup_load(matrix, ptr, stride)` — Load an 8x8 tile from memory
- `simdgroup_store(matrix, ptr, stride)` — Store an 8x8 tile to memory
- `simdgroup_multiply_accumulate(D, A, B, C)` — D = A × B + C (8x8 MMA)

## Hardware Requirements

simdgroup_matrix requires **M3 or later** (Apple GPU family 9+). On M1/M2, the kernel will fail to compile.

**Detection in Python:**

```python
import mlx.core as mx

def supports_simdgroup_matrix() -> bool:
    """Check if the current device supports simdgroup_matrix (M3+)."""
    if not mx.metal.is_available():
        return False
    info = mx.device_info()
    name = info.get("device_name", "")
    # M3, M3 Pro, M3 Max, M3 Ultra, M4, M4 Pro, M4 Max...
    return any(chip in name for chip in ["M3", "M4", "M5", "M6"])
```

**Graceful fallback:**

```python
if supports_simdgroup_matrix():
    matmul = simdgroup_matmul  # Custom MMA kernel
else:
    matmul = lambda a, b: a @ b  # Standard MLX matmul
```

## API Reference

### Matrix Types

```metal
// Declare an 8x8 matrix of floats
simdgroup_matrix<float, 8, 8> mat;

// Initialize to zero
simdgroup_matrix<float, 8, 8> acc(0);
```

Supported element types: `float`, `half`, `bfloat` (M3+).

### simdgroup_load

```metal
simdgroup_load(matrix, pointer, stride);
```

Loads an 8x8 tile from device memory in row-major order.

- `pointer`: `const device T*` pointing to the top-left element of the tile
- `stride`: Number of elements between the start of consecutive rows (i.e., the leading dimension of the full matrix, not the tile)

```metal
// Load an 8x8 tile starting at row r, column c of a matrix with N columns
simdgroup_matrix<float, 8, 8> tile;
simdgroup_load(tile, a + r * N + c, N);
//                                  ^ stride = N (full matrix width)
```

### simdgroup_store

```metal
simdgroup_store(matrix, pointer, stride);
```

Stores an 8x8 tile back to device memory. Same pointer/stride semantics as load.

```metal
simdgroup_store(result, out + r * N + c, N);
```

### simdgroup_multiply_accumulate

```metal
simdgroup_multiply_accumulate(D, A, B, C);
// D = A * B + C  (8x8 matrix multiply-accumulate)
```

A and B are 8x8 input matrices, C is the accumulator, D is the output. D and C can be the same variable for in-place accumulation:

```metal
simdgroup_multiply_accumulate(acc, matA, matB, acc);  // acc += A * B
```

### Mixed Precision

Use `half` (or `bfloat`) for inputs and `float` for the accumulator to get the best of both worlds — reduced memory bandwidth and full-precision accumulation:

```metal
simdgroup_matrix<half, 8, 8> matA, matB;    // fp16 inputs
simdgroup_matrix<float, 8, 8> acc(0);        // fp32 accumulator

simdgroup_load(matA, (const device half*)(a + ...), K);
simdgroup_load(matB, (const device half*)(b + ...), N);
simdgroup_multiply_accumulate(acc, matA, matB, acc);
```

## Tiled GEMM Pattern

The canonical use case: each simdgroup computes one 8x8 output tile of C = A × B.

### Grid Mapping

```
Matrix C (M × N):
┌───────┬───────┬───────┐
│ (0,0) │ (0,1) │ (0,2) │  ← each cell is one 8x8 tile
├───────┼───────┼───────┤     handled by one simdgroup
│ (1,0) │ (1,1) │ (1,2) │
└───────┴───────┴───────┘

Grid: (32, M/8, N/8)
       │    │    └── col tile index (Z)
       │    └─────── row tile index (Y)
       └──────────── 32 threads per simdgroup (X)

Threadgroup: (32, 1, 1)  — one simdgroup per threadgroup
```

### Kernel Structure

```metal
uint row_tile = threadgroup_position_in_grid.y;   // Output tile row
uint col_tile = threadgroup_position_in_grid.z;   // Output tile column

uint M = a_shape[0];
uint K = a_shape[1];
uint N = b_shape[1];

// Accumulator initialized to zero
simdgroup_matrix<float, 8, 8> acc(0);

// Tile over the K dimension in steps of 8
for (uint kt = 0; kt < K; kt += 8) {
    simdgroup_matrix<float, 8, 8> matA, matB;

    // Load 8x8 tile of A starting at (row_tile*8, kt)
    simdgroup_load(matA, a + row_tile * 8 * K + kt, K);

    // Load 8x8 tile of B starting at (kt, col_tile*8)
    simdgroup_load(matB, b + kt * N + col_tile * 8, N);

    // Accumulate: acc += A_tile * B_tile
    simdgroup_multiply_accumulate(acc, matA, matB, acc);
}

// Store the 8x8 output tile
simdgroup_store(acc, c + row_tile * 8 * N + col_tile * 8, N);
```

### Python Setup

```python
M, K = a.shape
_, N = b.shape
M_tiles = (M + 7) // 8
N_tiles = (N + 7) // 8

out = kernel(
    inputs=[a, b],
    template=[("T", mx.float32)],
    grid=(32, M_tiles, N_tiles),
    threadgroup=(32, 1, 1),
    output_shapes=[(M, N)],
    output_dtypes=[mx.float32],
)[0]
```

## Performance Notes

### When simdgroup_matrix Helps

- **Compute-bound operations**: Matrix multiplication, attention score computation, convolution inner loops
- **Fused operations**: Combining a matmul with a custom post-processing step that the built-in doesn't support

### When It Doesn't Help

- **Memory-bound operations**: Element-wise ops, reductions, normalization — these are limited by bandwidth, not compute
- **Small matrices**: The 8x8 tile overhead isn't amortized for very small dimensions

### Compared to MLX Built-in

MLX's built-in `mx.matmul` already uses simdgroup_matrix internally with sophisticated multi-level tiling (threadgroup tiles of simdgroup tiles, double-buffered loads). A simple custom kernel will be **slower** than the built-in for plain GEMM.

Custom simdgroup_matrix kernels are valuable when:
1. **Learning**: Understanding how hardware MMA works
2. **Fused operations**: Matmul + activation + bias in one kernel
3. **Non-standard patterns**: Sparse matmul, custom attention, quantized operations

## Integration with metal_kernel

No explicit `#include` is needed — `<metal_simdgroup_matrix>` is auto-included by MLX's kernel compilation pipeline.

**Pointer casting**: When loading from template-typed buffers, cast to the concrete type:

```metal
simdgroup_matrix<float, 8, 8> mat;
simdgroup_load(mat, (const device float*)(a + offset), stride);
```

**With template types**: You can load `half` matrices from template `T` buffers when `T` is `half`:

```metal
simdgroup_matrix<T, 8, 8> mat;
simdgroup_load(mat, a + offset, stride);
```

## Limitations

- **Only 8x8 tile size**: Unlike NVIDIA tensor cores which support 16x16 and 32x32, Apple's MMA is fixed at 8x8
- **M3+ only**: Attempting to use on M1/M2 gives a compile error — always guard with device detection
- **Padding required**: Input dimensions must be multiples of 8 (or the kernel must handle edge tiles). Pad in the Python wrapper and slice the result
- **One simdgroup per tile**: The 32-thread simdgroup is the unit of work. For larger tiles, you need multiple simdgroups with explicit coordination via threadgroup memory

## References

- `scripts/simdgroup_matmul_kernel.py` — Working tiled GEMM example with f32 and mixed-precision variants
- `references/apple-silicon-optimization-guide.md` — GPU architecture details, SIMD operations
- Apple Metal Shading Language Specification, Section 6.9 — simdgroup matrix operations
