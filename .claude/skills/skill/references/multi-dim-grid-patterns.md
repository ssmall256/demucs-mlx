# Multi-Dimensional Grid Patterns

## Overview

MLX's `metal_kernel` supports 3D grids: `grid=(X, Y, Z)`. Multi-dimensional grids simplify batched operations by mapping batch/row indices to grid dimensions instead of computing them from a flat thread ID.

## How 2D Grids Work

```python
grid=(X, Y, 1)
threadgroup=(TGX, TGY, 1)
```

Metal launches `ceil(X/TGX) * ceil(Y/TGY)` threadgroups. Each thread gets:
- `thread_position_in_grid.x` — element index (0 to X-1)
- `thread_position_in_grid.y` — batch/row index (0 to Y-1)

This is equivalent to a nested loop:
```python
for y in range(Y):       # batch dimension
    for x in range(X):   # element dimension
        kernel(x, y)
```

## Pattern: Batched Element-wise Operation

Apply a pointwise operation across a batch of vectors. The Y dimension indexes the batch, X indexes elements within each vector.

### Metal Source

```metal
uint elem = thread_position_in_grid.x;   // Element index
uint batch = thread_position_in_grid.y;   // Batch index
uint D = x_shape[x_ndim - 1];

if (elem >= D) return;

uint idx = batch * D + elem;
float val = (float)x[idx];
// ... apply operation ...
out[idx] = (T)result;
```

### Python Grid Setup

```python
N = x_2d.shape[0]   # Number of rows/batches
D = x_2d.shape[1]   # Elements per row

# X = elements per row, Y = number of rows
grid = (D, N, 1)
threadgroup = (min(256, D), 1, 1)
```

**Why `(D, N, 1)` instead of `(N*D, 1, 1)`?** The 2D grid eliminates manual index math (`batch = gid / D; elem = gid % D`) and avoids integer division, which is slow on GPUs.

## Pattern: Batched Reduction (One Threadgroup per Row)

For reductions (sum, max, softmax) across the last dimension of a batched tensor, use the Y dimension for the batch and X for threads within the reduction.

### Metal Source

```metal
uint tid = thread_index_in_threadgroup;
uint row = threadgroup_position_in_grid.y;   // Batch/row from Y
uint D = x_shape[x_ndim - 1];

// Each thread reduces strided elements within the row
float acc = 0.0f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    acc += (float)x[row * D + i];
}
acc = simd_sum(acc);
// ... cross-simdgroup reduction as usual ...
```

### Python Grid Setup

```python
N = x_2d.shape[0]
D = x_2d.shape[1]
tg = min(256, D)
tg = max(32, (tg // 32) * 32)

# One threadgroup per row, rows in Y dimension
grid = (tg, N, 1)
threadgroup = (tg, 1, 1)
```

**Note**: Here `grid.x = tg` (exactly one threadgroup wide), and `grid.y = N` gives one row per Y index. This is equivalent to the 1D pattern `grid=(N*tg, 1, 1)` with `row = threadgroup_position_in_grid.x`, but more readable.

## Comparison: 1D vs 2D Grid

| Approach | Grid | Row Index | Element Index |
|----------|------|-----------|---------------|
| 1D flat | `(N*D, 1, 1)` | `thread_position_in_grid.x / D` | `thread_position_in_grid.x % D` |
| 2D | `(D, N, 1)` | `thread_position_in_grid.y` | `thread_position_in_grid.x` |
| 1D reduction | `(N*tg, 1, 1)` | `threadgroup_position_in_grid.x` | `thread_index_in_threadgroup` |
| 2D reduction | `(tg, N, 1)` | `threadgroup_position_in_grid.y` | `thread_index_in_threadgroup` |

The 2D approach is cleaner for batched operations. Use 1D when you have a single flat array or when the operation doesn't have a natural batch dimension.

## When to Use 3D Grids

3D grids (`grid=(X, Y, Z)`) are useful for operations with three natural dimensions:
- **Batched matrix ops**: X = column, Y = row, Z = batch
- **3D convolutions**: X = width, Y = height, Z = depth/batch
- **Multi-head attention**: X = element, Y = query position, Z = head

```python
# Example: per-head, per-position operation
grid = (head_dim, seq_len, num_heads)
threadgroup = (min(256, head_dim), 1, 1)
```

In the kernel:
```metal
uint elem = thread_position_in_grid.x;   // Within head
uint pos  = thread_position_in_grid.y;   // Sequence position
uint head = thread_position_in_grid.z;   // Which head
```

## Pattern: 4D Tensor Operations (Batch, Heads, Seq, Dim)

Transformer models operate on 4D tensors of shape `(batch, heads, seq, dim)`. Since Metal grids are 3D, you need to flatten one dimension pair into a single grid axis.

### Strategy 1: Flatten batch × heads into Z

The most common approach. Decompose the Z index inside the kernel:

```python
B, H, S, D = x.shape
grid = (D, S, B * H)              # or (D // 2, S, B * H) for pair-wise ops
threadgroup = (min(256, D), 1, 1)
```

```metal
uint elem = thread_position_in_grid.x;
uint pos  = thread_position_in_grid.y;
uint bh   = thread_position_in_grid.z;   // Composite index

uint B = x_shape[0];
uint H = x_shape[1];
uint S = x_shape[2];
uint D = x_shape[3];

// Decompose composite index
uint batch_idx = bh / H;
uint head_idx  = bh % H;

// Flat index into contiguous (B, H, S, D) tensor
uint idx = batch_idx * (H * S * D) + head_idx * (S * D) + pos * D + elem;
```

This works well when the input is contiguous in `(B, H, S, D)` layout. The integer division `bh / H` and modulo `bh % H` are computed once per thread and are negligible.

### Strategy 2: `elem_to_loc` with strides for non-contiguous inputs

After operations like `mx.transpose`, the tensor may have non-standard strides. Instead of an implicit copy, use `ensure_row_contiguous=False` and read strides in the kernel:

```python
kernel = mx.fast.metal_kernel(
    name="my_4d_op",
    input_names=["x"],
    output_names=["out"],
    source=STRIDED_SOURCE,
    ensure_row_contiguous=False,  # Skip implicit copy
)
```

```metal
uint elem = thread_position_in_grid.x;
uint pos  = thread_position_in_grid.y;
uint bh   = thread_position_in_grid.z;

uint H = x_shape[1];
uint batch_idx = bh / H;
uint head_idx  = bh % H;

// Use strides for non-contiguous access
size_t offset = batch_idx * x_strides[0]
              + head_idx  * x_strides[1]
              + pos       * x_strides[2]
              + elem      * x_strides[3];

float val = (float)x[offset];
```

Write the output to a contiguous buffer (output is always contiguous).

### Comparison

| Approach | Pros | Cons |
|----------|------|------|
| Flatten B×H (contiguous) | Fast, simple indexing | Requires contiguous input |
| Strides (non-contiguous) | No copy overhead | Slightly slower index math |

**Rule of thumb**: Use Strategy 1 when you control the input layout (most cases). Use Strategy 2 when the input comes from user code and may be transposed or sliced.

See `scripts/multihead_rope_kernel.py` for a working example of both approaches on 4D tensors.
