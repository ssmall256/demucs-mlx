# Attention Kernel Guide

## What Attention Computes

Scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- `Q` (queries): shape `(seq_q, d)`
- `K` (keys): shape `(seq_k, d)`
- `V` (values): shape `(seq_k, d)`
- `d` = head dimension (commonly 64 or 128)

The naive implementation materializes the full `(seq_q, seq_k)` attention matrix, which is O(n²) in memory. For long sequences, this dominates memory usage.

## Why Tiling Helps

Instead of materializing the entire attention matrix:
1. Load a **tile** of Q rows into threadgroup memory
2. Iterate over tiles of K and V, computing partial attention scores
3. Accumulate the output without ever storing the full attention matrix

This reduces memory from O(seq_q × seq_k) to O(tile_size × d), which fits in threadgroup memory.

## Memory Budget

Apple Silicon provides **32KB threadgroup memory**. For a tiled attention kernel:

| Component | Size (float16) | Size (float32) |
|-----------|---------------|----------------|
| Q tile: Bq × d | Bq × d × 2B | Bq × d × 4B |
| K tile: Bk × d | Bk × d × 2B | Bk × d × 4B |
| V tile: Bk × d | Bk × d × 2B | Bk × d × 4B |
| Scores: Bq × Bk | Bq × Bk × 4B | Bq × Bk × 4B |
| Output accum: Bq × d | Bq × d × 4B | Bq × d × 4B |

### Practical Configuration

For `d=128`, `float16` inputs with `float32` accumulation:

- **Bq=32, Bk=32**: Q tile=8KB, K tile=8KB, V tile=8KB, scores=4KB, output=16KB → Total ≈ 44KB — **too large**
- **Bq=16, Bk=32**: Q tile=4KB, K tile=8KB, V tile=8KB, scores=2KB, output=8KB → Total ≈ 30KB — **fits**
- **Bq=16, Bk=16**: Q tile=4KB, K tile=4KB, V tile=4KB, scores=1KB, output=8KB → Total ≈ 21KB — **comfortable**

With sequential reuse (load K tile, compute scores, then reuse K memory for V tile), you can use larger tiles.

## Online Softmax Algorithm

Standard softmax requires two passes: one for max, one for exp+sum. **Online softmax** computes softmax incrementally as K tiles arrive, using running statistics:

```
For each K tile j:
    1. scores = Q_tile @ K_tile_j^T / sqrt(d)
    2. new_max = max(running_max, max(scores))
    3. correction = exp(running_max - new_max)
    4. running_sum = running_sum * correction + sum(exp(scores - new_max))
    5. output = output * correction + exp(scores - new_max) @ V_tile_j
    6. running_max = new_max

Final: output = output / running_sum
```

The key insight: when a new tile has a larger max, we **rescale** the previously accumulated output by `exp(old_max - new_max)` to keep everything numerically consistent.

## Simplified Attention Kernel Structure

The kernel below shows the high-level structure. Each threadgroup handles `Bq` query rows:

```metal
// Thread/group indices
uint tg_q = threadgroup_position_in_grid.y;  // Query tile index
uint tid = thread_index_in_threadgroup;

// Threadgroup memory for tiles
threadgroup float q_tile[Bq * D];
threadgroup float k_tile[Bk * D];

// Per-thread accumulators (in registers)
float output[D_PER_THREAD];  // Output accumulator
float row_max = -1e38f;      // Running max
float row_sum = 0.0f;        // Running sum

// Load Q tile into threadgroup memory (once)
load_tile(q_tile, Q, tg_q * Bq, Bq, D);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Iterate over K/V tiles
for (uint kv_start = 0; kv_start < seq_k; kv_start += Bk) {
    // Load K tile
    load_tile(k_tile, K, kv_start, Bk, D);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute scores = Q_tile @ K_tile^T / sqrt(d)
    // Apply online softmax update
    // Load V tile (can reuse k_tile memory)
    // Accumulate: output += softmax_weights @ V_tile

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Final normalization: output /= row_sum
```

## Important Considerations

### Use the Built-in for Production

MLX provides `mx.fast.scaled_dot_product_attention` which is already highly optimized with proper tiling, multi-head support, and memory efficiency. **Use it for production workloads.**

Custom attention kernels are valuable for:
- Learning how tiled attention works
- Implementing custom attention patterns (sliding window, sparse, linear)
- Fusing attention with other operations (e.g., RoPE + attention)

### Numerical Stability

- Always accumulate scores and softmax in **float32**, even with float16 inputs
- The online softmax rescaling involves `exp(old_max - new_max)` — this is numerically safe because `old_max ≤ new_max`, so the exponent is ≤ 0
- Watch for edge cases: empty sequences, single-token sequences

### Performance Notes

- Attention is **compute-bound** for large `d` and **memory-bound** for small `d`
- The inner loop (score computation) is essentially a small matrix multiply — tile sizes should balance threadgroup memory usage against compute efficiency
- On Apple Silicon, simdgroup operations can accelerate the small matmul within each tile

## Reference Implementation

See `scripts/attention_kernel.py` for a working single-head attention implementation with online softmax, including correctness validation against the naive `softmax(Q @ K^T / sqrt(d)) @ V` computation.

The script uses **Bk=1** (one K/V row per iteration) for simplicity. To extend to Bk>1:
1. Allocate a threadgroup memory tile for K: `threadgroup float k_tile[Bk * D]`
2. Load Bk rows of K into `k_tile` each iteration
3. Compute a vector of Bk scores (instead of one scalar score)
4. Apply online softmax across the Bk scores before accumulating into the output
