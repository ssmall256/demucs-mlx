# Kernel Debugging Walkthrough

A step-by-step narrative for diagnosing and fixing a broken Metal kernel, from "wrong output" to "correct and fast."

## The Scenario

You've written a fused RMSNorm+SiLU kernel. It compiles and runs without crashing, but produces wrong output. The reference `mx.fast.rms_norm(x, w, eps) * mx.sigmoid(x) * x` gives different results.

## Step 1: Isolate the Symptom

Compare against the reference and categorize the failure:

```python
expected = mx.fast.rms_norm(x, w, eps) * mx.sigmoid(x) * x
actual = my_fused_kernel(x, w, eps)
mx.eval(expected, actual)

diff = mx.abs(expected - actual)
print(f"max_diff:  {mx.max(diff).item():.2e}")
print(f"mean_diff: {mx.mean(diff).item():.2e}")
print(f"any_nan:   {mx.any(mx.isnan(actual)).item()}")
print(f"all_zero:  {mx.all(actual == 0).item()}")
```

The symptom tells you where to look:
- **All zeros** → Kernel probably never wrote to output (grid too small, or writing to wrong offset)
- **NaN/Inf** → Division by zero, `rsqrt(negative)`, or uninitialized memory read
- **Close but wrong** (diff ~1e-2) → Numerical issue or incorrect reduction
- **Completely wrong** (diff ~1e+0) → Logic error in the kernel source

## Step 2: Check the Generated Source

Use `verbose=True` to print the full Metal function signature that MLX generates:

```python
out = kernel(
    inputs=[x, w, eps_buf],
    template=[("T", mx.float32)],
    grid=(N * tg, 1, 1),
    threadgroup=(tg, 1, 1),
    output_shapes=[(N, D)],
    output_dtypes=[mx.float32],
    verbose=True,  # Prints generated Metal source
)
```

Check that:
- Input order in the signature matches your `input_names` order
- Template type `T` resolved to what you expect
- Auto-injected parameters (`x_shape`, `x_ndim`, etc.) appear for every input you reference `_shape`/`_strides` on

## Step 3: Shrink the Problem

Test with the smallest possible input to isolate the bug:

```python
# Single row, single simdgroup (32 threads) — simplest case
x_tiny = mx.random.normal((1, 32))
w_tiny = mx.ones((32,))
# If this passes, the basic math is correct

# Single row, multi-simdgroup (256 threads)
x_med = mx.random.normal((1, 256))
w_med = mx.ones((256,))
# If this fails but (1,32) passes → cross-simdgroup reduction bug

# Multiple rows
x_multi = mx.random.normal((4, 256))
w_multi = mx.ones((256,))
# If this fails but (1,256) passes → row indexing bug
```

**Key insight**: If the kernel works for D ≤ 32 (single simdgroup) but fails for D > 32, the bug is almost certainly in the cross-simdgroup reduction (shared memory, barriers, or `num_sg` calculation).

## Step 4: Debug with Output Buffers

Metal kernels can't print. Add extra outputs to capture intermediate values:

```python
debug_kernel = mx.fast.metal_kernel(
    name="debug_rmsnorm",
    input_names=["x", "w", "eps"],
    output_names=["out", "dbg_sum_sq", "dbg_rms"],
    source="""
    // ... same kernel code, but also write intermediates:
    if (tid == 0) {
        dbg_sum_sq[row] = shared[0];           // Total sum of squares
        dbg_rms[row] = metal::rsqrt(shared[0] / float(D) + eps[0]);
    }
    // ... rest of kernel
    """,
)

out, dbg_sum_sq, dbg_rms = debug_kernel(
    inputs=[x, w, eps_buf],
    template=[("T", mx.float32)],
    grid=(N * tg, 1, 1),
    threadgroup=(tg, 1, 1),
    output_shapes=[(N, D), (N,), (N,)],
    output_dtypes=[mx.float32, mx.float32, mx.float32],
)
mx.eval(out, dbg_sum_sq, dbg_rms)

# Compare against Python reference
ref_sum_sq = mx.sum(x * x, axis=-1)
print("sum_sq kernel:", dbg_sum_sq[:4].tolist())
print("sum_sq python:", ref_sum_sq[:4].tolist())
```

This pinpoints exactly which computation step diverges.

## Step 5: Grid and Threadgroup Math

The most common source of bugs. Print and verify:

```python
N = x_2d.shape[0]   # Number of rows
D = x_2d.shape[1]   # Hidden dimension
tg = 256             # Threadgroup size

print(f"N={N}, D={D}, tg={tg}")
print(f"grid=({N * tg}, 1, 1)")
print(f"num_threadgroups={N}")  # Should equal number of rows
```

**Common grid bugs:**

| Bug | What Happens |
|-----|-------------|
| `grid=(N, 1, 1)` with `tg=256` | Only `ceil(N/256)` threadgroups. Most rows never processed. |
| `grid=(N * D, 1, 1)` with `tg=256` | D/256 threadgroups per row. `threadgroup_position_in_grid.x` no longer equals row index. |
| `grid=(N * tg, 1, 1)` | Correct: exactly N threadgroups, one per row. |

**Common position bugs:**

| Bug | What Happens |
|-----|-------------|
| Using `thread_position_in_grid.x` as row | Each thread gets a unique global index, not a row index. |
| Using `threadgroup_position_in_grid.x` as row | Correct for reduction kernels with one threadgroup per row. |
| Using `thread_index_in_threadgroup.x` | Compile error: it's `uint`, not `uint3`. Use without `.x`. |

## Step 6: Check the Reduction

If your reduction produces wrong results, verify these three things:

**1. Number of simdgroups:**
```metal
uint num_sg = (tg_size + 31) / 32;
// For tg=256: num_sg=8. If you hardcode 8 but tg changes, bug.
```

**2. Barriers are in the right places:**
```metal
if (lane == 0) shared[sg] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);  // REQUIRED before reading shared[]

if (sg == 0) {
    float p = (lane < num_sg) ? shared[lane] : 0.0f;
    shared[0] = simd_sum(p);
}
threadgroup_barrier(mem_flags::mem_threadgroup);  // REQUIRED before using shared[0]
```

Missing either barrier causes a data race — the output may be correct sometimes and wrong other times.

**3. Shared memory size:**
```metal
threadgroup float shared[32];  // Enough for up to 32 simdgroups (1024 threads)
```

If your threadgroup has more than 32 simdgroups (> 1024 threads), the shared array is too small. Apple Silicon caps threadgroups at 1024, so `shared[32]` is always sufficient.

## Step 7: Validate Across Shapes

Once the kernel works for one shape, test a matrix of edge cases:

```python
test_shapes = [
    (1, 32),      # Single simdgroup
    (1, 33),      # Threads > elements (some threads idle)
    (1, 256),     # Exact threadgroup size
    (1, 4096),    # Large D, threads stride many elements
    (1, 16384),   # Very large D
    (4, 256),     # Multiple rows
    (64, 1024),   # Many rows
    (1, 1),       # Degenerate: D=1
    (1, 16),      # D < simdgroup width (32)
]
```

Pay special attention to cases where `D < threadgroup_size` — threads with `tid >= D` must contribute zero to the reduction, not garbage.

## Step 8: Benchmark

Once correct, check performance via bandwidth utilization:

```python
import time

bytes_read = x.size * x.itemsize + w.size * w.itemsize
bytes_written = out.size * out.itemsize
total_bytes = bytes_read + bytes_written

# Warmup
for _ in range(5):
    mx.eval(my_kernel(x, w))

mx.synchronize()
t0 = time.perf_counter()
iters = 100
for _ in range(iters):
    mx.eval(my_kernel(x, w))
mx.synchronize()
elapsed = (time.perf_counter() - t0) / iters

bw_gbps = total_bytes / elapsed / 1e9
print(f"Bandwidth: {bw_gbps:.1f} GB/s")
# Compare against chip peak (M1: 68, M2: 100, M3: 100, M4: 120 GB/s)
```

If bandwidth is low:
- Check memory access coalescing (adjacent threads should read adjacent addresses)
- Check if `ensure_row_contiguous` is copying your input unnecessarily
- Try different threadgroup sizes

## Quick Diagnosis Cheat Sheet

| Symptom | Likely Cause | Where to Look |
|---------|-------------|---------------|
| All zeros | Grid too small; output never written | Step 5: grid math |
| NaN everywhere | `rsqrt(negative)` or uninitialized read | Step 4: debug intermediate values |
| Correct for D≤32, wrong for D>32 | Cross-simdgroup reduction bug | Step 6: barriers, num_sg |
| Correct for 1 row, wrong for many | Row indexing error | Step 5: `threadgroup_position_in_grid.x` vs `thread_position_in_grid.x` |
| Intermittently wrong | Missing `threadgroup_barrier` (data race) | Step 6: barrier placement |
| Off by small amount (1e-3) | Float16 precision loss in accumulator | Use `float` for all intermediates |
| Correct but slow | Bad access pattern or small threadgroup | Step 8: benchmark bandwidth |
| Output shape mismatch | `output_shapes` doesn't match kernel writes | Check `output_shapes` parameter |
| Compile error: "not a structure" | Using `.x` on scalar built-in | `thread_index_in_threadgroup` is `uint`, not `uint3` |
