# Kernel Performance Profiling Guide

A systematic workflow for diagnosing and fixing kernel performance issues. Use this when your kernel is correct but slower than expected.

## Step 1: Establish Baselines

Before optimizing, you need two numbers:

**Measured baseline**: Time the MLX built-in (if one exists) and your custom kernel:

```python
import time
import mlx.core as mx

def benchmark(fn, *args, warmup=5, iters=20):
    """Benchmark a function with proper synchronization."""
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters

t_builtin = benchmark(lambda x, w: mx.fast.rms_norm(x, w, 1e-5), x, w)
t_custom = benchmark(my_rmsnorm, x, w)
print(f"Built-in: {t_builtin*1e6:.0f} us, Custom: {t_custom*1e6:.0f} us")
```

**Theoretical floor**: The minimum time based on memory bandwidth:

```python
bytes_accessed = x.size * x.itemsize + w.size * w.itemsize + out.size * out.itemsize
peak_bw = 120e9  # Bytes/sec — adjust for your chip (M4 base = 120 GB/s)
theoretical_min = bytes_accessed / peak_bw
print(f"Theoretical floor: {theoretical_min*1e6:.0f} us")
```

## Step 2: Diagnostic Decision Tree

```
Is the kernel correct?
├── No → See references/kernel-debugging-walkthrough.md
└── Yes
    Is it slower than the built-in?
    ├── No → Done (or continue optimizing for learning)
    └── Yes
        Compute achieved bandwidth (Step 3)
        │
        ├── >= 70% of peak bandwidth
        │   Kernel is near-optimal for a memory-bound operation.
        │   To go faster: fuse with adjacent operations to reduce
        │   total memory traffic (e.g., RMSNorm + residual add).
        │
        ├── 30-70% of peak bandwidth
        │   Room for improvement. Check:
        │   ├── Memory access coalesced? (adjacent threads → adjacent addresses)
        │   ├── Threadgroup size optimal? (try autotuning 32–1024)
        │   ├── Unnecessary barriers? (each costs GPU cycles)
        │   └── Redundant global reads? (cache values in registers)
        │
        └── < 30% of peak bandwidth
            Major issue. Check:
            ├── Grid/threadgroup mismatch? (too few threads = GPU idle)
            ├── Tensor too small? (launch overhead dominates for < 1K elements)
            ├── Hidden ensure_row_contiguous copy? (adds O(N) copy)
            └── Too many local variables? (register spilling to memory)
```

## Step 3: Measuring Achieved Bandwidth

The key metric for memory-bound kernels. If your kernel is close to peak bandwidth, it's running well.

```python
def measure_bandwidth(fn, *args, iters=50):
    """Measure achieved memory bandwidth in GB/s."""
    # Count total bytes read + written
    bytes_read = sum(a.size * a.itemsize for a in args if isinstance(a, mx.array))
    out = fn(*args)
    mx.eval(out)
    if isinstance(out, (list, tuple)):
        bytes_written = sum(o.size * o.itemsize for o in out)
    else:
        bytes_written = out.size * out.itemsize

    total_bytes = bytes_read + bytes_written

    # Time the kernel
    for _ in range(5):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    bw_gbps = total_bytes / elapsed / 1e9
    return bw_gbps, elapsed
```

**How to count bytes** for common operations:

| Operation | Bytes Read | Bytes Written |
|-----------|-----------|---------------|
| RMSNorm (N, D) | N×D + D (x + weight) | N×D (output) |
| LayerNorm (N, D) | N×D + 2×D (x + weight + bias) | N×D |
| Softmax (N, D) | N×D | N×D |
| Element-wise (N,) | N (input) | N (output) |

**Chip peak bandwidth reference:**

| Chip | Peak BW | 70% Target | 50% Target |
|------|---------|-----------|-----------|
| M1 | 68 GB/s | 48 GB/s | 34 GB/s |
| M2 | 100 GB/s | 70 GB/s | 50 GB/s |
| M3 | 100 GB/s | 70 GB/s | 50 GB/s |
| M4 | 120 GB/s | 84 GB/s | 60 GB/s |
| M4 Pro | 273 GB/s | 191 GB/s | 137 GB/s |
| M4 Max | 546 GB/s | 382 GB/s | 273 GB/s |

## Step 4: Common Bottlenecks and Fixes

### Uncoalesced Memory Access

**Symptom**: Achieved bandwidth well below peak despite correct kernel.

**Diagnosis**: Check if adjacent threads read adjacent memory addresses. Strided access (thread 0 reads `x[0]`, thread 1 reads `x[256]`) wastes bandwidth because the GPU fetches whole cache lines.

```metal
// BAD: Strided access — each thread reads from a different row
float val = (float)x[tid * D + col];  // Threads 0..255 hit 256 different rows

// GOOD: Coalesced access — adjacent threads read adjacent elements
float val = (float)x[row * D + tid];  // Threads 0..255 read consecutive elements
```

**Fix**: Restructure the kernel so that threads within a simdgroup access contiguous memory.

### Threadgroup Size Too Small

**Symptom**: Bandwidth scales up when increasing threadgroup size.

**Diagnosis**: Run a quick sweep:

```python
for tg in [32, 64, 128, 256, 512]:
    bw, _ = measure_bandwidth(lambda: kernel(..., threadgroup=(tg, 1, 1)))
    print(f"tg={tg:4d}: {bw:.1f} GB/s")
```

If bandwidth keeps increasing up to 256 or 512, you were underutilizing the GPU with smaller threadgroups.

**Fix**: Use 256 as the default; autotune with `[32, 64, 128, 256, 512, 1024]`.

### Excessive Threadgroup Memory

**Symptom**: Low occupancy — fewer concurrent threadgroups per GPU core.

**Diagnosis**: Each GPU core has 32KB of threadgroup memory shared across all concurrent threadgroups. Using more per threadgroup means fewer can run simultaneously:

| TG Memory | Concurrent TGs per Core |
|-----------|------------------------|
| ≤ 8 KB | 4 |
| ≤ 16 KB | 2 |
| ≤ 32 KB | 1 |

**Fix**: Reduce tile sizes. Reuse threadgroup memory sequentially with barriers instead of allocating larger arrays.

### Kernel Launch Overhead

**Symptom**: Custom kernel is slow for small tensors but matches or beats built-in for large tensors.

**Diagnosis**: Metal kernel launches have fixed overhead (~5-20 microseconds). For small tensors (< 1000 elements), this dominates.

**Fix**: Don't use custom kernels for small tensors. Guard with a size check:

```python
def my_op(x, w):
    if x.size < 4096:
        return fallback_mlx_op(x, w)
    return _custom_kernel(inputs=[x, w], ...)
```

### Hidden Copy from ensure_row_contiguous

**Symptom**: Kernel is fast on fresh tensors but slow on slices, transposes, or views.

**Diagnosis**: By default, `metal_kernel` copies non-contiguous inputs to row-major layout. This adds an O(N) copy you may not expect.

**Fix**: Either:
1. Ensure inputs are contiguous before calling the kernel (call once, amortize)
2. Set `ensure_row_contiguous=False` and handle strides in the kernel via `x_strides`

### Register Spilling

**Symptom**: Very low bandwidth despite correct coalesced access and optimal threadgroup size.

**Diagnosis**: Too many local `float` variables or arrays in the kernel body can exhaust the register file. The compiler then "spills" registers to threadgroup or device memory, which is much slower.

**Fix**: Reduce per-thread state. Process fewer elements per thread. Use `float` instead of `float4` arrays when possible. Simplify complex expressions.

## Step 5: Xcode Instruments (Advanced)

For deep profiling, Xcode's Metal System Trace can capture GPU counters:

1. Open Instruments → choose "Metal System Trace" template
2. Run your Python script from the terminal (Instruments attaches to the process)
3. Look for: kernel duration, GPU utilization %, memory throughput

**Caveats**:
- Requires macOS with Xcode installed
- Not straightforward to use with Python — MLX submits Metal commands internally
- The command buffer captures may include MLX runtime overhead
- Most useful for comparing two kernel variants rather than absolute numbers

For most optimization work, the bandwidth measurement approach in Step 3 is sufficient without Xcode.

## Quick Reference: Expected Performance

| Operation Type | Expected BW Efficiency | Primary Bottleneck |
|---------------|----------------------|-------------------|
| Element-wise (SiLU, ReLU) | 70-90% | Memory bandwidth |
| Row reduction (RMSNorm, LayerNorm) | 50-80% | Memory + reduction sync |
| Softmax (3-pass) | 40-70% | Memory (3 passes over data) |
| Tiled attention | 30-60% | Compute + memory mixed |
| Quantized matvec | 40-70% | Compute + irregular access |

**Rules of thumb**:
- If your RMSNorm achieves < 40% of peak bandwidth, something is likely wrong
- Element-wise kernels should be within 2x of peak bandwidth
- If a kernel is slower than the equivalent 2-3 pure MLX ops, `mx.compile` may be a better choice
- Fusing two memory-bound kernels into one gives ~2x speedup (halves memory traffic)
