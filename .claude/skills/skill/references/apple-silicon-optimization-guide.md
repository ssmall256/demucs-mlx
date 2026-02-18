# Apple Silicon GPU Optimization Guide

## Architecture Overview

Apple Silicon GPUs use a **tile-based deferred rendering** (TBDR) architecture with unified memory. Unlike NVIDIA GPUs, there is no separate VRAM — the CPU and GPU share the same physical memory pool with coherent caches. This means:

- **No host↔device copies**: Tensors are already in GPU-accessible memory
- **Memory bandwidth is the bottleneck** for most ML kernels (not compute)
- **Kernel launch overhead is low** compared to CUDA (no PCIe latency)

## Chip Specifications

### Base Chips

| Spec | M1 | M2 | M3 | M4 |
|------|----|----|----|----|
| GPU Cores | 7–8 | 8–10 | 10 | 10 |
| Memory BW | 68 GB/s | 100 GB/s | 100 GB/s | 120 GB/s |
| SIMD Width | 32 | 32 | 32 | 32 |
| Max Threadgroup | 1024 | 1024 | 1024 | 1024 |
| BFloat16 | No | No | Yes | Yes |
| GPU Family | Apple 7 | Apple 8 | Apple 9 | Apple 9+ |

### Pro/Max/Ultra Chips

| Spec | M1 Pro | M1 Max | M2 Max | M3 Max | M4 Pro | M4 Max |
|------|--------|--------|--------|--------|--------|--------|
| GPU Cores | 14–16 | 24–32 | 30–38 | 40 | 20 | 40 |
| Memory BW | 200 GB/s | 400 GB/s | 400 GB/s | 400 GB/s | 273 GB/s | 546 GB/s |

## Thread Execution Model

### SIMD Groups (Warps)

Apple GPUs execute threads in **simdgroups** of 32 threads — analogous to NVIDIA warps. All 32 threads in a simdgroup execute the same instruction in lockstep.

```metal
// Get your position within the execution hierarchy
uint gid    = thread_position_in_grid.x;          // Global thread ID
uint tgid   = threadgroup_position_in_grid.x;     // Which threadgroup
uint tid    = thread_index_in_threadgroup;           // Thread within threadgroup (scalar uint)
uint sid    = thread_index_in_simdgroup;           // Lane within simdgroup (0–31)
uint sgid   = simdgroup_index_in_threadgroup;      // Which simdgroup in threadgroup
```

### Threadgroup Sizing Guidelines

| Kernel Type | Recommended Size | Rationale |
|-------------|-----------------|-----------|
| Element-wise | 256–512 | High occupancy, simple ops |
| Row reduction (D ≤ 32) | 32 | Single simdgroup covers dimension |
| Row reduction (D ≤ 1024) | 256 | 8 simdgroups with cross-group reduction |
| Row reduction (D > 1024) | 256–512 | Loop within threads |
| Matrix multiply | 64–128 | Tile-based, register pressure |

**Rules of thumb:**
- Always use multiples of 32 (SIMD width)
- Start with 256, benchmark alternatives
- Larger threadgroups = more register pressure = fewer concurrent threadgroups
- Smaller threadgroups = less intra-group parallelism for reductions

### Autotuning Threadgroup Size

When performance matters, benchmark across candidates:

```python
import time
import mlx.core as mx

def autotune_threadgroup(kernel, inputs, grid, output_shapes, output_dtypes,
                         candidates=(32, 64, 128, 256, 512, 1024),
                         warmup=3, iters=10):
    best_tgx, best_time = 256, float("inf")
    for tgx in candidates:
        if tgx > grid[0]:
            continue
        # Warmup
        for _ in range(warmup):
            out = kernel(inputs=inputs, grid=grid, threadgroup=(tgx, 1, 1),
                         output_shapes=output_shapes, output_dtypes=output_dtypes)
            mx.eval(out)
        # Timed
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = kernel(inputs=inputs, grid=grid, threadgroup=(tgx, 1, 1),
                         output_shapes=output_shapes, output_dtypes=output_dtypes)
            mx.eval(out)
        mx.synchronize()
        dt = (time.perf_counter() - t0) / iters
        if dt < best_time:
            best_time, best_tgx = dt, tgx
    return best_tgx
```

## SIMD-Group Operations

These are the most important optimization primitives on Apple GPUs. They allow communication between threads within a simdgroup **without shared memory**.

### Reductions

```metal
float val = compute_something();
float sum = simd_sum(val);    // Sum across all 32 lanes → broadcast to all
float mx  = simd_max(val);    // Max across all 32 lanes → broadcast to all
float mn  = simd_min(val);    // Min across all 32 lanes → broadcast to all
```

### Shuffle

```metal
float val = my_value;
float from_lane_5 = simd_shuffle(val, 5);          // Read lane 5's value
float from_next   = simd_shuffle_down(val, 1);      // Read next lane's value
float from_prev   = simd_shuffle_up(val, 1);        // Read previous lane's value
float xor_partner = simd_shuffle_xor(val, 1);       // XOR-based butterfly
```

### Cross-Simdgroup Reduction (via threadgroup memory)

When your reduction spans multiple simdgroups, use threadgroup memory:

```metal
threadgroup float shared[32];  // One slot per simdgroup

float local_sum = simd_sum(my_val);

// First lane of each simdgroup writes to shared
if (thread_index_in_simdgroup == 0) {
    shared[simdgroup_index_in_threadgroup] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// First simdgroup reduces across all partial sums
if (simdgroup_index_in_threadgroup == 0) {
    float partial = (thread_index_in_simdgroup < num_simdgroups)
                    ? shared[thread_index_in_simdgroup] : 0.0f;
    float total = simd_sum(partial);
    // 'total' now holds the full reduction result
}
```

## Memory Access Patterns

### Coalesced Access

Adjacent threads should access adjacent memory locations:

```metal
// GOOD: Coalesced — thread i reads element i
out[gid] = x[gid] * scale;

// BAD: Strided — thread i reads element i * stride
out[gid] = x[gid * stride] * scale;
```

### Vectorized Loads (Advanced)

For large element-wise operations, load multiple values per thread:

```metal
// Load 4 floats at once using float4
uint base = thread_position_in_grid.x * 4;
float4 vals = *((const device float4*)(x + base));
vals = vals * scale;
*((device float4*)(out + base)) = vals;
```

Adjust grid size accordingly: `grid = (N // 4, 1, 1)`.

## Arithmetic Intensity Analysis

Most ML kernels on Apple Silicon are **memory-bandwidth bound**. Use this to estimate:

```
Arithmetic Intensity = FLOPs / Bytes Accessed

If AI < device_flops / device_bandwidth → memory bound
```

| Operation | FLOPs/element | Bytes/element | AI | Bound |
|-----------|---------------|---------------|-----|-------|
| RMSNorm | ~5 | 8 (read+write) | 0.6 | Memory |
| Softmax | ~5 | 8 | 0.6 | Memory |
| SiLU | 3 | 8 | 0.4 | Memory |
| GEMM (large) | 2N | 4 | N/2 | Compute |

For memory-bound kernels, the goal is to maximize memory throughput:
- Coalesced access patterns
- Minimize redundant reads (fuse operations)
- Use vectorized loads where possible

## Threadgroup Memory

Threadgroup memory (Metal's equivalent of CUDA shared memory) is fast on-chip SRAM shared by all threads in a threadgroup. It is essential for cross-simdgroup communication (reductions, tiling).

### Limits

All Apple Silicon chips provide **32KB (32768 bytes)** of threadgroup memory per threadgroup.

| Data Type | Max Elements in 32KB |
|-----------|---------------------|
| `float` (4 bytes) | 8192 |
| `half` / `float16` (2 bytes) | 16384 |
| `int` (4 bytes) | 8192 |
| `uint8_t` (1 byte) | 32768 |

### Occupancy Impact

More threadgroup memory per threadgroup = fewer threadgroups can run concurrently on a GPU core. This reduces occupancy and can hurt performance.

| Threadgroup Memory Used | Approx. Concurrent TGs per Core |
|------------------------|-------------------------------|
| ≤ 4KB | High (limited by registers/threads) |
| 8KB | Moderate |
| 16KB | Reduced |
| 32KB (max) | 1 threadgroup per core |

**Rule of thumb**: Keep threadgroup memory usage under 16KB unless the algorithm demands more.

### Tile Size Examples

For tiled algorithms (matrix multiply, attention), threadgroup memory holds tiles of input data:

| Tile Config | float32 | float16 | Fits in 32KB? |
|------------|---------|---------|---------------|
| 32×32 tile | 4KB | 2KB | Yes |
| 32×64 tile | 8KB | 4KB | Yes |
| 64×64 tile | 16KB | 8KB | Yes (float16 only for 2 tiles) |
| 32×128 tile | 16KB | 8KB | Yes |
| 64×128 tile | 32KB | 16KB | Tight (float32) / Yes (float16) |

### Sequential Reuse with Barriers

When you need more data than fits in threadgroup memory at once, load tiles sequentially with barriers between phases:

```metal
threadgroup float tile[4096];  // 16KB — one tile at a time

// Phase 1: Load and process tile A
for (uint i = tid; i < 4096; i += tg_size) {
    tile[i] = (float)a[offset_a + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... read from tile ...

threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 2: Reuse same memory for tile B
for (uint i = tid; i < 4096; i += tg_size) {
    tile[i] = (float)b[offset_b + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... read from tile ...
```

**Critical**: Always place `threadgroup_barrier(mem_flags::mem_threadgroup)` between writing to and reading from threadgroup memory, and between reuse phases. Missing barriers cause data races that produce intermittent wrong results.

## Metal Standard Library Functions

Commonly used functions available in kernel source:

```metal
metal::exp(x)       metal::log(x)       metal::sqrt(x)
metal::rsqrt(x)     metal::abs(x)       metal::max(a, b)
metal::min(a, b)    metal::clamp(x,a,b) metal::tanh(x)
metal::fast::exp(x) // Faster but less precise
```

## MLX-Specific Utilities

MLX auto-includes `mlx/backend/metal/kernels/utils.h` which provides:

```metal
// Convert flat index to strided offset (for non-contiguous arrays)
size_t elem_to_loc(uint elem, const device int* shape,
                   const device size_t* strides, int ndim);
```

Use this when `ensure_row_contiguous=False` and inputs may have non-trivial strides.
