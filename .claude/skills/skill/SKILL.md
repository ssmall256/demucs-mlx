---
name: mlx-metal-kernels
description: "Provides guidance for writing custom Metal compute kernels using MLX's mx.fast.metal_kernel() API for Apple Silicon GPUs (M1, M2, M3, M4). Covers kernel patterns for RMSNorm, LayerNorm, softmax, attention variants (causal, sliding window, GQA), and reductions. Includes profiling, debugging, simdgroup_matrix MMA (M3+), and integration with mlx-lm."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: rmsnorm, layernorm, softmax, rope, activations, reduction, benchmark, attention, attention-variants, causal-attention, gqa, multihead, quantized, cuda-porting, debugging, production, profiling, simdgroup-matrix"
---

# MLX Metal Kernels for Apple Silicon

## Routing Guide (Pick the Right Template)

- **Elementwise / pointwise ops** → `scripts/batch_elementwise_kernel.py`
- **Row-wise reductions (RMSNorm / LayerNorm / Softmax)** → `scripts/rmsnorm_kernel.py`, `scripts/layernorm_kernel.py`, `scripts/softmax_kernel.py`
- **Attention** → Prefer MLX built-ins first (`mx.fast.scaled_dot_product_attention`) and use `references/attention-variants-guide.md` only when you need a custom layout/mask.
- **Quantized matvec / dequant patterns** → `scripts/dequant_matvec_kernel.py`
- **M3+ matrix ops (simdgroup_matrix)** → `scripts/simdgroup_matmul_kernel.py` + `references/simdgroup-matrix-guide.md`

## Pre-Benchmark Checklist (Avoid “bench lies”)

1. **Force evaluation**: time with `mx.eval(out); mx.synchronize()` (MLX is lazy).
2. **Contiguity**: if your kernel does `x[row * D + i]`, require contiguous inputs (or call `mx.ascontiguousarray`).
3. **Bounds checks**: if you round `grid` up, guard `if (tid >= ...) return;`.
4. **Dtype expectations**: float16 I/O is common; accumulate in float32 for stability.
5. **Threadgroup limits**: `threadgroup.x` must be ≤ 1024 and a multiple of 32 (one simdgroup).
6. **First-call compile**: ignore the first run when benchmarking (compile + cache effects).

This skill provides patterns and guidance for developing custom Metal compute kernels using MLX's `mx.fast.metal_kernel()` API, targeting Apple Silicon GPUs (M1, M2, M3, M4).

## Quick Start

```python
import mlx.core as mx

kernel = mx.fast.metal_kernel(
    name="my_relu",
    input_names=["x"],
    output_names=["out"],
    source="uint i = thread_position_in_grid.x; out[i] = max(x[i], T(0));",
)
x = mx.random.normal((1024,))
out = kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(1024, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(1024,)],
    output_dtypes=[mx.float32],
)[0]
mx.eval(out)
```

**For working kernel implementations:**
```bash
python scripts/rmsnorm_kernel.py    # RMSNorm with simdgroup reduction
python scripts/softmax_kernel.py    # Numerically stable softmax
python scripts/layernorm_kernel.py  # LayerNorm with two-pass reduction
```

**For benchmarking:**
```bash
python scripts/benchmark_rmsnorm.py  # Compare custom vs mx.fast.rms_norm
```

## When This Skill Applies

- You are targeting **Apple Silicon** (M1, M2, M3, M4) GPUs
- You are using the **MLX framework** (`import mlx.core as mx`)
- You need a **custom operation** not available in `mx.fast.*` or `mx.nn.*`
- You want to **fuse multiple ops** into a single kernel launch
- You want to **optimize a bottleneck** in an MLX inference or training pipeline

## When NOT to Write Custom Kernels

- **For fusing 2–3 standard ops**: Try `mx.compile` first — it fuses standard MLX operations automatically and is simpler to maintain.
- **For small tensors** (< 1000 elements): Kernel launch overhead dominates. Use native MLX ops.
- **When an MLX built-in exists**: `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention` are already highly optimized.
- **For prototyping**: Get correctness first with pure MLX ops, then optimize hot paths with custom kernels.

## `mx.fast.metal_kernel()` API Reference

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Kernel identifier (used for JIT cache) |
| `input_names` | `list[str]` | Names of input buffers; generates `const device T* name` |
| `output_names` | `list[str]` | Names of output buffers; generates `device T* name` |
| `source` | `str` | Metal kernel body (function signature is auto-generated) |
| `header` | `str` | Helper functions/includes placed before the kernel function |
| `ensure_row_contiguous` | `bool` | Copy non-contiguous inputs to row-major (default: `True`) |
| `atomic_outputs` | `bool` | Use `device atomic<T>*` for outputs (default: `False`) |

**Calling the compiled kernel:**

```python
outputs = kernel(
    inputs=[array1, array2],          # MLX arrays matching input_names
    template=[("T", mx.float32)],     # Type substitutions
    grid=(total_x, total_y, total_z), # Total threads
    threadgroup=(tg_x, tg_y, tg_z),  # Threads per threadgroup
    output_shapes=[(shape1,), ...],   # Output tensor shapes
    output_dtypes=[mx.float32, ...],  # Output tensor dtypes
    init_value=0,                     # Optional: initialize outputs
    verbose=True,                     # Optional: print generated Metal source
)
```

**Auto-injected parameters** (available in `source` if referenced):
- `name_shape` (`const int*`) — dimension sizes
- `name_strides` (`const size_t*`) — memory strides
- `name_ndim` (`const int&`) — number of dimensions
- `thread_position_in_grid`, `threadgroup_position_in_grid`, `threads_per_grid`
- `thread_index_in_threadgroup`, `threads_per_threadgroup`
- `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup`

## Apple Silicon GPU Reference

| Chip | Memory BW | GPU Cores | SIMD Width | Max Threadgroup | Notes |
|------|-----------|-----------|------------|-----------------|-------|
| M1 | 68 GB/s | 7–8 | 32 | 1024 | First Apple Silicon |
| M2 | 100 GB/s | 8–10 | 32 | 1024 | +GPU core option |
| M3 | 100 GB/s | 10 | 32 | 1024 | +bfloat16, dynamic caching |
| M4 | 120 GB/s | 10 | 32 | 1024 | Improved ray tracing |
| M4 Pro | 273 GB/s | 20 | 32 | 1024 | 2x memory bus |
| M4 Max | 546 GB/s | 40 | 32 | 1024 | 4x memory bus |

**Key architecture facts:**
- Unified memory: CPU and GPU share the same address space (no host↔device copies)
- SIMD width is always 32 threads — threadgroup sizes should be multiples of 32
- `bfloat16` requires M3 or later (Apple GPU family 9+)

## Core Kernel Patterns

### Simdgroup Reduction (for RMSNorm, LayerNorm, Softmax)
```metal
// Sum across a 32-thread simdgroup — no shared memory needed
float local = compute_value(x[idx]);
float total = simd_sum(local);  // broadcasts result to all lanes
```

### Type-Polymorphic Templates
```python
source = "out[i] = ({T})(float(x[i]) * float(x[i]));"
kernel = mx.fast.metal_kernel(name="sq", input_names=["x"], output_names=["out"], source=source)
result = kernel(inputs=[x], template=[("T", mx.float16)], ...)
```

**Note:** Template parameters also support `int` and `bool` values for compile-time constants (e.g., block sizes, feature flags), not just `mx.Dtype`.

### Float32 Accumulation (for numerical stability)
```metal
// Always accumulate in float32, cast output at the end
float acc = 0.0f;
for (uint i = sid; i < D; i += 32) {
    acc += (float)x[row * D + i] * (float)x[row * D + i];
}
acc = simd_sum(acc);
float rms = metal::rsqrt(acc / float(D) + eps);
out[row * D + sid] = (T)((float)x[row * D + sid] * rms * (float)w[sid]);
```

### Grid and Threadgroup Sizing

**CRITICAL**: `grid` is the **total thread count**, not the number of threadgroups. Metal creates `ceil(grid / threadgroup)` threadgroups.

- **Element-wise** (using `thread_position_in_grid.x`): `grid=(N, 1, 1)`
- **Reduction** (using `threadgroup_position_in_grid.x` as row index): `grid=(N * tg, 1, 1)` — one threadgroup per row

```python
# Element-wise: grid = total elements
kernel(..., grid=(N, 1, 1), threadgroup=(256, 1, 1), ...)

# Row reduction: grid = rows × threadgroup_size
kernel(..., grid=(num_rows * tg, 1, 1), threadgroup=(tg, 1, 1), ...)
```

**Threadgroup size recommendations:**
- Default: **256** (8 simdgroups × 32 threads)
- For reductions over dimension D ≤ 32: use **32** (single simdgroup)
- For large element-wise ops: try **512** or **1024**
- **Autotune** by benchmarking candidates `[32, 64, 128, 256, 512, 1024]`

**Metal built-in types:**
- `thread_position_in_grid` — `uint3` (use `.x`, `.y`, `.z`)
- `threadgroup_position_in_grid` — `uint3`
- `threads_per_threadgroup` — `uint3`
- `thread_index_in_threadgroup` — `uint` (scalar, NOT uint3)
- `thread_index_in_simdgroup` — `uint` (scalar)
- `simdgroup_index_in_threadgroup` — `uint` (scalar)

## Data Type Support

| MLX dtype | Metal type | Notes |
|-----------|------------|-------|
| `mx.float32` | `float` | Always supported, use for accumulation |
| `mx.float16` | `half` | Supported on all Apple Silicon |
| `mx.bfloat16` | `bfloat` | M3+ only (Apple GPU family 9+) |
| `mx.int32` | `int` | For indices and parameters |
| `mx.bool_` | `bool` | For masks |

## Project Structure

```
my_kernel_project/
├── kernels/
│   ├── rmsnorm.py       # Kernel definition + Python wrapper
│   └── softmax.py       # Another kernel
├── benchmark.py          # Micro-benchmark script
├── test_kernels.py       # Correctness tests (allclose vs reference)
└── README.md
```

## Reference Documents

- `references/apple-silicon-optimization-guide.md` — GPU architecture deep dive, threadgroup tuning, SIMD operations, threadgroup memory
- `references/kernel-templates.md` — Copy-paste templates for element-wise, reduction, softmax, fused, and attention kernels
- `references/mlx-integration.md` — Integration with mlx-lm models, kernel caching, autotuning, end-to-end example
- `references/troubleshooting.md` — Common Metal compilation errors, lazy eval debugging, performance pitfalls
- `references/attention-kernel-guide.md` — Tiled attention with online softmax, memory budget analysis
- `references/cuda-to-metal-guide.md` — CUDA-to-Metal porting cheat sheet (thread indexing, memory, sync, math)
- `references/multi-dim-grid-patterns.md` — 2D/3D grid dispatch for batched operations
- `references/quantized-kernel-patterns.md` — 4-bit weight dequantization, fused dequant-matvec
- `references/testing-patterns.md` — Edge-case testing categories, reusable test runner, numerical extremes
- `references/production-error-handling.md` — Device checks, input validation, graceful fallback, error diagnostics
- `references/mx-compile-interaction.md` — How mx.compile interacts with metal_kernel, shape specialization, recompilation
- `references/kernel-debugging-walkthrough.md` — Step-by-step debugging narrative from wrong output to correct kernel
- `references/attention-variants-guide.md` — Causal, sliding window, and grouped-query attention patterns
- `references/profiling-guide.md` — Diagnostic workflow and decision tree for kernel performance
- `references/simdgroup-matrix-guide.md` — Hardware 8x8 MMA operations (M3+ only), tiled GEMM

## Example Scripts

- `scripts/rmsnorm_kernel.py` — RMSNorm with simdgroup reduction, correctness validation
- `scripts/softmax_kernel.py` — Numerically stable softmax, comparison with `mx.softmax()`
- `scripts/benchmark_rmsnorm.py` — Multi-shape benchmark with timing table
- `scripts/attention_kernel.py` — Single-head tiled attention with online softmax
- `scripts/batch_elementwise_kernel.py` — Batched SiLU using 2D grid dispatch
- `scripts/dequant_matvec_kernel.py` — 4-bit dequantized matrix-vector multiply
- `scripts/e2e_custom_kernels.py` — End-to-end: patch mlx-lm model with custom kernels
- `scripts/layernorm_kernel.py` — LayerNorm with two-pass reduction (mean + variance)
- `scripts/multihead_rope_kernel.py` — Multi-head RoPE with 3D grid and 4D tensor patterns
- `scripts/attention_variants_kernel.py` — Causal, sliding window, and GQA attention kernels
- `scripts/simdgroup_matmul_kernel.py` — Matrix multiply using simdgroup_matrix MMA (M3+)

## Known Quirks

- Some MLX versions print "mx.metal.device_info is deprecated…" even when not called directly. Safe to ignore.
