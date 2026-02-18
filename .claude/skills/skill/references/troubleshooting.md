# Troubleshooting MLX Metal Kernels

## Metal Compilation Errors

### "Failed to compile Metal kernel"

**Cause**: Syntax error in the Metal source string.

**Common mistakes:**

```python
# BAD: Missing semicolon
source = "out[i] = x[i] * 2.0f"

# GOOD
source = "out[i] = x[i] * 2.0f;"
```

```python
# BAD: Using C++ features not available in Metal
source = "auto val = x[i];"  # 'auto' not always supported

# GOOD: Use explicit types
source = "float val = (float)x[i];"
```

```python
# BAD: Using CUDA-style functions
source = "__shfl_xor_sync(0xffffffff, val, offset);"

# GOOD: Use Metal simdgroup functions
source = "simd_shuffle_xor(val, offset);"
```

### "Use of undeclared identifier"

**Cause**: Referencing a variable not in `input_names`, `output_names`, or Metal built-ins.

```python
# BAD: 'scale' is not declared anywhere
kernel = mx.fast.metal_kernel(
    name="bad",
    input_names=["x"],
    output_names=["out"],
    source="out[i] = x[i] * scale;",  # Error: 'scale' undeclared
)

# FIX Option 1: Add as input
kernel = mx.fast.metal_kernel(
    name="fix1",
    input_names=["x", "scale"],  # Pass scale as a 1-element array
    output_names=["out"],
    source="out[i] = x[i] * scale[0];",
)

# FIX Option 2: Use header for constants
kernel = mx.fast.metal_kernel(
    name="fix2",
    input_names=["x"],
    output_names=["out"],
    source="out[i] = x[i] * SCALE;",
    header="constant float SCALE = 2.0f;",
)
```

### "Invalid type" or template errors

**Cause**: Template type `T` not matching the data.

```python
# BAD: bfloat16 on M1/M2 (not supported)
kernel(inputs=[bf16_array], template=[("T", mx.bfloat16)], ...)

# FIX: Check device support
if mx.metal.is_available():
    # bfloat16 requires M3+ (Apple GPU family 9)
    # Fall back to float16 on older chips
    dtype = mx.bfloat16 if supports_bf16() else mx.float16
```

### "member reference base type 'uint' is not a structure or union"

**Cause**: Accessing `.x` on a scalar Metal built-in. Some Metal attributes are `uint` (scalar), not `uint3`:

```python
# BAD: thread_index_in_threadgroup is uint, not uint3
source = "uint tid = thread_index_in_threadgroup.x;"  # Error!

# GOOD: Use directly (it's already a scalar)
source = "uint tid = thread_index_in_threadgroup;"
```

**Scalar types (uint):** `thread_index_in_threadgroup`, `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup`

**Vector types (uint3, use .x/.y/.z):** `thread_position_in_grid`, `threadgroup_position_in_grid`, `threads_per_threadgroup`, `threads_per_grid`

## Grid and Threadgroup Errors

### "Grid size must be > 0"

```python
# BAD: Empty input
x = mx.array([])
kernel(inputs=[x], grid=(0, 1, 1), ...)  # Error

# FIX: Guard against empty inputs
N = x.size
if N == 0:
    return mx.array([], dtype=x.dtype)
```

### Reduction kernel produces wrong results (grid too small)

**Cause**: `grid` is **total threads**, not number of threadgroups. For reduction kernels that use `threadgroup_position_in_grid.x` as the row index, you need `grid = rows * threadgroup_size`.

```python
# BAD: grid=(N, 1, 1) with tg=256 → only 1 threadgroup for N < 256
# All threads think they're on row 0!
kernel(..., grid=(N, 1, 1), threadgroup=(256, 1, 1), ...)

# GOOD: One threadgroup per row
kernel(..., grid=(N * 256, 1, 1), threadgroup=(256, 1, 1), ...)
```

**Rule**: If your kernel uses `threadgroup_position_in_grid` as a work-item index (e.g., row), multiply grid by threadgroup size. If it uses `thread_position_in_grid`, grid = total elements.

### Threadgroup size exceeds maximum

```python
# BAD: Threadgroup > 1024 (max on all Apple Silicon)
kernel(..., threadgroup=(2048, 1, 1), ...)

# FIX: Clamp to max
tg = min(desired_tg, 1024)
```

### Wrong output shape / size

```python
# BAD: Grid doesn't match output shape
kernel(
    inputs=[x],
    grid=(100, 1, 1),
    output_shapes=[(200,)],  # Mismatch! Only 100 threads write
    ...
)

# The extra elements in out[100:200] will be uninitialized garbage

# FIX: Match grid to output, or use bounds checking in source
source = """
uint i = thread_position_in_grid.x;
if (i < out_shape[0]) {
    out[i] = x[i] * 2.0f;
}
"""
```

## Race Conditions

### Multiple threads writing to the same index

```python
# BAD: Multiple threads write to grad[target] without atomics
source = "grad[indices[i]] += values[i];"  # Race condition!

# FIX: Use atomic operations
kernel = mx.fast.metal_kernel(
    ...,
    source="""
    atomic_fetch_add_explicit(
        (device atomic<float>*)&grad[indices[i]],
        (float)values[i],
        memory_order_relaxed
    );
    """,
    atomic_outputs=True,
)
```

## Performance Issues

### Kernel is slower than MLX built-in

**Common causes:**

1. **Kernel too small**: Launch overhead dominates for small tensors (< 1000 elements). Use MLX native ops for small inputs.

2. **Not reusing compiled kernel**: Calling `mx.fast.metal_kernel()` repeatedly instead of caching the object.

3. **Unnecessary `ensure_row_contiguous` copy**: If your input is already contiguous, this is free. But if you're calling a kernel on a slice or transpose, the copy adds overhead.

4. **Threadgroup too small**: Single simdgroup (32 threads) when 256 would keep the GPU busier.

### Benchmark doesn't show improvement

```python
# BAD: Not forcing evaluation
for _ in range(100):
    out = kernel(inputs=[x], ...)
# All 100 calls are lazily queued but not timed

# GOOD: Force evaluation each iteration
for _ in range(100):
    out = kernel(inputs=[x], ...)
    mx.eval(out)  # Forces synchronous execution
```

```python
# EVEN BETTER: Use mx.synchronize() around the timing loop
mx.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    out = kernel(inputs=[x], ...)
    mx.eval(out)
mx.synchronize()
elapsed = time.perf_counter() - t0
```

### Memory bandwidth calculation

To check if your kernel is achieving good bandwidth utilization:

```python
bytes_accessed = input_bytes + output_bytes  # Total bytes read + written
bandwidth_gbps = bytes_accessed / elapsed_seconds / 1e9

# Compare against your chip's theoretical max
# M1: 68 GB/s, M2: 100 GB/s, M3: 100 GB/s, M4: 120-546 GB/s
peak_bw = 120  # M4 base, adjust for your chip
efficiency = bandwidth_gbps / peak_bw * 100
print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s ({efficiency:.1f}% of peak)")
```

## Debugging Techniques

### Use `verbose=True` to inspect generated Metal source

Pass `verbose=True` when calling a compiled kernel to print the fully generated Metal source code, including the auto-generated function signature and all injected parameters:

```python
out = kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(N,)],
    output_dtypes=[mx.float32],
    verbose=True,  # Prints the complete Metal source
)
```

This is invaluable for debugging type mismatches, verifying that `input_names`/`output_names` generated the expected function parameters, and understanding what auto-injected variables (shapes, strides, ndim) are available.

### Print intermediate values via output buffers

Metal kernels can't print. Instead, write debug values to an extra output:

```python
kernel = mx.fast.metal_kernel(
    name="debug_kernel",
    input_names=["x"],
    output_names=["out", "debug"],
    source="""
    uint i = thread_position_in_grid.x;
    float val = (float)x[i];
    float result = metal::exp(val);
    out[i] = (T)result;
    debug[i] = val;  // Capture intermediate value
    """,
)
out, debug = kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(N,), (N,)],
    output_dtypes=[mx.float32, mx.float32],
)
mx.eval(out, debug)
print("Debug values:", debug[:10])
```

### Correctness validation

Always compare against a reference implementation:

```python
# Reference
expected = mx.fast.rms_norm(x, weight, eps=1e-5)

# Custom
actual = my_rmsnorm(x, weight, eps=1e-5)

mx.eval(expected, actual)

max_diff = mx.max(mx.abs(expected - actual)).item()
print(f"Max absolute difference: {max_diff:.2e}")
assert mx.allclose(expected, actual, atol=1e-5), f"Mismatch: max diff {max_diff}"
```

## Lazy Evaluation and Error Surfacing

Metal kernel errors in MLX are **deferred** — they surface at `mx.eval()` time, not when you call the kernel. This makes debugging confusing because the traceback points to `eval`, not the broken kernel.

### The Confusing Traceback

```python
# Line 10: kernel with a bug (wrong variable name)
out = my_kernel(inputs=[x], ...)

# Line 20: some other work
y = mx.add(out, 1.0)

# Line 30: error surfaces HERE, not at line 10
mx.eval(y)
# RuntimeError: ... Failed to compile Metal kernel ...
```

The traceback will point to line 30 (`mx.eval`), but the actual bug is on line 10.

### Debugging Pattern: Eval After Each Kernel

During development, force evaluation immediately after each kernel call to pinpoint errors:

```python
# Debug mode: eval after each kernel to get accurate error locations
out = my_kernel(inputs=[x], ...)
mx.eval(out)  # If the kernel has a bug, error surfaces HERE

out2 = another_kernel(inputs=[out], ...)
mx.eval(out2)  # Isolates errors to the specific kernel
```

Once your kernels are correct, remove the intermediate `mx.eval()` calls — they prevent MLX from fusing operations across kernel boundaries.

### `verbose=True` Prints at Creation Time

The `verbose=True` flag prints the generated Metal source when the kernel is **called** (queued), not when `mx.eval()` runs it. This is helpful: you can see the source even before the error appears.

```python
# verbose=True prints source immediately, before eval
out = kernel(inputs=[x], ..., verbose=True)
# ^^^ Metal source printed here

mx.eval(out)
# ^^^ Compilation error surfaces here (if any)
```

Use `verbose=True` to inspect the generated function signature and verify your `input_names`, `output_names`, and template types are correct.

### Common Symptom: "Works with small inputs, fails with large"

If a kernel works for small tensors but produces wrong results (not errors) for large ones, the kernel compiled fine but has a **logic bug** — typically a grid/threadgroup sizing issue. Since there's no compilation error, `mx.eval()` succeeds but the output is wrong. Always validate against a reference implementation across multiple sizes.

## Common Pitfalls

### Float16 precision loss

```python
# BAD: Accumulating in float16
source = "half sum = 0.0h; for (...) sum += x[i]; ..."

# GOOD: Accumulate in float32, cast at the end
source = "float sum = 0.0f; for (...) sum += (float)x[i]; ... out[i] = (T)sum;"
```

### Integer overflow in index computation

```python
# BAD: uint overflow for negative values
source = "uint idx = row * D - 1;"  # Underflow if row*D == 0!

# GOOD: Use int for indices that can be negative
source = "int idx = (int)row * (int)D - 1;"
```

### Forgetting bounds checking

```python
# BAD: No bounds check → out-of-bounds write
source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x];"
# Crashes if grid > array size

# GOOD: Check bounds
source = """
uint i = thread_position_in_grid.x;
if (i < out_shape[0]) {
    out[i] = x[i];
}
"""
```

### Python format strings vs Metal braces

When using Python `.format()` or f-strings with Metal source containing braces:

```python
# BAD: Metal braces conflict with Python format
source = f"if (x > 0) {{ out[i] = x; }}"  # Need double braces

# GOOD: Use .format() with double braces
source = """
if (x > 0) {{
    out[i] = ({T})x;
}}
""".format(T="float")

# OR: Use template parameter instead of format strings
kernel = mx.fast.metal_kernel(..., source="""
if (x > 0) {
    out[i] = (T)x;  // T comes from template=[("T", dtype)]
}
""")
```
