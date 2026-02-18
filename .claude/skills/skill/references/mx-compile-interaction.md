# mx.compile and Custom Metal Kernels

## How mx.compile Works

`mx.compile` traces a Python function containing MLX operations and builds a fused compute graph. On subsequent calls with the same input shapes, it replays the fused graph without re-tracing — avoiding Python overhead and enabling kernel fusion of standard MLX ops.

```python
@mx.compile
def fused_gelu(x):
    # These three ops get fused into one kernel automatically
    return x * 0.5 * (1 + mx.erf(x / 1.4142135))
```

Key behavior:
- First call traces and compiles; subsequent calls replay the compiled graph
- **Recompiles when input shapes change** (not dtype — dtype is fixed at trace time)
- Standard MLX ops (add, multiply, exp, matmul) are fusible within the graph

## Custom Kernels Inside mx.compile

`metal_kernel` calls **are compatible** with `mx.compile`, but they act as **opaque boundaries** in the fused graph. The compiler cannot see inside or fuse across them:

```
[fusible MLX ops] → [opaque metal_kernel] → [fusible MLX ops]
     ↑ fused               ↑ untouched            ↑ fused
```

```python
@mx.compile
def forward(x, weight, eps_buf):
    # These MLX ops before the kernel get fused together
    x = x + 0.1
    x = mx.maximum(x, 0)

    # Custom kernel: opaque boundary, not fused with surrounding ops
    out = _rmsnorm_kernel(
        inputs=[x, weight, eps_buf],
        template=[("T", x.dtype)],
        grid=(x.shape[0] * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

    # These ops after the kernel also get fused together
    return out * 2.0
```

This works correctly. The compile graph contains three segments: fused pre-ops, the kernel call, and fused post-ops.

## Shape Specialization and Recompilation

`mx.compile` recompiles when input shapes change. If your kernel's grid depends on input shape (which it almost always does), each new shape triggers a new compilation:

```python
@mx.compile
def norm(x, weight, eps_buf):
    N = x.shape[0]
    tg = 256
    return _kernel(
        inputs=[x, weight, eps_buf],
        grid=(N * tg, 1, 1),   # Grid changes with N
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

# Each of these triggers a separate compilation:
norm(mx.zeros((1, 4096)), w, eps)    # compiles for N=1
norm(mx.zeros((32, 4096)), w, eps)   # recompiles for N=32
norm(mx.zeros((128, 4096)), w, eps)  # recompiles for N=128
```

### Pad-to-Fixed-Sizes Pattern

If you know the set of possible shapes (e.g., sequence lengths in a transformer), pad to a fixed set of sizes to reduce recompilations:

```python
BUCKET_SIZES = [32, 64, 128, 256, 512, 1024]

def bucket_pad(x, axis=0):
    """Pad first dimension to the next bucket size."""
    n = x.shape[axis]
    target = next(s for s in BUCKET_SIZES if s >= n)
    if target == n:
        return x, n
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target - n)
    return mx.pad(x, pad_width), n

@mx.compile
def norm_bucketed(x, weight, eps_buf):
    N = x.shape[0]
    tg = 256
    return _kernel(
        inputs=[x, weight, eps_buf],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

# Usage: pad, compute, slice back
x_padded, orig_n = bucket_pad(x)
out = norm_bucketed(x_padded, weight, eps_buf)
out = out[:orig_n]
```

## When mx.compile Alone Is Enough

`mx.compile` fuses standard MLX ops into optimized kernels automatically. You often **don't need** a custom `metal_kernel` for:

- **Fusing 2–3 element-wise ops**: `mx.compile` handles this natively
  ```python
  @mx.compile
  def silu(x):
      return x * mx.sigmoid(x)  # Fused into one kernel
  ```
- **Sequential standard ops**: matmul → add → activation chains are fused well
- **Simple reductions**: `mx.sum`, `mx.mean`, `mx.max` along an axis

## When You Need metal_kernel

Write a custom kernel when:

- **Custom reduction logic**: Cross-simdgroup reductions with shared memory (RMSNorm, LayerNorm, custom softmax)
- **Fused ops with shared intermediate state**: Multiple passes that share threadgroup memory (e.g., softmax's max-subtract-exp-sum-normalize)
- **Operations not in MLX**: Novel activation functions, custom quantization, specialized attention patterns
- **Fine-grained memory control**: Tiled algorithms, explicit coalescing, register-level optimizations

## Common Pitfalls

### Redundant Kernel Inside Compile

If your custom kernel already does the full operation, wrapping it in `mx.compile` adds launch overhead without benefit:

```python
# Unnecessary — the kernel is already opaque, nothing to fuse
@mx.compile
def just_rmsnorm(x, w, eps_buf):
    return _rmsnorm_kernel(inputs=[x, w, eps_buf], ...)[0]

# Better — skip mx.compile if the function is just one kernel call
def just_rmsnorm(x, w, eps_buf):
    return _rmsnorm_kernel(inputs=[x, w, eps_buf], ...)[0]

# Useful — mx.compile adds value when there are MLX ops around the kernel
@mx.compile
def block_forward(x, w_norm, w_proj, eps_buf):
    normed = _rmsnorm_kernel(inputs=[x, w_norm, eps_buf], ...)[0]
    return normed @ w_proj  # This matmul benefits from compile
```

### Shape-Dependent Grid Causing Excessive Recompilation

If your model sees many different sequence lengths:

```python
# Problem: new compilation for every unique sequence length
@mx.compile
def norm(x, w, eps_buf):
    return _kernel(inputs=[x, w, eps_buf], grid=(x.shape[0] * 256, 1, 1), ...)
```

Solutions:
1. Use the pad-to-bucket pattern shown above
2. Don't use `mx.compile` for functions that are mostly custom kernels
3. Accept recompilation if shapes are stable (e.g., fixed batch size)

### Expecting Compile to Optimize Kernel Internals

`mx.compile` cannot optimize the Metal source inside your `metal_kernel`. It only optimizes the graph of MLX operations around it. If your kernel is slow, you need to optimize the Metal source itself — `mx.compile` won't help.
