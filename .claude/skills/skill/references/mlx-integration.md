# MLX Model Integration Guide

## Replacing Operations in mlx-lm Models

`mlx-lm` provides implementations of popular LLMs (Llama, Mistral, Qwen, Phi, Gemma) as `mlx.nn.Module` subclasses. To inject custom kernels, you replace specific modules or forward methods.

### Finding Target Modules

```python
import mlx.nn as nn
from mlx_lm import load

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# List all module types
for name, module in model.named_modules():
    class_name = type(module).__name__
    if "Norm" in class_name or "RMSNorm" in class_name:
        print(f"{name}: {class_name}")
```

Typical output for Llama:
```
model.norm: RMSNorm
model.layers.0.input_layernorm: RMSNorm
model.layers.0.post_attention_layernorm: RMSNorm
...
```

### Pattern: Replace nn.RMSNorm with Custom Kernel

```python
import mlx.core as mx
import mlx.nn as nn

RMSNORM_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

float sum_sq = 0.0f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float val = (float)x[row * D + i];
    sum_sq += val * val;
}
sum_sq = simd_sum(sum_sq);
threadgroup float shared[32];
if (lane == 0) shared[sg] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float p = (lane < (threads_per_threadgroup.x + 31) / 32)
              ? shared[lane] : 0.0f;
    shared[0] = simd_sum(p);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float rms = metal::rsqrt(shared[0] / float(D) + eps_val);

for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    out[row * D + i] = (T)((float)x[row * D + i] * rms * (float)w[i]);
}
"""

_rmsnorm_kernel = mx.fast.metal_kernel(
    name="custom_rmsnorm",
    input_names=["x", "w", "eps"],
    output_names=["out"],
    source=RMSNORM_SOURCE,
)


class CustomRMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm using a custom Metal kernel."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims
        self._eps_arr = mx.array([eps], dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)
        N = x_2d.shape[0]
        tg = min(256, D)
        tg = max(32, (tg // 32) * 32)
        out = _rmsnorm_kernel(
            inputs=[x_2d, self.weight, self._eps_arr],
            template=[("T", x.dtype)],
            grid=(N * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(N, D)],
            output_dtypes=[x.dtype],
        )[0]
        return out.reshape(orig_shape)


def patch_rmsnorm(model: nn.Module) -> int:
    """Replace all nn.RMSNorm modules with CustomRMSNorm."""
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "RMSNorm":
            dims = module.weight.shape[0]
            eps = getattr(module, "eps", 1e-5)
            replacement = CustomRMSNorm(dims, eps)
            replacement.weight = module.weight  # Share weights
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            count += 1
    return count
```

### Usage with mlx-lm

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
n = patch_rmsnorm(model)
print(f"Patched {n} RMSNorm modules")

# Model works as before, now with custom kernels
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
```

## End-to-End Example

See `scripts/e2e_custom_kernels.py` for a complete example that:
1. Loads a quantized LLM via `mlx-lm`
2. Benchmarks baseline generation speed
3. Patches all RMSNorm modules with a custom Metal kernel
4. Benchmarks patched speed and prints a comparison table

Run it with: `python scripts/e2e_custom_kernels.py --model mlx-community/Llama-3.2-1B-Instruct-4bit`

Requires `mlx-lm`: `pip install mlx-lm`

## Kernel Caching Best Practices

### Compile Once, Use Many Times

`mx.fast.metal_kernel()` JIT-compiles the kernel on first use and caches the compiled binary. However, **calling `metal_kernel()` itself has overhead** from parsing and preparing the source. Keep the kernel object alive:

```python
# GOOD: Compile once at module level
_my_kernel = mx.fast.metal_kernel(name="my_op", ...)

def my_op(x):
    return _my_kernel(inputs=[x], ...)[0]


# BAD: Recompiles every call
def my_op(x):
    kernel = mx.fast.metal_kernel(name="my_op", ...)  # Wasteful!
    return kernel(inputs=[x], ...)[0]
```

For a reusable class-based caching pattern that supports multiple kernel types and dtypes, see the **Kernel Caching Pattern** section in `kernel-templates.md`.

## Autotuning Threadgroup Sizes

For production use, autotune and persist results:

```python
import json
import time
from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "my_kernels" / "autotune.json"


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def autotune_and_cache(kernel, name, inputs, grid, output_shapes, output_dtypes):
    cache = load_cache()
    device = mx.device_info().get("device_name", "unknown")
    key = f"{device}:{name}:{grid[0]}"

    if key in cache:
        return cache[key]

    best_tg, best_time = 256, float("inf")
    for tg in (32, 64, 128, 256, 512):
        if tg > grid[0]:
            continue
        # Warmup
        for _ in range(3):
            mx.eval(kernel(inputs=inputs, grid=grid, threadgroup=(tg, 1, 1),
                           output_shapes=output_shapes, output_dtypes=output_dtypes))
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            mx.eval(kernel(inputs=inputs, grid=grid, threadgroup=(tg, 1, 1),
                           output_shapes=output_shapes, output_dtypes=output_dtypes))
        mx.synchronize()
        dt = (time.perf_counter() - t0) / 10
        if dt < best_time:
            best_time, best_tg = dt, tg

    cache[key] = best_tg
    save_cache(cache)
    return best_tg
```

## Interaction with mx.compile

`mx.compile` traces MLX operations into a fused graph. Custom `metal_kernel` calls **are compatible** with `mx.compile`:

```python
@mx.compile
def forward(x, weight):
    # Custom kernel call inside compiled function
    out = _rmsnorm_kernel(
        inputs=[x, weight],
        template=[("T", x.dtype)],
        grid=(x.shape[0], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
    return out
```

For a deeper treatment including shape specialization pitfalls and recompilation patterns, see `references/mx-compile-interaction.md`.

**When to use which:**

| Scenario | Use `mx.compile` | Use `metal_kernel` |
|----------|-------------------|--------------------|
| Standard ops (matmul, add) | Yes | No |
| Fusing 2–3 simple ops | Yes (often sufficient) | Optional |
| Complex reductions | Maybe | Yes |
| Operations not in MLX | No | Yes |
| Tight inner loops | Yes (wrapping kernel) | Yes (the kernel itself) |

## Working with Non-Contiguous Arrays

By default, `metal_kernel` copies inputs to row-contiguous layout. For performance-critical paths where you control memory layout:

```python
kernel = mx.fast.metal_kernel(
    name="strided_op",
    input_names=["x"],
    output_names=["out"],
    source="""
    uint i = thread_position_in_grid.x;
    // Use auto-injected strides for manual indexing
    size_t offset = elem_to_loc(i, x_shape, x_strides, x_ndim);
    out[i] = (T)((float)x[offset] * 2.0f);
    """,
    ensure_row_contiguous=False,  # Skip the copy
)
```

This avoids an O(N) copy but requires careful indexing in the kernel source.

## Custom Gradients with `custom_function()`

Custom Metal kernels don't automatically support `mx.grad()`. To enable backpropagation through custom kernels (needed for training), use `mx.custom_function()` to define a VJP (vector-Jacobian product):

```python
@mx.custom_function
def custom_silu(x):
    # Forward: custom kernel
    return _silu_kernel(
        inputs=[x.reshape(-1)],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(x.size,)],
        output_dtypes=[x.dtype],
    )[0].reshape(x.shape)

@custom_silu.vjp
def custom_silu_vjp(primals, cotangents, outputs):
    x = primals[0]
    # SiLU gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    sig = mx.sigmoid(x)
    grad = cotangents[0] * (sig + x * sig * (1 - sig))
    return (grad,)
```

This lets you use custom kernels for the forward pass while defining gradients with standard MLX ops (or another custom kernel for the backward pass).

## Runtime Device Detection

Use `mx.device_info()` to detect the chip at runtime and make optimization decisions:

```python
info = mx.device_info()
# Returns dict with keys like "device_name", "memory_size", etc.

# Example: choose bfloat16 on M3+ (Apple GPU family 9)
device_name = info.get("device_name", "")
supports_bf16 = any(chip in device_name for chip in ["M3", "M4"])
dtype = mx.bfloat16 if supports_bf16 else mx.float16
```
