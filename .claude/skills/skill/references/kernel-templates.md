# Metal Kernel Templates for MLX

Copy-paste ready templates for common kernel types. Each template includes the Metal source, Python wrapper, and grid/threadgroup configuration.

## Template 1: Element-wise Operation (Activation Functions)

Use for: SiLU, GELU, ReLU, Swish, or any pointwise transform.

### Metal Source

```python
SILU_SOURCE = """
uint i = thread_position_in_grid.x;
if (i >= out_shape[0]) return;
float val = (float)x[i];
float sigmoid = 1.0f / (1.0f + metal::exp(-val));
out[i] = (T)(val * sigmoid);
"""
```

### Python Wrapper

```python
import mlx.core as mx

_silu_kernel = mx.fast.metal_kernel(
    name="silu",
    input_names=["x"],
    output_names=["out"],
    source=SILU_SOURCE,
)

def silu(x: mx.array) -> mx.array:
    dtype = x.dtype
    flat = x.reshape(-1)
    N = flat.size
    out = _silu_kernel(
        inputs=[flat],
        template=[("T", dtype)],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[dtype],
    )[0]
    return out.reshape(x.shape)
```

### GELU Variant (Approximate)

```python
GELU_SOURCE = """
uint i = thread_position_in_grid.x;
if (i >= out_shape[0]) return;
float val = (float)x[i];
// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float cube = val * val * val;
float inner = 0.7978845608f * (val + 0.044715f * cube);
out[i] = (T)(0.5f * val * (1.0f + metal::tanh(inner)));
"""
```

## Template 2: Row-wise Reduction — RMSNorm

Use for: RMSNorm, LayerNorm variance, row sums, row max.

### Basic Version (Single Simdgroup, D ≤ 32)

```python
RMSNORM_BASIC_SOURCE = """
uint row = thread_position_in_grid.x;
uint lane = thread_index_in_simdgroup;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

if (row >= out_shape[0]) return;

float sum_sq = 0.0f;
for (uint i = lane; i < D; i += 32) {
    float val = (float)x[row * D + i];
    sum_sq += val * val;
}
sum_sq = simd_sum(sum_sq);
float rms = metal::rsqrt(sum_sq / float(D) + eps_val);

for (uint i = lane; i < D; i += 32) {
    float val = (float)x[row * D + i];
    out[row * D + i] = (T)(val * rms * (float)w[i]);
}
"""

_rmsnorm_kernel = mx.fast.metal_kernel(
    name="rmsnorm",
    input_names=["x", "w", "eps"],
    output_names=["out"],
    source=RMSNORM_BASIC_SOURCE,
)

def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    eps_arr = mx.array([eps], dtype=mx.float32)
    # grid = total threads; one threadgroup of 32 per row
    out = _rmsnorm_kernel(
        inputs=[x_2d, weight, eps_arr],
        template=[("T", x.dtype)],
        grid=(N * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)
```

### Optimized Version (Multiple Simdgroups, Large D)

For hidden dimensions > 32, use multiple simdgroups with threadgroup memory:

```python
RMSNORM_OPTIMIZED_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

// Each thread accumulates partial sum across strided elements
float sum_sq = 0.0f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float val = (float)x[row * D + i];
    sum_sq += val * val;
}

// Reduce within simdgroup
sum_sq = simd_sum(sum_sq);

// Cross-simdgroup reduction via threadgroup memory
threadgroup float shared[32];
if (lane == 0) shared[sg] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    float partial = (lane < (threads_per_threadgroup.x + 31) / 32)
                    ? shared[lane] : 0.0f;
    float total = simd_sum(partial);
    shared[0] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float rms = metal::rsqrt(shared[0] / float(D) + eps_val);

// Apply normalization + weight
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float val = (float)x[row * D + i];
    out[row * D + i] = (T)(val * rms * (float)w[i]);
}
"""

_rmsnorm_opt_kernel = mx.fast.metal_kernel(
    name="rmsnorm_opt",
    input_names=["x", "w", "eps"],
    output_names=["out"],
    source=RMSNORM_OPTIMIZED_SOURCE,
)

def rmsnorm_optimized(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    tg = min(256, D)  # One threadgroup per row
    tg = max(32, (tg // 32) * 32)  # Round to SIMD multiple
    eps_arr = mx.array([eps], dtype=mx.float32)
    # grid = total threads; one threadgroup per row
    out = _rmsnorm_opt_kernel(
        inputs=[x_2d, weight, eps_arr],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)
```

## Template 3: Fused Reduction + Element-wise (RMSNorm + Residual Add)

Fusing operations avoids redundant memory reads:

```python
FUSED_RMSNORM_RESIDUAL_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

// Step 1: Compute residual = x + residual, and accumulate sum_sq
float sum_sq = 0.0f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float r = (float)x[row * D + i] + (float)residual[row * D + i];
    residual_out[row * D + i] = (T)r;
    sum_sq += r * r;
}

// Step 2: Reduce
sum_sq = simd_sum(sum_sq);
threadgroup float shared[32];
if (lane == 0) shared[sg] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float partial = (lane < (threads_per_threadgroup.x + 31) / 32)
                    ? shared[lane] : 0.0f;
    shared[0] = simd_sum(partial);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float rms = metal::rsqrt(shared[0] / float(D) + eps_val);

// Step 3: Normalize + scale (reads residual_out which is in cache)
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float val = (float)residual_out[row * D + i];
    out[row * D + i] = (T)(val * rms * (float)w[i]);
}
"""
# Note: input_names must include "eps" as a float32 buffer, e.g.:
# input_names=["x", "residual", "w", "eps"]
# output_names=["out", "residual_out"]
```

## Template 4: Softmax with Numerical Stability

Three-pass approach: find max, compute exp and sum, then normalize. For a more memory-efficient alternative, see the "online softmax" algorithm which combines the max and exp passes, avoiding a full write/read of intermediate values to global memory.



```python
SOFTMAX_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint D = x_shape[x_ndim - 1];

// Pass 1: Find row max
float local_max = -1e38f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    local_max = max(local_max, (float)x[row * D + i]);
}
local_max = simd_max(local_max);

threadgroup float shared[32];
if (lane == 0) shared[sg] = local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float partial = (lane < (threads_per_threadgroup.x + 31) / 32)
                    ? shared[lane] : -1e38f;
    shared[0] = simd_max(partial);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_max = shared[0];

// Pass 2: Compute exp(x - max) and sum
float local_sum = 0.0f;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    float e = metal::exp((float)x[row * D + i] - row_max);
    out[row * D + i] = (T)e;
    local_sum += e;
}
local_sum = simd_sum(local_sum);

if (lane == 0) shared[sg] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float partial = (lane < (threads_per_threadgroup.x + 31) / 32)
                    ? shared[lane] : 0.0f;
    shared[0] = simd_sum(partial);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_sum = shared[0];

// Pass 3: Normalize
float inv_sum = 1.0f / row_sum;
for (uint i = tid; i < D; i += threads_per_threadgroup.x) {
    out[row * D + i] = (T)((float)out[row * D + i] * inv_sum);
}
"""

_softmax_kernel = mx.fast.metal_kernel(
    name="softmax",
    input_names=["x"],
    output_names=["out"],
    source=SOFTMAX_SOURCE,
)

def softmax(x: mx.array) -> mx.array:
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    tg = min(256, D)
    tg = max(32, (tg // 32) * 32)
    # grid = total threads; one threadgroup per row
    out = _softmax_kernel(
        inputs=[x_2d],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)
```

## Template 5: Parameterized Kernel via Integer Buffer

For passing runtime parameters (like `eps`, dimensions) without recompilation:

Note: This source uses `{{` double braces because it's passed through Python's `.format()` —
`{{` produces a literal `{` in the Metal output.

```python
PARAMETERIZED_SOURCE = """
uint t = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
int hop = params[0];
int frame_len = params[1];
int n_frames = params[2];
int out_len = params[3];

if (t >= (uint)out_len) return;

float acc = 0.0f;
int base = (int)b * n_frames * frame_len;

for (int k = 0; k < n_frames; ++k) {{
    int off = (int)t - k * hop;
    if (off >= 0 && off < frame_len) {{
        acc += (float)frames[base + k * frame_len + off];
    }}
}}

out[(int)b * out_len + (int)t] = ({T})acc;
"""

# Pass parameters as an int32 array
params = mx.array([hop, frame_len, n_frames, out_len], dtype=mx.int32)
kernel = mx.fast.metal_kernel(
    name="ola",
    input_names=["frames", "params"],
    output_names=["out"],
    source=PARAMETERIZED_SOURCE.format(T="float"),
)
```

## Template 6: Atomic Reduction (for Gradient Accumulation)

```python
ATOMIC_ADD_SOURCE = """
uint i = thread_position_in_grid.x;
uint target = indices[i];
atomic_fetch_add_explicit(
    (device atomic<float>*)&grad[target],
    (float)values[i],
    memory_order_relaxed
);
"""

kernel = mx.fast.metal_kernel(
    name="scatter_add",
    input_names=["values", "indices"],
    output_names=["grad"],
    source=ATOMIC_ADD_SOURCE,
    atomic_outputs=True,
)
# Use init_value=0 to zero the output before accumulating
result = kernel(
    inputs=[values, indices],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(vocab_size,)],
    output_dtypes=[mx.float32],
    init_value=0,
)
```

## Template 7: Rotary Position Embeddings (RoPE)

Apply rotary embeddings to query/key tensors. Pairs of elements are rotated by position-dependent angles.

```python
ROPE_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint D = x_shape[x_ndim - 1];
uint half_D = D / 2;

// Each thread handles one pair (i, i + half_D)
for (uint i = tid; i < half_D; i += threads_per_threadgroup.x) {
    float freq = 1.0f / metal::pow(10000.0f, float(2 * i) / float(D));
    float pos = (float)offset[0] + (float)row;
    float angle = pos * freq;
    float cos_val = metal::cos(angle);
    float sin_val = metal::sin(angle);

    float x0 = (float)x[row * D + i];
    float x1 = (float)x[row * D + i + half_D];

    out[row * D + i]          = (T)(x0 * cos_val - x1 * sin_val);
    out[row * D + i + half_D] = (T)(x1 * cos_val + x0 * sin_val);
}
"""

_rope_kernel = mx.fast.metal_kernel(
    name="rope",
    input_names=["x", "offset"],
    output_names=["out"],
    source=ROPE_SOURCE,
)

def rope(x: mx.array, offset: int = 0) -> mx.array:
    """Apply rotary position embeddings.

    Args:
        x: Input of shape (..., D) where D is even.
        offset: Position offset (for KV-cache continuation).
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    assert D % 2 == 0, "RoPE requires even last dimension"
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    tg = min(256, D // 2)
    tg = max(32, (tg // 32) * 32)
    offset_arr = mx.array([offset], dtype=mx.int32)
    out = _rope_kernel(
        inputs=[x_2d, offset_arr],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)
```

> **Note:** For production use, prefer `mx.fast.rope()` which is already optimized. This template is useful for custom RoPE variants (e.g., NTK-aware scaling, YaRN, dynamic scaling) that differ from the built-in.

## Template 8: Tiled Attention (Online Softmax)

Single-head scaled dot-product attention: `softmax(Q @ K^T / sqrt(d)) @ V` with online softmax to avoid materializing the full attention matrix.

**Key ideas:**
- Each threadgroup handles one query row
- Iterate over K/V rows, computing scores incrementally
- Online softmax: maintain running `max` and `sum`, rescale output when max changes
- Float32 accumulation for numerical stability

**When to use:** Learning/custom attention patterns (sliding window, sparse). For production, use `mx.fast.scaled_dot_product_attention`.

**Grid setup:**
```python
grid = (tg, seq_q, 1)        # Y = query index
threadgroup = (tg, 1, 1)      # Threads cooperate on one query row
```

See `references/attention-kernel-guide.md` for the full algorithm explanation and `scripts/attention_kernel.py` for the working implementation.

## Kernel Caching Pattern

Compile the kernel once, reuse across calls:

```python
class MyKernels:
    _cache = {}

    @classmethod
    def get(cls, name: str, dtype: mx.Dtype):
        key = (name, dtype)
        if key not in cls._cache:
            cls._cache[key] = mx.fast.metal_kernel(
                name=f"{name}_{dtype}",
                input_names=["x", "w"],
                output_names=["out"],
                source=SOURCES[name],
            )
        return cls._cache[key]
```
