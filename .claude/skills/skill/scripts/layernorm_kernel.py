"""LayerNorm Metal kernel for MLX with two-pass simdgroup reduction.

Demonstrates:
- Two-pass reduction: first compute mean, then compute variance
- Shared memory reuse between passes via threadgroup barriers
- Weight (gamma) and bias (beta) affine transform
- Float32 accumulation with half-precision I/O
- Correctness validation against a manual reference implementation

Key difference from RMSNorm:
  RMSNorm uses a single pass (sum of squares) and no mean subtraction.
  LayerNorm requires two passes because variance depends on the mean:
    Pass 1: mean = sum(x) / D
    Pass 2: var  = sum((x - mean)^2) / D
  We use E[(x-mean)^2] rather than E[x^2] - mean^2 because the two-pass
  formula is more numerically stable (avoids catastrophic cancellation).
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.utils import scalar_f32
from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel source: one threadgroup per row, two-pass mean + variance
# ---------------------------------------------------------------------------
LAYERNORM_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint D = x_shape[x_ndim - 1];
float eps_val = (float)eps[0];
uint num_sg = (tg_size + 31) / 32;

// Shared memory is reused across both passes via barriers
threadgroup float shared[32];

// ---- Pass 1: Compute mean ----
float local_sum = 0.0f;
for (uint i = tid; i < D; i += tg_size) {
    local_sum += (float)x[row * D + i];
}
local_sum = simd_sum(local_sum);
if (lane == 0) {
    shared[sg] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float partial = (lane < num_sg) ? shared[lane] : 0.0f;
    shared[0] = simd_sum(partial);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared[0] / float(D);

// ---- Pass 2: Compute variance = E[(x - mean)^2] ----
// Two-pass variance is more stable than E[x^2] - mean^2
float local_var = 0.0f;
for (uint i = tid; i < D; i += tg_size) {
    float diff = (float)x[row * D + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (lane == 0) {
    shared[sg] = local_var;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float partial = (lane < num_sg) ? shared[lane] : 0.0f;
    shared[0] = simd_sum(partial);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv_std = metal::rsqrt(shared[0] / float(D) + eps_val);

// ---- Normalize with affine transform: out = (x - mean) / std * w + b ----
for (uint i = tid; i < D; i += tg_size) {
    float val = (float)x[row * D + i];
    out[row * D + i] = (T)((val - mean) * inv_std * (float)w[i] + (float)b[i]);
}
"""

# Compile kernel once at module level
_layernorm_kernel = mx.fast.metal_kernel(
    name="layernorm",
    input_names=["x", "w", "b", "eps"],
    output_names=["out"],
    source=LAYERNORM_SOURCE,
)


def layernorm(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5
) -> mx.array:
    """Apply LayerNorm using a custom Metal kernel.

    Args:
        x: Input tensor of shape (..., D).
        weight: Scale (gamma) of shape (D,).
        bias: Shift (beta) of shape (D,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of same shape as x.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]
    
    # eps passed as a 1-element float32 buffer so the kernel can read it
    eps_buf = scalar_f32(eps)

    # Choose threadgroup size: multiple of 32, capped at 256
    def _run(tgx: int):
        tgx = max(32, (tgx // 32) * 32)
        return _layernorm_kernel(
            inputs=[x_2d, weight, bias, eps_buf],
            template=[("T", x.dtype)],
            grid=(N, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(N, D)],
            output_dtypes=[x.dtype],
        )[0]

    tg = pick_threadgroup(
        kernel_name="layernorm",
        shape_sig=f"N={N},D={D}",
        dtype_sig=str(x.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=min(256, D),
    )

    # grid = total threads; one threadgroup per row -> N * tg total threads
    out = _run(tg)
    return out.reshape(orig_shape)


def _reference_layernorm(x: mx.array, weight: mx.array, bias: mx.array, eps: float):
    """Pure-MLX reference (no nn.LayerNorm dependency)."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * weight + bias


def validate(shapes=None, dtype=mx.float32):
    """Validate custom LayerNorm against manual reference."""
    if shapes is None:
        shapes = [
            (1, 128, 4096),
            (4, 256, 1024),
            (1, 1, 32),
            (1, 1),
            (1, 16),
            (1, 16384),
        ]

    print(f"Validating LayerNorm kernel (dtype={dtype})")
    print("-" * 60)

    all_passed = True
    for shape in shapes:
        D = shape[-1]
        x = mx.random.normal(shape).astype(dtype)
        w = mx.random.normal((D,)).astype(dtype)
        b = mx.random.normal((D,)).astype(dtype)

        expected = _reference_layernorm(x, w, b, eps=1e-5)
        actual = layernorm(x, w, b, eps=1e-5)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        # Two-pass reduction accumulates more FP16 error than single-pass RMSNorm
        atol = 2e-2 if dtype == mx.float16 else 1e-5
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {str(shape):>20s}  max_diff={max_diff:.2e}  [{status}]")

    # Test with transposed input (non-contiguous)
    x_t = mx.random.normal((64, 128)).astype(dtype).T  # shape (128, 64)
    D = 64
    w = mx.random.normal((D,)).astype(dtype)
    b = mx.random.normal((D,)).astype(dtype)
    expected = _reference_layernorm(x_t, w, b, eps=1e-5)
    actual = layernorm(x_t, w, b, eps=1e-5)
    mx.eval(expected, actual)
    max_diff = mx.max(mx.abs(expected - actual)).item()
    atol = 2e-2 if dtype == mx.float16 else 1e-5
    passed = max_diff < atol
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  {'(128,64) transposed':>20s}  max_diff={max_diff:.2e}  [{status}]")

    print("-" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== LayerNorm Metal Kernel Demo ===\n")

    # Quick demo
    x = mx.random.normal((2, 8, 4096))
    w = mx.ones((4096,))
    b = mx.zeros((4096,))
    out = layernorm(x, w, b)
    mx.eval(out)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean:  {mx.mean(out).item():.4f}")
    print(f"Output std:   {mx.var(out).item() ** 0.5:.4f}")
    print()

    # Validate correctness
    ok1 = validate(dtype=mx.float32)
    print()
    ok2 = validate(dtype=mx.float16)

    if not (ok1 and ok2):
        raise SystemExit(1)
