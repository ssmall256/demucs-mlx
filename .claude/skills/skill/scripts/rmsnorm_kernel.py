"""RMSNorm Metal kernel for MLX with simdgroup reduction.

Demonstrates:
- mx.fast.metal_kernel() API usage
- Simdgroup and cross-simdgroup reduction
- Threadgroup memory for partial sums
- Float32 accumulation with half-precision I/O
- Passing runtime parameters (eps) via input buffer
- Correctness validation against mx.fast.rms_norm()
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.utils import scalar_f32
from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel source: one threadgroup per row, threads stride over the hidden dim
# eps is passed as a 1-element float32 input buffer (not hardcoded)
# ---------------------------------------------------------------------------
RMSNORM_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

// Step 1: Each thread accumulates sum of squares over strided elements
float sum_sq = 0.0f;
for (uint i = tid; i < D; i += tg_size) {
    float val = (float)x[row * D + i];
    sum_sq += val * val;
}

// Step 2: Reduce within each simdgroup (32 threads)
sum_sq = simd_sum(sum_sq);

// Step 3: Cross-simdgroup reduction via threadgroup memory
threadgroup float shared[32];
if (lane == 0) {
    shared[sg] = sum_sq;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    uint num_sg = (tg_size + 31) / 32;
    float partial = (lane < num_sg) ? shared[lane] : 0.0f;
    float total = simd_sum(partial);
    shared[0] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Step 4: Compute RMS scale factor
float rms = metal::rsqrt(shared[0] / float(D) + eps_val);

// Step 5: Apply normalization and weight
for (uint i = tid; i < D; i += tg_size) {
    float val = (float)x[row * D + i];
    out[row * D + i] = (T)(val * rms * (float)w[i]);
}
"""

# Compile kernel once at module level
_rmsnorm_kernel = mx.fast.metal_kernel(
    name="rmsnorm",
    input_names=["x", "w", "eps"],
    output_names=["out"],
    source=RMSNORM_SOURCE,
)


def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    """Apply RMSNorm using a custom Metal kernel.

    Args:
        x: Input tensor of shape (..., D).
        weight: Scale weights of shape (D,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of same shape as x.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    # Choose threadgroup size: multiple of 32, capped at 256
    tg = min(256, D)
    tg = max(32, (tg // 32) * 32)

    eps_arr = scalar_f32(float(eps))

    # grid = total threads; one threadgroup per row → N * tg total threads
    out = _rmsnorm_kernel(
        inputs=[x_2d, weight, eps_arr],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)


def validate(shapes=None, dtype=mx.float32):
    """Validate custom RMSNorm against mx.fast.rms_norm."""
    if shapes is None:
        shapes = [
            (1, 128, 4096),
            (1, 512, 2048),
            (4, 256, 1024),
            (1, 1, 32),
            (2, 2048, 4096),
            # Edge cases
            (1, 1),
            (1, 16),
            (1, 16384),
        ]

    print(f"Validating RMSNorm kernel (dtype={dtype})")
    print("-" * 60)

    all_passed = True
    for shape in shapes:
        D = shape[-1]
        x = mx.random.normal(shape).astype(dtype)
        w = mx.random.normal((D,)).astype(dtype)

        expected = mx.fast.rms_norm(x, w, eps=1e-5)
        actual = rmsnorm(x, w, eps=1e-5)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < 1e-2 if dtype == mx.float16 else max_diff < 1e-5
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {str(shape):>20s}  max_diff={max_diff:.2e}  [{status}]")

    # Test non-contiguous input (transposed)
    D = 128
    x_t = mx.random.normal((D, 4)).T.astype(dtype)  # (4, 128) non-contiguous
    w_t = mx.random.normal((D,)).astype(dtype)
    expected_t = mx.fast.rms_norm(x_t, w_t, eps=1e-5)
    actual_t = rmsnorm(x_t, w_t, eps=1e-5)
    mx.eval(expected_t, actual_t)
    max_diff_t = mx.max(mx.abs(expected_t - actual_t)).item()
    atol = 1e-2 if dtype == mx.float16 else 1e-5
    passed_t = max_diff_t < atol
    if not passed_t:
        all_passed = False
    print(f"  {'(4, 128) transposed':>20s}  max_diff={max_diff_t:.2e}  [{'PASS' if passed_t else 'FAIL'}]")

    print("-" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== RMSNorm Metal Kernel Demo ===\n")

    # Quick demo
    x = mx.random.normal((2, 8, 4096))
    w = mx.ones((4096,))
    out = rmsnorm(x, w)
    mx.eval(out)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean:  {mx.mean(out).item():.4f}")
    print()

    # Validate correctness
    validate(dtype=mx.float32)
    print()
    validate(dtype=mx.float16)
