"""Numerically stable softmax Metal kernel for MLX.

Demonstrates:
- Three-pass softmax: find max, compute exp+sum, normalize
- Simdgroup reductions (simd_max, simd_sum)
- Cross-simdgroup reduction via threadgroup memory
- Correctness validation against mx.softmax()
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel source: one threadgroup per row, three-pass stable softmax
# ---------------------------------------------------------------------------
SOFTMAX_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint D = x_shape[x_ndim - 1];
uint num_sg = (tg_size + 31) / 32;

threadgroup float shared[32];

// ---- Pass 1: Find row maximum (for numerical stability) ----
float local_max = -1e38f;
for (uint i = tid; i < D; i += tg_size) {
    local_max = max(local_max, (float)x[row * D + i]);
}
local_max = simd_max(local_max);
if (lane == 0) shared[sg] = local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float p = (lane < num_sg) ? shared[lane] : -1e38f;
    shared[0] = simd_max(p);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_max = shared[0];

// ---- Pass 2: Compute exp(x - max) and sum ----
float local_sum = 0.0f;
for (uint i = tid; i < D; i += tg_size) {
    float e = metal::exp((float)x[row * D + i] - row_max);
    out[row * D + i] = (T)e;  // Store temporarily
    local_sum += e;
}
local_sum = simd_sum(local_sum);
if (lane == 0) shared[sg] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float p = (lane < num_sg) ? shared[lane] : 0.0f;
    shared[0] = simd_sum(p);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv_sum = 1.0f / shared[0];

// ---- Pass 3: Normalize ----
for (uint i = tid; i < D; i += tg_size) {
    out[row * D + i] = (T)((float)out[row * D + i] * inv_sum);
}
"""

_softmax_kernel = mx.fast.metal_kernel(
    name="softmax",
    input_names=["x"],
    output_names=["out"],
    source=SOFTMAX_SOURCE,
)


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Apply softmax using a custom Metal kernel.

    Args:
        x: Input tensor.
        axis: Axis along which to compute softmax (must be -1 or last axis).

    Returns:
        Softmax probabilities with same shape as x.
    """
    if axis != -1 and axis != len(x.shape) - 1:
        raise ValueError("Custom softmax only supports axis=-1 (last dimension)")

    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    base_tg = min(256, D)
    base_tg = max(32, (base_tg // 32) * 32)
    tg = pick_threadgroup(
        kernel_name="softmax",
        shape_sig=f"N={N},D={D}",
        dtype_sig=str(x.dtype),
        candidates=(32, 64, 128, 256, 512, 1024),
        default=base_tg,
        run=lambda tgx: _softmax_kernel(
            inputs=[x_2d],
            template=[("T", x.dtype)],
            grid=(N * tgx, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(N, D)],
            output_dtypes=[x.dtype],
        )[0],
        warmup=2,
        iters=30,
    )

    # grid = total threads; one threadgroup per row → N * tg total threads
    out = _softmax_kernel(
        inputs=[x_2d],
        template=[("T", x.dtype)],
        grid=(N * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)


def validate(shapes=None, dtype=mx.float32):
    """Validate custom softmax against mx.softmax."""
    if shapes is None:
        shapes = [
            (1, 128),
            (4, 1024),
            (2, 4096),
            (8, 32),
            (1, 50257),
            (2, 8, 4096),
            # Edge cases
            (1, 1),
            (1, 16),
            (1, 16384),
        ]

    print(f"Validating softmax kernel (dtype={dtype})")
    print("-" * 60)

    all_passed = True
    for shape in shapes:
        x = mx.random.normal(shape).astype(dtype)

        expected = mx.softmax(x, axis=-1)
        actual = softmax(x, axis=-1)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        # Check probabilities sum to 1
        sum_check = mx.max(mx.abs(mx.sum(actual, axis=-1) - 1.0)).item()

        atol = 1e-3 if dtype == mx.float16 else 1e-5
        passed = max_diff < atol and sum_check < atol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  {str(shape):>20s}  max_diff={max_diff:.2e}  "
            f"sum_err={sum_check:.2e}  [{status}]"
        )

    # Test non-contiguous input (transposed)
    D = 128
    x_t = mx.random.normal((D, 4)).T.astype(dtype)  # (4, 128) non-contiguous
    expected_t = mx.softmax(x_t, axis=-1)
    actual_t = softmax(x_t, axis=-1)
    mx.eval(expected_t, actual_t)
    max_diff_t = mx.max(mx.abs(expected_t - actual_t)).item()
    sum_check_t = mx.max(mx.abs(mx.sum(actual_t, axis=-1) - 1.0)).item()
    atol = 1e-3 if dtype == mx.float16 else 1e-5
    passed_t = max_diff_t < atol and sum_check_t < atol
    if not passed_t:
        all_passed = False
    print(
        f"  {'(4, 128) transposed':>20s}  max_diff={max_diff_t:.2e}  "
        f"sum_err={sum_check_t:.2e}  [{'PASS' if passed_t else 'FAIL'}]"
    )

    print("-" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== Softmax Metal Kernel Demo ===\n")

    # Quick demo
    x = mx.random.normal((2, 4096))
    out = softmax(x)
    mx.eval(out)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Row sums:     {mx.sum(out, axis=-1).tolist()}")
    print(f"Min value:    {mx.min(out).item():.6f}")
    print(f"Max value:    {mx.max(out).item():.6f}")
    print()

    # Validate correctness
    validate(dtype=mx.float32)
    print()
    validate(dtype=mx.float16)
