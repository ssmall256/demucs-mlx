"""Batched element-wise SiLU using a 2D grid dispatch.

Demonstrates:
- 2D grid: Y dimension for batch/row, X dimension for elements
- thread_position_in_grid.x/.y for 2D indexing
- Avoiding integer division for batch index computation
- Correctness validation against mlx.nn.silu()
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.autotune_cache import pick_threadgroup
import mlx.nn

# ---------------------------------------------------------------------------
# Kernel source: 2D grid — .y = batch, .x = element
# ---------------------------------------------------------------------------
BATCH_SILU_SOURCE = """
uint elem = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint D = x_shape[x_ndim - 1];
uint N = x_shape[0];

if (elem >= D || row >= N) return;

uint idx = row * D + elem;
float val = (float)x[idx];
float sigmoid = 1.0f / (1.0f + metal::exp(-val));
out[idx] = (T)(val * sigmoid);
"""

# Compile kernel once at module level
_batch_silu_kernel = mx.fast.metal_kernel(
    name="batch_silu_2d",
    input_names=["x"],
    output_names=["out"],
    source=BATCH_SILU_SOURCE,
)


def batch_silu(x: mx.array) -> mx.array:
    """Apply SiLU activation using a 2D grid dispatch.

    Args:
        x: Input tensor of shape (..., D).

    Returns:
        SiLU(x) with same shape as input.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    N = x_2d.shape[0]

    tgx = min(256, D)

    out = _batch_silu_kernel(
        inputs=[x_2d],
        template=[("T", x.dtype)],
        grid=(D, N, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(orig_shape)


def validate(shapes=None, dtype=mx.float32):
    """Validate batched SiLU against mlx.nn.silu."""
    if shapes is None:
        shapes = [
            (1, 128),
            (4, 1024),
            (8, 4096),
            (2, 8, 2048),
            (1, 1),
            (1, 16),
            (1, 16384),
        ]

    print(f"Validating batch SiLU kernel (dtype={dtype})")
    print("-" * 60)

    all_passed = True
    for shape in shapes:
        x = mx.random.normal(shape).astype(dtype)

        expected = mlx.nn.silu(x)
        actual = batch_silu(x)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        atol = 1e-2 if dtype == mx.float16 else 1e-5
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {str(shape):>20s}  max_diff={max_diff:.2e}  [{status}]")

    print("-" * 60)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== Batched SiLU (2D Grid) Metal Kernel Demo ===\n")

    # Quick demo
    x = mx.random.normal((4, 2048))
    out = batch_silu(x)
    mx.eval(out)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean:  {mx.mean(out).item():.4f}")
    print()

    # Validate correctness
    validate(dtype=mx.float32)
    print()
    validate(dtype=mx.float16)
