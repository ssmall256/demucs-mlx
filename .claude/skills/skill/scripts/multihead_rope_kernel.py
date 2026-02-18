"""Multi-head RoPE (Rotary Position Embeddings) kernel for 4D tensors.

Demonstrates:
- 3D grid dispatch over (pair, seq_position, batch*heads)
- Operating on 4D tensors of shape (batch, heads, seq, dim)
- Decomposing composite grid indices (batch*heads → batch, head)
- Strided variant using elem_to_loc for non-contiguous inputs
- Correctness validation against pure-MLX reference

RoPE applies a rotation to pairs of elements at positions (i, i+D/2):
    x0' = x0 * cos(theta) - x1 * sin(theta)
    x1' = x1 * cos(theta) + x0 * sin(theta)
where theta = position * (1 / 10000^(2i/D)).
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel source: 3D grid over (pair_index, seq_position, batch*heads)
# Assumes contiguous (B, H, S, D) layout
# ---------------------------------------------------------------------------
MULTIHEAD_ROPE_SOURCE = """
uint pair = thread_position_in_grid.x;
uint pos = thread_position_in_grid.y;
uint bh = thread_position_in_grid.z;   // batch * heads composite index

uint B = x_shape[0];
uint H = x_shape[1];
uint S = x_shape[2];
uint D = x_shape[3];
uint half_D = D / 2;

if (pair >= half_D) return;

uint batch_idx = bh / H;
uint head_idx = bh % H;

// Flat index into contiguous (B, H, S, D) tensor
uint base = batch_idx * (H * S * D) + head_idx * (S * D) + pos * D;

// Compute rotation angle: theta = (offset + pos) / 10000^(2*pair/D)
float freq = 1.0f / metal::pow(10000.0f, float(2 * pair) / float(D));
float angle = (float(offset[0]) + float(pos)) * freq;
float cos_val = metal::cos(angle);
float sin_val = metal::sin(angle);

float x0 = (float)x[base + pair];
float x1 = (float)x[base + pair + half_D];

out[base + pair]          = (T)(x0 * cos_val - x1 * sin_val);
out[base + pair + half_D] = (T)(x1 * cos_val + x0 * sin_val);
"""

# ---------------------------------------------------------------------------
# Strided variant: handles non-contiguous inputs via x_strides
# ---------------------------------------------------------------------------
MULTIHEAD_ROPE_STRIDED_SOURCE = """
uint pair = thread_position_in_grid.x;
uint pos = thread_position_in_grid.y;
uint bh = thread_position_in_grid.z;

uint B = x_shape[0];
uint H = x_shape[1];
uint S = x_shape[2];
uint D = x_shape[3];
uint half_D = D / 2;

if (pair >= half_D) return;

uint batch_idx = bh / H;
uint head_idx = bh % H;

// Use strides for non-contiguous access (e.g., after transpose)
size_t off0 = batch_idx * x_strides[0] + head_idx * x_strides[1]
            + pos * x_strides[2] + pair * x_strides[3];
size_t off1 = off0 + half_D * x_strides[3];

float freq = 1.0f / metal::pow(10000.0f, float(2 * pair) / float(D));
float angle = (float(offset[0]) + float(pos)) * freq;
float cos_val = metal::cos(angle);
float sin_val = metal::sin(angle);

float x0 = (float)x[off0];
float x1 = (float)x[off1];

// Write to contiguous output
uint out_base = batch_idx * (H * S * D) + head_idx * (S * D) + pos * D;
out[out_base + pair]          = (T)(x0 * cos_val - x1 * sin_val);
out[out_base + pair + half_D] = (T)(x1 * cos_val + x0 * sin_val);
"""

_rope_kernel = mx.fast.metal_kernel(
    name="multihead_rope",
    input_names=["x", "offset"],
    output_names=["out"],
    source=MULTIHEAD_ROPE_SOURCE,
)

_rope_strided_kernel = mx.fast.metal_kernel(
    name="multihead_rope_strided",
    input_names=["x", "offset"],
    output_names=["out"],
    source=MULTIHEAD_ROPE_STRIDED_SOURCE,
    ensure_row_contiguous=False,
)


def multihead_rope(x: mx.array, offset: int = 0) -> mx.array:
    """Apply RoPE to a 4D tensor using a custom Metal kernel.

    Args:
        x: Input of shape (batch, heads, seq, dim). dim must be even.
        offset: Position offset (for incremental decoding).

    Returns:
        Rotated tensor of same shape.
    """
    B, H, S, D = x.shape
    assert D % 2 == 0, f"dim must be even, got {D}"
    half_D = D // 2
    offset_buf = mx.array([offset], dtype=mx.int32)

    # 3D grid: X=pair index, Y=seq position, Z=batch*heads
    def _run(tgx: int):
        return _rope_kernel(
            inputs=[x, offset_buf],
            template=[("T", x.dtype)],
            grid=(half_D, S, B * H),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(B, H, S, D)],
            output_dtypes=[x.dtype],
        )[0]

    tgx = pick_threadgroup(
        kernel_name="rope_mh",
        shape_sig=f"B={B},H={H},S={S},D={D}",
        dtype_sig=str(x.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=min(256, half_D),
    )

    out = _run(tgx)
    return out


def multihead_rope_strided(x: mx.array, offset: int = 0) -> mx.array:
    """Apply RoPE using strides for non-contiguous inputs.

    Same API as multihead_rope but uses x_strides in the kernel,
    allowing it to handle transposed/non-contiguous tensors without
    the implicit copy from ensure_row_contiguous.
    """
    B, H, S, D = x.shape
    assert D % 2 == 0, f"dim must be even, got {D}"
    half_D = D // 2
    offset_buf = mx.array([offset], dtype=mx.int32)
    def _run(tgx: int):
        return _rope_strided_kernel(
            inputs=[x, offset_buf],
            template=[("T", x.dtype)],
            grid=(half_D, S, B * H),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(B, H, S, D)],
            output_dtypes=[x.dtype],
        )[0]

    tgx = pick_threadgroup(
        kernel_name="rope_mh_strided",
        shape_sig=f"heads={n_heads},S={S},D={D}",
        dtype_sig=str(x.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=min(256, half_D),
    )

    out = _run(tgx)
    return out


def _rope_reference(x: mx.array, offset: int = 0) -> mx.array:
    """Pure-MLX RoPE reference implementation."""
    B, H, S, D = x.shape
    half_D = D // 2

    # Build frequency table
    freqs = 1.0 / (10000.0 ** (mx.arange(0, D, 2, dtype=mx.float32) / D))
    positions = mx.arange(offset, offset + S, dtype=mx.float32)
    angles = positions[:, None] * freqs[None, :]  # (S, half_D)

    cos_vals = mx.cos(angles).astype(x.dtype)  # (S, half_D)
    sin_vals = mx.sin(angles).astype(x.dtype)

    x0 = x[..., :half_D]
    x1 = x[..., half_D:]

    out0 = x0 * cos_vals - x1 * sin_vals
    out1 = x1 * cos_vals + x0 * sin_vals
    return mx.concatenate([out0, out1], axis=-1)


def validate(shapes=None, dtype=mx.float32):
    """Validate multi-head RoPE kernels against reference."""
    if shapes is None:
        shapes = [
            (1, 1, 8, 64),
            (2, 4, 16, 128),
            (1, 32, 1, 128),    # Single-token generation, many heads
            (4, 8, 32, 64),
        ]

    print(f"Validating multi-head RoPE kernel (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    atol = 5e-2 if dtype == mx.float16 else 1e-4

    for shape in shapes:
        x = mx.random.normal(shape).astype(dtype)
        expected = _rope_reference(x, offset=0)

        # Test contiguous kernel
        actual = multihead_rope(x, offset=0)
        mx.eval(expected, actual)
        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label = f"{shape} contiguous"
        print(f"  {label:>35s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

        # Test strided kernel
        actual_s = multihead_rope_strided(x, offset=0)
        mx.eval(actual_s)
        max_diff_s = mx.max(mx.abs(expected - actual_s)).item()
        passed_s = max_diff_s < atol
        if not passed_s:
            all_passed = False
        label = f"{shape} strided"
        print(f"  {label:>35s}  max_diff={max_diff_s:.2e}  [{'PASS' if passed_s else 'FAIL'}]")

    # Test with non-contiguous input (transposed heads and seq dims)
    x_orig = mx.random.normal((2, 16, 4, 64)).astype(dtype)  # (B, S, H, D)
    x_t = mx.transpose(x_orig, axes=[0, 2, 1, 3])  # (B, H, S, D) non-contiguous

    expected_t = _rope_reference(x_t, offset=5)
    actual_t = multihead_rope_strided(x_t, offset=5)
    mx.eval(expected_t, actual_t)
    max_diff_t = mx.max(mx.abs(expected_t - actual_t)).item()
    passed_t = max_diff_t < atol
    if not passed_t:
        all_passed = False
    print(f"  {'(2,4,16,64) transposed, off=5':>35s}  max_diff={max_diff_t:.2e}  [{'PASS' if passed_t else 'FAIL'}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== Multi-Head RoPE Metal Kernel Demo ===\n")

    # Quick demo
    x = mx.random.normal((2, 8, 16, 128))
    out = multihead_rope(x)
    mx.eval(out)
    print(f"Input shape:  {x.shape}  (batch, heads, seq, dim)")
    print(f"Output shape: {out.shape}")
    print(f"Output mean:  {mx.mean(out).item():.4f}")
    print()

    # Validate correctness
    ok1 = validate(dtype=mx.float32)
    print()
    ok2 = validate(dtype=mx.float16)

    if not (ok1 and ok2):
        raise SystemExit(1)
