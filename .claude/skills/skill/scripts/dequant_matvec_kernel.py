"""4-bit dequantized matrix-vector multiply kernel for MLX.

Demonstrates:
- 4-bit weight extraction from packed uint32
- Fused dequantization + dot product
- Group-wise scale/bias application
- Cross-simdgroup reduction for row dot products
- Correctness validation against manual dequant + matmul
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel source: one threadgroup per output row
# Each thread accumulates partial dot product, then reduces
# ---------------------------------------------------------------------------
DEQUANT_MATVEC_SOURCE = """
uint out_row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint in_features = x_shape[0];
uint packed_per_row = in_features / 8;  // 4-bit: 8 elements per uint32
uint groups_per_row = in_features / GROUP_SIZE;

float acc = 0.0f;

// Each thread handles strided elements across input dimension
for (uint i = tid; i < in_features; i += tg_size) {
    // Extract 4-bit weight from packed uint32
    uint packed_idx = out_row * packed_per_row + i / 8;
    uint nibble_idx = i % 8;
    uint packed = w_packed[packed_idx];
    uint nibble = (packed >> (nibble_idx * 4)) & 0xF;

    // Look up group scale and bias
    uint group_idx = out_row * groups_per_row + i / GROUP_SIZE;
    float scale = (float)scales[group_idx];
    float bias = (float)biases[group_idx];

    // Dequantize and multiply with input
    float w_val = scale * (float)nibble + bias;
    acc += w_val * (float)x[i];
}

// Reduce within simdgroup
acc = simd_sum(acc);

// Cross-simdgroup reduction
threadgroup float shared[32];
if (lane == 0) shared[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (sg == 0) {
    uint num_sg = (tg_size + 31) / 32;
    float partial = (lane < num_sg) ? shared[lane] : 0.0f;
    float total = simd_sum(partial);
    if (lane == 0) {
        out[out_row] = (T)total;
    }
}
"""

_dequant_matvec_kernel = mx.fast.metal_kernel(
    name="dequant_matvec",
    input_names=["w_packed", "scales", "biases", "x"],
    output_names=["out"],
    source=DEQUANT_MATVEC_SOURCE,
)


def dequant_matvec(
    w_packed: mx.array,
    scales: mx.array,
    biases: mx.array,
    x: mx.array,
    group_size: int = 64,
) -> mx.array:
    """4-bit dequantized matrix-vector multiply: y = dequant(W_packed) @ x.

    Args:
        w_packed: Packed 4-bit weights, shape (out_features, in_features // 8), dtype uint32.
        scales: Per-group scales, shape (out_features, in_features // group_size), dtype float16.
        biases: Per-group biases, shape (out_features, in_features // group_size), dtype float16.
        x: Input vector, shape (in_features,), dtype float16.
        group_size: Number of elements per quantization group.

    Returns:
        Output vector, shape (out_features,), dtype float16.
    """
    out_features = w_packed.shape[0]
    def _run(tgx: int):
        return _dequant_matvec_kernel(
            inputs=[w_packed, scales, biases, x],
            template=[("T", mx.float16), ("GROUP_SIZE", group_size)],
            grid=(out_features * tgx, 1, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(out_features,)],
            output_dtypes=[mx.float16],
        )[0]

    tg = pick_threadgroup(
        kernel_name="dequant_matvec",
        shape_sig=f"out={out_features},in={x.shape[0]},g={group_size}",
        dtype_sig=str(x.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=256,
    )

    out = _run(tg)
    return out


def quantize_4bit(w: mx.array, group_size: int = 64):
    """Quantize a float matrix to 4-bit packed format.

    Args:
        w: Weight matrix, shape (out_features, in_features), dtype float32/float16.
        group_size: Elements per quantization group.

    Returns:
        (w_packed, scales, biases) tuple.
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0
    assert in_features % 8 == 0

    w_f32 = w.astype(mx.float32)

    # Reshape into groups
    w_grouped = w_f32.reshape(out_features, -1, group_size)  # (O, G, group_size)

    # Compute per-group min/max
    g_min = mx.min(w_grouped, axis=-1, keepdims=True)  # (O, G, 1)
    g_max = mx.max(w_grouped, axis=-1, keepdims=True)

    # Scale and bias: val = scale * int_val + bias
    # Map [min, max] -> [0, 15] for 4-bit
    scales_arr = (g_max - g_min) / 15.0
    # Avoid division by zero
    scales_arr = mx.where(scales_arr == 0, mx.ones_like(scales_arr), scales_arr)
    biases_arr = g_min

    # Quantize to [0, 15]
    quantized = mx.clip(mx.round((w_grouped - g_min) / scales_arr), 0, 15).astype(mx.uint32)
    mx.eval(quantized, scales_arr, biases_arr)

    # Pack 8 x 4-bit values into uint32
    # (Python loop over 8 nibbles for clarity; not performance-critical)
    quantized_flat = quantized.reshape(out_features, -1)  # (O, in_features)
    packed_list = []
    for k in range(8):
        packed_list.append(quantized_flat[:, k::8] << (k * 4))
    w_packed = packed_list[0]
    for p in packed_list[1:]:
        w_packed = w_packed | p
    # w_packed shape: (out_features, in_features // 8)

    scales_out = scales_arr.squeeze(-1).astype(mx.float16)
    biases_out = biases_arr.squeeze(-1).astype(mx.float16)

    mx.eval(w_packed, scales_out, biases_out)
    return w_packed, scales_out, biases_out


def dequantize_reference(w_packed: mx.array, scales: mx.array, biases: mx.array, group_size: int = 64):
    """Dequantize packed 4-bit weights back to float."""
    out_features = w_packed.shape[0]
    packed_per_row = w_packed.shape[1]
    in_features = packed_per_row * 8

    # Extract nibbles
    cols = []
    for k in range(8):
        nibbles = ((w_packed >> (k * 4)) & 0xF).astype(mx.float32)
        cols.append(nibbles)

    # Interleave: element i comes from packed[i//8] nibble i%8
    # cols[k] has shape (O, packed_per_row) — these are elements k, k+8, k+16, ...
    # We need to interleave them
    # Stack and transpose to get (O, packed_per_row, 8) then reshape
    stacked = mx.stack(cols, axis=-1)  # (O, packed_per_row, 8)
    w_int = stacked.reshape(out_features, in_features)  # (O, in_features)

    # Apply scale and bias per group
    groups = in_features // group_size
    w_grouped = w_int.reshape(out_features, groups, group_size)
    scales_expanded = scales.astype(mx.float32).reshape(out_features, groups, 1)
    biases_expanded = biases.astype(mx.float32).reshape(out_features, groups, 1)
    w_dequant = scales_expanded * w_grouped + biases_expanded
    return w_dequant.reshape(out_features, in_features)


def validate(configs=None):
    """Validate dequant matvec against reference dequant + matmul."""
    if configs is None:
        configs = [
            # (out_features, in_features, group_size)
            (64, 128, 64),
            (128, 256, 64),
            (256, 512, 64),
            (128, 512, 128),
            (64, 64, 32),
        ]

    print("Validating dequant matvec kernel")
    print("-" * 65)

    all_passed = True
    for out_f, in_f, gs in configs:
        # Create random weights and quantize
        w = mx.random.normal((out_f, in_f))
        w_packed, scales, biases = quantize_4bit(w, group_size=gs)

        # Input vector
        x = mx.random.normal((in_f,)).astype(mx.float16)

        # Reference: dequantize then matmul
        w_deq = dequantize_reference(w_packed, scales, biases, group_size=gs)
        expected = (w_deq.astype(mx.float16) @ x.astype(mx.float16))

        # Custom kernel
        actual = dequant_matvec(w_packed, scales, biases, x, group_size=gs)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()
        # Tolerance allows for float16 accumulation differences (typical diffs ~0.03)
        passed = max_diff < 0.5
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        label = f"({out_f}x{in_f}, gs={gs})"
        print(f"  {label:>25s}  max_diff={max_diff:.2e}  [{status}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== 4-bit Dequant MatVec Metal Kernel Demo ===\n")

    # Create a quantized weight matrix
    out_features, in_features = 256, 512
    group_size = 64

    w = mx.random.normal((out_features, in_features))
    w_packed, scales, biases = quantize_4bit(w, group_size=group_size)
    x = mx.random.normal((in_features,)).astype(mx.float16)

    print(f"Weight shape:  ({out_features}, {in_features})")
    print(f"Packed shape:  {w_packed.shape} (uint32)")
    print(f"Scales shape:  {scales.shape}")
    print(f"Input shape:   {x.shape}")
    print(f"Group size:    {group_size}")
    print(f"Compression:   {out_features * in_features * 2 / (w_packed.nbytes + scales.nbytes + biases.nbytes):.1f}x vs float16")
    print()

    out = dequant_matvec(w_packed, scales, biases, x, group_size=group_size)
    mx.eval(out)
    print(f"Output shape:  {out.shape}")
    print(f"Output mean:   {mx.mean(out).item():.4f}")
    print()

    # Validate
    validate()
