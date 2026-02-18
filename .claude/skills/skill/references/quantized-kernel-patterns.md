# Quantized Kernel Patterns

## MLX Quantized Format

MLX stores quantized weights in a packed format:

- **Packed weights** (`uint32`): Multiple low-bit values packed into 32-bit integers
  - 4-bit: 8 elements per `uint32`
  - 2-bit: 16 elements per `uint32`
  - 8-bit: 4 elements per `uint32`
- **Scales** (`float16`): One scale per group
- **Biases** (`float16`): One bias per group
- **Group size**: 32, 64, or 128 elements share one scale/bias pair

### Memory Layout

For a weight matrix of shape `(out_features, in_features)` quantized to 4-bit with group_size=64:

```
packed_w: shape (out_features, in_features / 8)   dtype uint32
scales:   shape (out_features, in_features / 64)   dtype float16
biases:   shape (out_features, in_features / 64)   dtype float16
```

### Dequantization Formula

To recover the float value from a quantized element:

```
float_value = scale * quantized_int + bias
```

Where `quantized_int` is extracted from the packed uint32 via bit shifting and masking.

## 4-Bit Extraction from uint32

Each `uint32` holds 8 4-bit values. Extract the k-th nibble (0-indexed):

```metal
// Extract 4-bit value at position k from packed uint32
uint packed = w_packed[packed_idx];
uint nibble = (packed >> (k * 4)) & 0xF;
float dequantized = scale * (float)nibble + bias;
```

For a row of `in_features` elements:
```metal
uint group_size = 64;
uint elem_per_packed = 8;  // 4-bit: 8 elements per uint32

for (uint i = tid; i < in_features; i += tg_size) {
    uint packed_idx = row * (in_features / elem_per_packed) + i / elem_per_packed;
    uint nibble_idx = i % elem_per_packed;
    uint packed = w_packed[packed_idx];
    uint nibble = (packed >> (nibble_idx * 4)) & 0xF;

    uint group_idx = row * (in_features / group_size) + i / group_size;
    float scale = (float)scales[group_idx];
    float bias = (float)biases[group_idx];

    float w_val = scale * (float)nibble + bias;
    // ... use w_val ...
}
```

## Template: Fused Dequant + Matrix-Vector Multiply

The dominant operation in quantized LLM inference is `y = W_q @ x`, where `W_q` is a quantized weight matrix and `x` is a float16 vector. Fusing dequantization with the dot product avoids materializing the full dequantized weight matrix.

### Why Fuse?

| Approach | Memory Read | Memory Write |
|----------|-------------|--------------|
| Separate: dequant then matmul | W_q + W_full + x | W_full + y |
| Fused: dequant-matvec | W_q + scales + biases + x | y |

The fused version reads ~4× less data for 4-bit quantization (packed weights are 8× smaller than float16, and we skip materializing the full matrix).

### Kernel Structure

Each threadgroup computes one output element (one row of W dotted with x):

```metal
uint out_row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint in_features = x_shape[0];
uint group_size = 64;

float acc = 0.0f;

// Each thread handles a stride of input elements
for (uint i = tid; i < in_features; i += threads_per_threadgroup.x) {
    // Dequantize weight element
    uint packed_idx = out_row * (in_features / 8) + i / 8;
    uint nibble_idx = i % 8;
    uint packed = w_packed[packed_idx];
    uint nibble = (packed >> (nibble_idx * 4)) & 0xF;

    uint group_idx = out_row * (in_features / group_size) + i / group_size;
    float w_val = (float)scales[group_idx] * (float)nibble + (float)biases[group_idx];

    // Multiply with input
    acc += w_val * (float)x[i];
}

// Reduce across threadgroup
acc = simd_sum(acc);
// ... cross-simdgroup reduction ...

// First thread writes result
if (tid == 0) {
    out[out_row] = (T)acc;
}
```

### Python Setup

```python
out_features = packed_w.shape[0]
tg = 256

kernel(
    inputs=[packed_w, scales, biases, x],
    template=[("T", mx.float16)],
    grid=(out_features * tg, 1, 1),
    threadgroup=(tg, 1, 1),
    output_shapes=[(out_features,)],
    output_dtypes=[mx.float16],
)
```

## When to Write Custom Quantized Kernels

MLX's built-in quantized matmul (`mx.quantized_matmul`) is already optimized for standard quantized operations. Write custom kernels when you want to:

- **Fuse dequant with another op**: e.g., dequant + RMSNorm + matmul
- **Custom quantization schemes**: Non-standard bit widths, asymmetric quantization, mixed precision per layer
- **Specialized access patterns**: Sparse quantized weights, grouped quantization with non-standard group sizes

For standard quantized inference (loading a 4-bit model from `mlx-community`), use the built-in support — it's faster and handles edge cases.

## Reference Implementation

See `scripts/dequant_matvec_kernel.py` for a working 4-bit dequantized matrix-vector multiply with correctness validation.
