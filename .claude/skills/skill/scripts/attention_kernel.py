"""Simplified single-head row-by-row attention kernel for MLX.

Demonstrates:
- Row-by-row attention with online softmax (Bk=1 simplification)
- 2D grid: Y for query row index, X for threads within row
- Float32 accumulation with float16 I/O
- Correctness validation against naive softmax(Q@K^T/sqrt(d))@V

This processes one K/V row per iteration (Bk=1), which is a simplification
of the tiled approach (Bk>1) described in attention-kernel-guide.md.
For production use, prefer mx.fast.scaled_dot_product_attention.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.utils import scalar_f32
from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Kernel: simplified tiled attention, single head
# Each threadgroup handles one query row.
# Iterates over K/V in tiles of Bk, using online softmax.
# ---------------------------------------------------------------------------
ATTENTION_SOURCE = """
uint q_row = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint seq_q = q_shape[0];
uint seq_k = k_shape[0];
uint D = q_shape[1];

if (q_row >= seq_q) return;

// Per-thread accumulators (each thread handles a subset of D)
float row_max = -1e38f;
float row_sum = 0.0f;

// We accumulate output in registers: each thread handles D elements strided
// For simplicity, we store partial output in global memory and normalize at the end
// Initialize output to 0
for (uint d = tid; d < D; d += tg_size) {
    out[q_row * D + d] = (T)0.0f;
}

// Threadgroup memory for reduction
threadgroup float shared[32];

// Iterate over K/V rows one at a time (Bk=1 for simplicity; for Bk>1,
// load a tile of K into threadgroup memory and compute a tile of scores)
for (uint kv = 0; kv < seq_k; kv++) {
    // Step 1: Compute dot product q[q_row] . k[kv] / sqrt(D)
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_row * D + d] * (float)k[kv * D + d];
    }
    // Reduce dot product across threadgroup
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < (tg_size + 31) / 32) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = shared[0] * metal::rsqrt((float)D);

    // Step 2: Online softmax update
    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score = metal::exp(score - new_max);

    // Rescale running sum and add new score
    row_sum = row_sum * correction + exp_score;

    // Step 3: Rescale existing output and add new contribution
    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[q_row * D + d];
        float v_val = (float)v[kv * D + d];
        out[q_row * D + d] = (T)(prev * correction + exp_score * v_val);
    }

    row_max = new_max;
}

// Final normalization: output /= row_sum
// No broadcast needed — score is written to shared[0] each iteration,
// so all threads already see the same running row_sum.
float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[q_row * D + d] = (T)((float)out[q_row * D + d] * inv_sum);
}
"""

_attention_kernel = mx.fast.metal_kernel(
    name="tiled_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=ATTENTION_SOURCE,
)

# ---------------------------------------------------------------------------
# Kernel: batched single-head attention (q,k,v are 3D)
# grid.y indexes rows in the flattened space: row = b*seq_q + q_row
# ---------------------------------------------------------------------------
ATTENTION_BATCHED_SOURCE = """
uint row = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint B     = q_shape[0];
uint seq_q = q_shape[1];
uint D     = q_shape[2];

uint seq_k = k_shape[1]; // k: (B, seq_k, D)

if (seq_q == 0 || D == 0) return;

uint b = row / seq_q;
uint q_row = row - b * seq_q;
if (b >= B || q_row >= seq_q) return;

// Flattened base offsets into 1D memory (row-major contiguous)
uint q_base   = (b * seq_q + q_row) * D;
uint out_base = q_base;

// Per-row online softmax accumulators
float row_max = -1e38f;
float row_sum = 0.0f;

// Initialize output row to 0
for (uint d = tid; d < D; d += tg_size) {
    out[out_base + d] = (T)0.0f;
}

// Threadgroup memory for reduction (supports up to 1024 threads => 32 simdgroups)
threadgroup float shared[32];

for (uint kv = 0; kv < seq_k; kv++) {
    uint k_base = (b * seq_k + kv) * D;

    // Dot(q_row, k_kv)
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_base + d] * (float)k[k_base + d];
    }

    // Reduce dot across threadgroup
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < (tg_size + 31) / 32) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score = shared[0] * metal::rsqrt((float)D);

    // Online softmax update
    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score  = metal::exp(score - new_max);

    row_sum = row_sum * correction + exp_score;

    // Update output accumulation
    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[out_base + d];
        float vval = (float)v[k_base + d];
        out[out_base + d] = (T)(prev * correction + exp_score * vval);
    }

    row_max = new_max;
}

// Normalize
float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[out_base + d] = (T)((float)out[out_base + d] * inv_sum);
}
"""

_attention_batched_kernel = mx.fast.metal_kernel(
    name="tiled_attention_batched",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=ATTENTION_BATCHED_SOURCE,
)


def attention_batched(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Batched single-head attention using a 3D-aware Metal kernel.

    Args:
        q: (B, seq_q, D)
        k: (B, seq_k, D)
        v: (B, seq_k, D)

    Returns:
        (B, seq_q, D)
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"attention_batched expects 3D q,k,v. Got q{q.shape}, k{k.shape}, v{v.shape}")
    if k.shape != v.shape:
        raise ValueError(f"k and v must match, got k{k.shape} vs v{v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"Batch/D mismatch: q{q.shape}, k{k.shape}, v{v.shape}")

    B, seq_q, D = map(int, q.shape)
    seq_k = int(k.shape[1])
    shape_sig = f"B={B},seq_q={seq_q},seq_k={seq_k},d={D}"

    def _run(tgx: int):
        tgx = max(32, (tgx // 32) * 32)
        tgx = min(tgx, 1024)
        return _attention_batched_kernel(
            inputs=[q, k, v],
            template=[("T", q.dtype)],
            grid=(1, B * seq_q, 1),
            threadgroup=(tgx, 1, 1),
            output_shapes=[(B, seq_q, D)],
            output_dtypes=[q.dtype],
        )[0]

    tg = pick_threadgroup(
        kernel_name="attention_single_head_batched",
        shape_sig=shape_sig,
        dtype_sig=str(q.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=min(256, D),
    )

    return _run(tg)


def attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Single-head scaled dot-product attention using a custom Metal kernel.

    Args:
        q: (seq_q, d)
        k: (seq_k, d)
        v: (seq_k, d)
    """
    if q.ndim == 3:
        return attention_batched(q, k, v)

    # 2D mode: q,k,v are (seq, d)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(
            f"attention() expects 2D inputs (seq, d) or 3D batched inputs (B, seq, d). "
            f"Got q{q.shape}, k{k.shape}, v{v.shape}."
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must have same shape, got k{k.shape} vs v{v.shape}")
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"q and k must have same d, got q{q.shape} vs k{k.shape}")

    seq_q, d = map(int, q.shape)
    seq_k = int(k.shape[0])

    shape_sig = f"seq_q={seq_q},seq_k={seq_k},d={d}"

    def _run(tgx: int):
        tgx = max(32, (tgx // 32) * 32)
        return _attention_kernel(
            inputs=[q, k, v],
            template=[("T", q.dtype)],
            grid=(1, seq_q, 1),              # Y dimension = query rows
            threadgroup=(tgx, 1, 1),
            output_shapes=[(seq_q, d)],
            output_dtypes=[q.dtype],
        )[0]

    tg = pick_threadgroup(
        kernel_name="attention_single_head",
        shape_sig=shape_sig,
        dtype_sig=str(q.dtype),
        candidates=[32, 64, 128, 256, 512, 1024],
        run=_run,
        default=min(256, d),
    )

    out = _run(tg)
    return out


def attention_reference(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Naive attention: softmax(Q @ K^T / sqrt(d)) @ V."""
    # Support both 2D and 3D batched shapes.
    if q.ndim == 3:
        outs = [attention_reference(q[b], k[b], v[b]) for b in range(int(q.shape[0]))]
        return mx.stack(outs, axis=0)
    d = q.shape[-1]
    scale = mx.rsqrt(scalar_f32(float(d)))
    scores = (q @ k.T) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v


def validate(configs=None, dtype=mx.float32):
    """Validate custom attention against naive reference."""
    if configs is None:
        configs = [
            # (seq_q, seq_k, d)
            (1, 1, 32),
            (4, 4, 64),
            (8, 8, 128),
            (16, 32, 128),
            (32, 64, 64),
            (1, 16, 128),
        ]

    print(f"Validating attention kernel (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    for seq_q, seq_k, d in configs:
        q = mx.random.normal((seq_q, d)).astype(dtype) * 0.1
        k = mx.random.normal((seq_k, d)).astype(dtype) * 0.1
        v = mx.random.normal((seq_k, d)).astype(dtype) * 0.1

        expected = attention_reference(q, k, v)
        actual = attention(q, k, v)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        atol = 5e-2 if dtype == mx.float16 else 1e-4
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        label = f"q=({seq_q},{d}) k=({seq_k},{d})"
        print(f"  {label:>30s}  max_diff={max_diff:.2e}  [{status}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== Tiled Attention Metal Kernel Demo ===\n")

    # Quick demo
    seq_q, seq_k, d = 16, 32, 128
    q = mx.random.normal((seq_q, d)) * 0.1
    k = mx.random.normal((seq_k, d)) * 0.1
    v = mx.random.normal((seq_k, d)) * 0.1

    out = attention(q, k, v)
    mx.eval(out)
    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {mx.mean(out).item():.6f}")
    print()

    # Validate correctness
    validate(dtype=mx.float32)
    print()
    validate(dtype=mx.float16)
