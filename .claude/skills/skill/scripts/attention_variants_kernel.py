"""Multi-head attention variants: causal, sliding window, and GQA.

Demonstrates:
- 3D grid dispatch: X=threads per row, Y=query row, Z=head index
- Causal masking by bounding the K/V loop (skip future positions)
- Sliding window attention (Mistral-style) with template int parameter
- Grouped-query attention (GQA) where multiple Q heads share K/V heads
- Online softmax for all variants
- Correctness validation against pure-MLX references

Builds on the single-head attention pattern in attention_kernel.py.
For production use, prefer mx.fast.scaled_dot_product_attention.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from kernels.autotune_cache import pick_threadgroup

# ---------------------------------------------------------------------------
# Causal multi-head attention: each query only attends to positions <= its own
# Input shapes: Q, K, V are (heads, seq, dim)
# ---------------------------------------------------------------------------
CAUSAL_ATTENTION_SOURCE = """
uint q_row = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint seq_q = q_shape[1];
uint seq_k = k_shape[1];
uint D = q_shape[2];

if (q_row >= seq_q) return;

uint q_base = head * seq_q * D + q_row * D;
uint k_head_base = head * seq_k * D;
uint v_head_base = head * seq_k * D;

float row_max = -1e38f;
float row_sum = 0.0f;

// Initialize output to 0
for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)0.0f;
}

threadgroup float shared[32];
uint num_sg = (tg_size + 31) / 32;

// Causal: only attend to positions <= q_row
uint kv_end = min(q_row + 1, seq_k);
for (uint kv = 0; kv < kv_end; kv++) {
    // Dot product: q[q_row] . k[kv] / sqrt(D)
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_base + d] * (float)k[k_head_base + kv * D + d];
    }
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < num_sg) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = shared[0] * metal::rsqrt((float)D);

    // Online softmax update
    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score = metal::exp(score - new_max);
    row_sum = row_sum * correction + exp_score;

    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[q_base + d];
        float v_val = (float)v[v_head_base + kv * D + d];
        out[q_base + d] = (T)(prev * correction + exp_score * v_val);
    }
    row_max = new_max;
}

// Final normalization
float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)((float)out[q_base + d] * inv_sum);
}
"""

# ---------------------------------------------------------------------------
# Sliding window attention: attend to positions in [q_row - WINDOW + 1, q_row]
# WINDOW is a compile-time template int parameter
# ---------------------------------------------------------------------------
SLIDING_WINDOW_SOURCE = """
uint q_row = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint seq_q = q_shape[1];
uint seq_k = k_shape[1];
uint D = q_shape[2];

if (q_row >= seq_q) return;

uint q_base = head * seq_q * D + q_row * D;
uint k_head_base = head * seq_k * D;
uint v_head_base = head * seq_k * D;

float row_max = -1e38f;
float row_sum = 0.0f;

for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)0.0f;
}

threadgroup float shared[32];
uint num_sg = (tg_size + 31) / 32;

// Sliding window: attend to [max(0, q_row - WINDOW + 1), q_row]
// Bounding the loop saves compute: O(n*W) instead of O(n^2)
uint kv_start = (q_row >= (uint)(WINDOW - 1)) ? (q_row - WINDOW + 1) : 0;
uint kv_end = min(q_row + 1, seq_k);

for (uint kv = kv_start; kv < kv_end; kv++) {
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_base + d] * (float)k[k_head_base + kv * D + d];
    }
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < num_sg) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = shared[0] * metal::rsqrt((float)D);

    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score = metal::exp(score - new_max);
    row_sum = row_sum * correction + exp_score;

    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[q_base + d];
        float v_val = (float)v[v_head_base + kv * D + d];
        out[q_base + d] = (T)(prev * correction + exp_score * v_val);
    }
    row_max = new_max;
}

float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)((float)out[q_base + d] * inv_sum);
}
"""

# ---------------------------------------------------------------------------
# Grouped-query attention (GQA): multiple Q heads share K/V heads
# Q shape: (num_q_heads, seq, dim), K/V shape: (num_kv_heads, seq, dim)
# Q_PER_KV = num_q_heads / num_kv_heads (template int parameter)
# ---------------------------------------------------------------------------
GQA_SOURCE = """
uint q_row = threadgroup_position_in_grid.y;
uint q_head = threadgroup_position_in_grid.z;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint seq_q = q_shape[1];
uint seq_k = k_shape[1];
uint D = q_shape[2];

if (q_row >= seq_q) return;

// Key insight: multiple Q heads share one K/V head
uint kv_head = q_head / Q_PER_KV;

uint q_base = q_head * seq_q * D + q_row * D;
uint k_head_base = kv_head * seq_k * D;
uint v_head_base = kv_head * seq_k * D;

float row_max = -1e38f;
float row_sum = 0.0f;

for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)0.0f;
}

threadgroup float shared[32];
uint num_sg = (tg_size + 31) / 32;

// Full (non-causal) attention — combine with causal mask as needed
for (uint kv = 0; kv < seq_k; kv++) {
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_base + d] * (float)k[k_head_base + kv * D + d];
    }
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < num_sg) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = shared[0] * metal::rsqrt((float)D);

    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score = metal::exp(score - new_max);
    row_sum = row_sum * correction + exp_score;

    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[q_base + d];
        float v_val = (float)v[v_head_base + kv * D + d];
        out[q_base + d] = (T)(prev * correction + exp_score * v_val);
    }
    row_max = new_max;
}

float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)((float)out[q_base + d] * inv_sum);
}
"""

# ---------------------------------------------------------------------------
# GQA + causal attention (Llama 2/3 pattern): the most common real-world combo
# Combines GQA head mapping with causal loop bounding
# ---------------------------------------------------------------------------
GQA_CAUSAL_SOURCE = """
uint q_row = threadgroup_position_in_grid.y;
uint q_head = threadgroup_position_in_grid.z;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;

uint seq_q = q_shape[1];
uint seq_k = k_shape[1];
uint D = q_shape[2];

if (q_row >= seq_q) return;

// GQA: multiple Q heads share one K/V head
uint kv_head = q_head / Q_PER_KV;

uint q_base = q_head * seq_q * D + q_row * D;
uint k_head_base = kv_head * seq_k * D;
uint v_head_base = kv_head * seq_k * D;

float row_max = -1e38f;
float row_sum = 0.0f;

for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)0.0f;
}

threadgroup float shared[32];
uint num_sg = (tg_size + 31) / 32;

// Causal + GQA: only attend to positions <= q_row, index K/V by kv_head
uint kv_end = min(q_row + 1, seq_k);
for (uint kv = 0; kv < kv_end; kv++) {
    float dot = 0.0f;
    for (uint d = tid; d < D; d += tg_size) {
        dot += (float)q[q_base + d] * (float)k[k_head_base + kv * D + d];
    }
    dot = simd_sum(dot);
    if (lane == 0) shared[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float p = (lane < num_sg) ? shared[lane] : 0.0f;
        shared[0] = simd_sum(p);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = shared[0] * metal::rsqrt((float)D);

    float new_max = max(row_max, score);
    float correction = metal::exp(row_max - new_max);
    float exp_score = metal::exp(score - new_max);
    row_sum = row_sum * correction + exp_score;

    for (uint d = tid; d < D; d += tg_size) {
        float prev = (float)out[q_base + d];
        float v_val = (float)v[v_head_base + kv * D + d];
        out[q_base + d] = (T)(prev * correction + exp_score * v_val);
    }
    row_max = new_max;
}

float inv_sum = 1.0f / row_sum;
for (uint d = tid; d < D; d += tg_size) {
    out[q_base + d] = (T)((float)out[q_base + d] * inv_sum);
}
"""

# Compile kernels once at module level
_causal_kernel = mx.fast.metal_kernel(
    name="causal_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=CAUSAL_ATTENTION_SOURCE,
)

_sliding_window_kernel = mx.fast.metal_kernel(
    name="sliding_window_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=SLIDING_WINDOW_SOURCE,
)

_gqa_kernel = mx.fast.metal_kernel(
    name="gqa_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=GQA_SOURCE,
)

_gqa_causal_kernel = mx.fast.metal_kernel(
    name="gqa_causal_attention",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=GQA_CAUSAL_SOURCE,
)


def causal_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Multi-head causal attention. Each query attends only to past positions.

    Args:
        q, k, v: Shape (heads, seq, dim).

    Returns:
        Attention output of shape (heads, seq, dim).
    """
    H, S, D = q.shape
    def _run(tgx: int):

        tgx = max(32, (tgx // 32) * 32)

        return _causal_kernel(

            inputs=[q, k, v],

            template=[("T", q.dtype)],

            grid=(tgx, S, H),

            threadgroup=(tgx, 1, 1),

            output_shapes=[(H, S, D)],

            output_dtypes=[q.dtype],

        )[0]


    tg = pick_threadgroup(

        kernel_name="attention_causal_mh",

        shape_sig=f"D={D}",

        dtype_sig=str(q.dtype),

        candidates=[32, 64, 128, 256, 512, 1024],

        run=_run,

        default=min(256, D),

    )

    return _run(tg)


def sliding_window_attention(
    q: mx.array, k: mx.array, v: mx.array, window_size: int
) -> mx.array:
    """Multi-head sliding window attention (Mistral-style).

    Each query attends to at most `window_size` past positions.

    Args:
        q, k, v: Shape (heads, seq, dim).
        window_size: Number of positions to attend to.

    Returns:
        Attention output of shape (heads, seq, dim).
    """
    H, S, D = q.shape
    def _run(tgx: int):

        tgx = max(32, (tgx // 32) * 32)

        return _sliding_window_kernel(

            inputs=[q, k, v],

            template=[("T", q.dtype), ("WINDOW", window_size)],

            grid=(tgx, S, H),

            threadgroup=(tgx, 1, 1),

            output_shapes=[(H, S, D)],

            output_dtypes=[q.dtype],

        )[0]


    tg = pick_threadgroup(

        kernel_name="attention_sliding_window",

        shape_sig=f"D={D}",

        dtype_sig=str(q.dtype),

        candidates=[32, 64, 128, 256, 512, 1024],

        run=_run,

        default=min(256, D),

    )

    return _run(tg)


def gqa_attention(
    q: mx.array, k: mx.array, v: mx.array, num_kv_heads: int
) -> mx.array:
    """Grouped-query attention. Multiple Q heads share each K/V head.

    Args:
        q: Shape (num_q_heads, seq, dim).
        k, v: Shape (num_kv_heads, seq, dim).
        num_kv_heads: Number of K/V heads.

    Returns:
        Attention output of shape (num_q_heads, seq, dim).
    """
    num_q_heads, S, D = q.shape
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    q_per_kv = num_q_heads // num_kv_heads

    def _run(tgx: int):


        tgx = max(32, (tgx // 32) * 32)


        return _gqa_kernel(


            inputs=[q, k, v],


            template=[("T", q.dtype), ("Q_PER_KV", q_per_kv)],


            grid=(tgx, S, num_q_heads),


            threadgroup=(tgx, 1, 1),


            output_shapes=[(num_q_heads, S, D)],


            output_dtypes=[q.dtype],


        )[0]



    tg = pick_threadgroup(


        kernel_name="attention_gqa",


        shape_sig=f"D={D}",


        dtype_sig=str(q.dtype),


        candidates=[32, 64, 128, 256, 512, 1024],


        run=_run,


        default=min(256, D),


    )


    return _run(tg)


def gqa_causal_attention(
    q: mx.array, k: mx.array, v: mx.array, num_kv_heads: int
) -> mx.array:
    """GQA with causal masking — the Llama 2/3 pattern.

    Combines grouped-query attention with causal masking, the most common
    attention configuration in modern LLMs.

    Args:
        q: Shape (num_q_heads, seq, dim).
        k, v: Shape (num_kv_heads, seq, dim).
        num_kv_heads: Number of K/V heads.

    Returns:
        Attention output of shape (num_q_heads, seq, dim).
    """
    num_q_heads, S, D = q.shape
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    q_per_kv = num_q_heads // num_kv_heads

    def _run(tgx: int):


        tgx = max(32, (tgx // 32) * 32)


        return _gqa_causal_kernel(


            inputs=[q, k, v],


            template=[("T", q.dtype), ("Q_PER_KV", q_per_kv)],


            grid=(tgx, S, num_q_heads),


            threadgroup=(tgx, 1, 1),


            output_shapes=[(num_q_heads, S, D)],


            output_dtypes=[q.dtype],


        )[0]



    tg = pick_threadgroup(


        kernel_name="attention_gqa_causal",


        shape_sig=f"D={D}",


        dtype_sig=str(q.dtype),


        candidates=[32, 64, 128, 256, 512, 1024],


        run=_run,


        default=min(256, D),


    )


    return _run(tg)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _causal_attention_ref(q, k, v):
    """Reference: apply triangular causal mask, then softmax, then V."""
    H, S_q, D = q.shape
    S_k = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    scores = (q @ mx.transpose(k, axes=[0, 2, 1])) * scale  # (H, S_q, S_k)
    # Causal mask: -inf for positions where key > query
    mask = mx.triu(mx.full((S_q, S_k), -1e9), k=1)
    scores = scores + mask
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v


def _sliding_window_ref(q, k, v, window_size):
    """Reference: band mask for sliding window."""
    H, S_q, D = q.shape
    S_k = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    scores = (q @ mx.transpose(k, axes=[0, 2, 1])) * scale

    # Mask: allow only positions in [q_pos - window + 1, q_pos]
    q_idx = mx.arange(S_q)[:, None]
    k_idx = mx.arange(S_k)[None, :]
    mask = mx.where(
        (k_idx <= q_idx) & (k_idx > q_idx - window_size),
        mx.array(0.0),
        mx.array(-1e9),
    )
    scores = scores + mask
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v


def _gqa_attention_ref(q, k, v, num_kv_heads):
    """Reference: repeat K/V heads to match Q, then standard attention."""
    num_q_heads = q.shape[0]
    q_per_kv = num_q_heads // num_kv_heads

    # Repeat K/V heads to match number of Q heads
    k_expanded = mx.repeat(k, q_per_kv, axis=0)  # (num_q_heads, seq, dim)
    v_expanded = mx.repeat(v, q_per_kv, axis=0)

    D = q.shape[-1]
    scale = 1.0 / (D ** 0.5)
    scores = (q @ mx.transpose(k_expanded, axes=[0, 2, 1])) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v_expanded


def _gqa_causal_attention_ref(q, k, v, num_kv_heads):
    """Reference: repeat K/V heads, apply causal mask, softmax, matmul."""
    num_q_heads = q.shape[0]
    q_per_kv = num_q_heads // num_kv_heads
    S_q, S_k, D = q.shape[1], k.shape[1], q.shape[2]

    k_expanded = mx.repeat(k, q_per_kv, axis=0)
    v_expanded = mx.repeat(v, q_per_kv, axis=0)

    scale = 1.0 / (D ** 0.5)
    scores = (q @ mx.transpose(k_expanded, axes=[0, 2, 1])) * scale
    mask = mx.triu(mx.full((S_q, S_k), -1e9), k=1)
    scores = scores + mask
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v_expanded


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_causal(dtype=mx.float32):
    """Validate causal attention."""
    configs = [
        (4, 16, 64),
        (8, 32, 128),
        (2, 8, 32),
        (1, 1, 64),
    ]
    print(f"Validating causal attention (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    atol = 5e-2 if dtype == mx.float16 else 1e-4

    for H, S, D in configs:
        q = mx.random.normal((H, S, D)).astype(dtype) * 0.1
        k = mx.random.normal((H, S, D)).astype(dtype) * 0.1
        v = mx.random.normal((H, S, D)).astype(dtype) * 0.1

        expected = _causal_attention_ref(q, k, v)
        actual = causal_attention(q, k, v)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label = f"({H},{S},{D})"
        print(f"  {label:>20s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def validate_sliding_window(dtype=mx.float32):
    """Validate sliding window attention."""
    configs = [
        (4, 16, 64, 4),
        (4, 16, 64, 8),
        (8, 32, 128, 8),
        (2, 8, 32, 3),
    ]
    print(f"Validating sliding window attention (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    atol = 5e-2 if dtype == mx.float16 else 1e-4

    for H, S, D, W in configs:
        q = mx.random.normal((H, S, D)).astype(dtype) * 0.1
        k = mx.random.normal((H, S, D)).astype(dtype) * 0.1
        v = mx.random.normal((H, S, D)).astype(dtype) * 0.1

        expected = _sliding_window_ref(q, k, v, W)
        actual = sliding_window_attention(q, k, v, W)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label = f"({H},{S},{D}) W={W}"
        print(f"  {label:>25s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def validate_gqa(dtype=mx.float32):
    """Validate grouped-query attention."""
    configs = [
        # (num_q_heads, num_kv_heads, seq, dim)
        (8, 2, 16, 64),
        (8, 1, 16, 64),
        (4, 2, 32, 128),
        (16, 4, 8, 64),
    ]
    print(f"Validating GQA attention (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    atol = 5e-2 if dtype == mx.float16 else 1e-4

    for Hq, Hkv, S, D in configs:
        q = mx.random.normal((Hq, S, D)).astype(dtype) * 0.1
        k = mx.random.normal((Hkv, S, D)).astype(dtype) * 0.1
        v = mx.random.normal((Hkv, S, D)).astype(dtype) * 0.1

        expected = _gqa_attention_ref(q, k, v, Hkv)
        actual = gqa_attention(q, k, v, Hkv)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label = f"Hq={Hq} Hkv={Hkv} S={S} D={D}"
        print(f"  {label:>30s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def validate_gqa_causal(dtype=mx.float32):
    """Validate GQA + causal attention."""
    configs = [
        # (num_q_heads, num_kv_heads, seq, dim)
        (8, 2, 16, 64),
        (8, 1, 16, 64),
        (32, 4, 32, 128),
    ]
    print(f"Validating GQA + causal attention (dtype={dtype})")
    print("-" * 65)

    all_passed = True
    atol = 5e-2 if dtype == mx.float16 else 1e-4

    for Hq, Hkv, S, D in configs:
        q = mx.random.normal((Hq, S, D)).astype(dtype) * 0.1
        k = mx.random.normal((Hkv, S, D)).astype(dtype) * 0.1
        v = mx.random.normal((Hkv, S, D)).astype(dtype) * 0.1

        expected = _gqa_causal_attention_ref(q, k, v, Hkv)
        actual = gqa_causal_attention(q, k, v, Hkv)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        if not passed:
            all_passed = False
        label = f"Hq={Hq} Hkv={Hkv} S={S} D={D}"
        print(f"  {label:>30s}  max_diff={max_diff:.2e}  [{'PASS' if passed else 'FAIL'}]")

    print("-" * 65)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("=== Attention Variants Metal Kernel Demo ===\n")

    # Quick demos
    H, S, D = 8, 16, 64
    q = mx.random.normal((H, S, D)) * 0.1
    k = mx.random.normal((H, S, D)) * 0.1
    v = mx.random.normal((H, S, D)) * 0.1

    out_c = causal_attention(q, k, v)
    out_sw = sliding_window_attention(q, k, v, window_size=4)
    mx.eval(out_c, out_sw)
    print(f"Causal attention:        {q.shape} -> {out_c.shape}")
    print(f"Sliding window (W=4):    {q.shape} -> {out_sw.shape}")

    # GQA: 8 query heads, 2 KV heads
    k_gqa = mx.random.normal((2, S, D)) * 0.1
    v_gqa = mx.random.normal((2, S, D)) * 0.1
    out_gqa = gqa_attention(q, k_gqa, v_gqa, num_kv_heads=2)
    out_gqa_c = gqa_causal_attention(q, k_gqa, v_gqa, num_kv_heads=2)
    mx.eval(out_gqa, out_gqa_c)
    print(f"GQA (Hq=8, Hkv=2):      Q{q.shape} K{k_gqa.shape} -> {out_gqa.shape}")
    print(f"GQA+causal (Llama):      Q{q.shape} K{k_gqa.shape} -> {out_gqa_c.shape}")
    print()

    # Validate all variants
    all_ok = True
    all_ok &= validate_causal(dtype=mx.float32)
    print()
    all_ok &= validate_causal(dtype=mx.float16)
    print()
    all_ok &= validate_sliding_window(dtype=mx.float32)
    print()
    all_ok &= validate_sliding_window(dtype=mx.float16)
    print()
    all_ok &= validate_gqa(dtype=mx.float32)
    print()
    all_ok &= validate_gqa(dtype=mx.float16)
    print()
    all_ok &= validate_gqa_causal(dtype=mx.float32)
    print()
    all_ok &= validate_gqa_causal(dtype=mx.float16)

    if not all_ok:
        raise SystemExit(1)
