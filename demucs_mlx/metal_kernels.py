"""
Custom fused Metal kernels for Demucs-MLX inference.

Each kernel includes:
- Metal source code
- Python wrapper with shape handling
- float32 accumulation for numerical stability

When Metal is not available (e.g., MLX on Linux/CPU), all functions
fall back to equivalent pure-MLX operations automatically.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _has_metal() -> bool:
    """Check whether Metal GPU backend is available."""
    try:
        metal = getattr(mx, "metal", None)
        if metal is None:
            return False
        # mx.metal.is_available() is the canonical check
        is_avail = getattr(metal, "is_available", None)
        if callable(is_avail):
            return bool(is_avail())
        # If the module exists but has no is_available, assume Metal is present
        # (older MLX versions on macOS).
        return True
    except Exception:
        return False


HAS_METAL = _has_metal()

# ==============================================================================
# Fused GLU: a * sigmoid(b) where [a, b] = split(x, 2, axis)
# ==============================================================================
_GLU_SOURCE = r"""
uint k = thread_position_in_grid.x;
uint mc = thread_position_in_grid.y;

uint K = params[0];
uint C = params[1];
uint MC = params[2];

if (k >= K || mc >= MC) return;

uint m = mc / C;
uint c = mc % C;

uint in_base = m * (2 * C * K) + k;
uint in_idx_a = in_base + c * K;
uint in_idx_b = in_base + (c + C) * K;

float a = (float)x[in_idx_a];
float b = (float)x[in_idx_b];
float sig_b = 1.0f / (1.0f + metal::exp(-b));

out[mc * K + k] = (T)(a * sig_b);
"""

_GLU_VEC4_SOURCE = r"""
uint k = thread_position_in_grid.x;
uint mc = thread_position_in_grid.y;

uint K4 = params[0];
uint C = params[1];
uint MC = params[2];

if (k >= K4 || mc >= MC) return;

uint m = mc / C;
uint c = mc % C;

uint in_base = m * (2 * C * K4) + k;
uint in_idx_a = in_base + c * K4;
uint in_idx_b = in_base + (c + C) * K4;

const device vec<T, 4>* x_vec = (const device vec<T, 4>*)x;
device vec<T, 4>* out_vec = (device vec<T, 4>*)out;

vec<T, 4> a = x_vec[in_idx_a];
vec<T, 4> b = x_vec[in_idx_b];
vec<float, 4> b_f = (vec<float, 4>)b;
vec<float, 4> sig_b = 1.0f / (1.0f + metal::exp(-b_f));

out_vec[mc * K4 + k] = (vec<T, 4>)((vec<float, 4>)a * sig_b);
"""

_glu_kernel = None
_glu_vec4_kernel = None


def _get_glu_kernel():
    global _glu_kernel
    if _glu_kernel is None:
        _glu_kernel = mx.fast.metal_kernel(
            name="fused_glu",
            input_names=["x", "params"],
            output_names=["out"],
            source=_GLU_SOURCE,
        )
    return _glu_kernel


def _get_glu_vec4_kernel():
    global _glu_vec4_kernel
    if _glu_vec4_kernel is None:
        _glu_vec4_kernel = mx.fast.metal_kernel(
            name="fused_glu_vec4",
            input_names=["x", "params"],
            output_names=["out"],
            source=_GLU_VEC4_SOURCE,
        )
    return _glu_vec4_kernel


def fused_glu(x: mx.array, axis: int = 1) -> mx.array:
    """Zero-transpose fused GLU: split x in half along axis, compute a * sigmoid(b).

    Operates directly on native tensor memory layout with 2D coalesced SIMD access
    and 128-bit vectorization, eliminating transposition and contiguous memory
    allocation overhead.
    """
    if not HAS_METAL:
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    ndim = x.ndim
    axis = axis % ndim

    shape = list(x.shape)
    if shape[axis] % 2 != 0:
        raise ValueError(f"Axis {axis} size must be even, got {shape[axis]}")

    x = mx.contiguous(x)

    M = 1
    for s in shape[:axis]:
        M *= s
    C = shape[axis] // 2
    K = 1
    for s in shape[axis + 1 :]:
        K *= s

    MC = M * C
    total_out = MC * K
    out_shape = shape[:axis] + [C] + shape[axis + 1 :]

    # Vectorized 128-bit load/store when inner dimension is divisible by 4
    if K % 4 == 0 and x.size % 4 == 0:
        K4 = K // 4
        params = mx.array([K4, C, MC], dtype=mx.int32)
        tg_x = min(256, K4)
        if tg_x >= 32:
            tg_x = (tg_x // 32) * 32
        tg_y = min(256 // max(1, tg_x), MC)
        tg_y = max(1, tg_y)
        kernel = _get_glu_vec4_kernel()
        result = kernel(
            inputs=[x, params],
            template=[("T", x.dtype)],
            grid=(K4, MC, 1),
            threadgroup=(max(1, tg_x), tg_y, 1),
            output_shapes=[(total_out,)],
            output_dtypes=[x.dtype],
        )[0]
    else:
        params = mx.array([K, C, MC], dtype=mx.int32)
        tg_x = min(256, K) if K > 0 else 1
        if tg_x >= 32:
            tg_x = (tg_x // 32) * 32
        tg_y = min(256 // max(1, tg_x), MC) if tg_x > 0 else 1
        tg_y = max(1, tg_y)
        kernel = _get_glu_kernel()
        result = kernel(
            inputs=[x, params],
            template=[("T", x.dtype)],
            grid=(K, MC, 1),
            threadgroup=(max(1, tg_x), tg_y, 1),
            output_shapes=[(total_out,)],
            output_dtypes=[x.dtype],
        )[0]

    return result.reshape(out_shape)



# Threshold: when elems_per_group exceeds this, use pure MLX ops
# instead of the single-threadgroup Metal kernel (which underutilizes the GPU).
_HYBRID_THRESHOLD = 32768


# ==============================================================================
# Fused GroupNorm + GELU (erf-based, approximating MLX's nn.gelu -- see the
# A&S note below; it is not exact)
# ==============================================================================
# Each threadgroup handles one (batch, group) pair.
# Uses simdgroup reductions for mean/variance.
# GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
# MSL provides no erf in any namespace (metal::erf, erf and
# metal::precise::erf all fail to compile, with or without
# <metal_math>), so an approximation is required. A&S 7.1.26 is
# accurate to ~1.5e-7.

_GROUPNORM_GELU_HEADER = r"""
// Abramowitz & Stegun approximation of erf, max error ~1.5e-7
inline float erf_approx(float x) {
    // erf(-x) = -erf(x)
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float a = metal::abs(x);
    // A&S formula 7.1.26
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float poly = 0.254829592f * t
               - 0.284496736f * t2
               + 1.421413741f * t3
               - 1.453152027f * t4
               + 1.061405429f * t5;
    float result = 1.0f - poly * metal::exp(-a * a);
    return sign * result;
}
"""

_GROUPNORM_GELU_SOURCE = r"""
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;
// Every simdgroup reads slot 0 above, and pass 2 below has simdgroup 0 write
// that same slot. Without this barrier a fast simdgroup can clobber the mean
// before a lagging one has loaded it, poisoning the whole (batch, group) slab.
// Widest window at small elems_per_group -- the freq-branch DConv shapes.
threadgroup_barrier(mem_flags::mem_threadgroup);

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Pass 3: Normalize, apply affine, apply erf-based GELU
float rsqrt2 = 0.7071067811865475f;  // 1/sqrt(2)
for (uint i = tid; i < elems_per_group; i += tg_size) {
    uint c_local = i / spatial_size;
    uint c_global = group_idx * channels_per_group + c_local;
    float val = ((float)x[base + i] - mean) * inv_std;
    val = val * (float)weight[c_global] + (float)bias[c_global];
    // GELU via the A&S erf approximation above, NOT mx.erf: ~1e-7 from the
    // unfused path, which calls mx.erf.
    val = 0.5f * val * (1.0f + erf_approx(val * rsqrt2));
    out[base + i] = (T)val;
}
"""

_groupnorm_gelu_kernel = None


def _get_groupnorm_gelu_kernel():
    global _groupnorm_gelu_kernel
    if _groupnorm_gelu_kernel is None:
        _groupnorm_gelu_kernel = mx.fast.metal_kernel(
            name="fused_groupnorm_gelu",
            input_names=["x", "weight", "bias", "eps", "params"],
            output_names=["out"],
            header=_GROUPNORM_GELU_HEADER,
            source=_GROUPNORM_GELU_SOURCE,
        )
    return _groupnorm_gelu_kernel


@mx.compile
def _groupnorm_gelu_fallback(
    x: mx.array, weight: mx.array, bias: mx.array,
    num_groups: int, eps: float = 1e-5,
) -> mx.array:
    """Pure-MLX GroupNorm + GELU (no Metal required)."""
    B, C = x.shape[0], x.shape[1]
    cpg = C // num_groups
    x_r = x.reshape(B, num_groups, cpg, *x.shape[2:])
    axes = tuple(range(2, x_r.ndim))
    mean = x_r.mean(axis=axes, keepdims=True)
    var = mx.var(x_r, axis=axes, keepdims=True)
    x_norm = (x_r - mean) * mx.rsqrt(var + eps)
    x_out = x_norm.reshape(x.shape)
    w_shape = [1, C] + [1] * (x.ndim - 2)
    x_out = x_out * weight.reshape(w_shape) + bias.reshape(w_shape)
    return nn.gelu(x_out)


def fused_groupnorm_gelu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int,
    eps: float = 1e-5,
) -> mx.array:
    """Fused GroupNorm + GELU for NCL or NCHW layout.

    Equivalent to:
        x = groupnorm(x, weight, bias, num_groups, eps)
        x = gelu(x)
    """
    if not HAS_METAL:
        return _groupnorm_gelu_fallback(x, weight, bias, num_groups, eps)

    orig_shape = x.shape
    B = x.shape[0]
    C = x.shape[1]
    spatial_size = 1
    for d in x.shape[2:]:
        spatial_size *= d

    if C % num_groups != 0:
        raise ValueError(f"channels {C} not divisible by num_groups {num_groups}")
    channels_per_group = C // num_groups
    elems_per_group = channels_per_group * spatial_size

    # For large groups (few threadgroups), use pure MLX ops which parallelize
    # better than the single-threadgroup Metal kernel.
    if elems_per_group > _HYBRID_THRESHOLD:
        return _groupnorm_gelu_fallback(x, weight, bias, num_groups, eps)

    x_contig = mx.contiguous(x.reshape(B, C, spatial_size))
    weight = mx.contiguous(weight.astype(mx.float32))
    bias = mx.contiguous(bias.astype(mx.float32))
    eps_arr = mx.array([eps], dtype=mx.float32)
    params = mx.array([num_groups, channels_per_group, spatial_size, C], dtype=mx.int32)

    total_groups = B * num_groups
    tg = min(1024, max(32, ((elems_per_group + 31) // 32) * 32))

    result = _get_groupnorm_gelu_kernel()(
        inputs=[x_contig, weight, bias, eps_arr, params],
        template=[("T", x.dtype)],
        grid=(total_groups * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, C, spatial_size)],
        output_dtypes=[x.dtype],
    )[0]

    return result.reshape(orig_shape)


# ==============================================================================
# Fused GroupNorm + GLU (GroupNorm over 2C channels, then split+sigmoid+mul)
# ==============================================================================
# Each threadgroup handles one (batch, group) pair.
# GroupNorm normalizes over 2C channels, then GLU splits into a (first C) and
# b (second C), computing output = a * sigmoid(b). Output has half the channels.

_GROUPNORM_GLU_SOURCE = r"""
uint bg = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint sid = thread_index_in_simdgroup;
uint wid = simdgroup_index_in_threadgroup;
uint num_simdgroups = tg_size / 32;

uint num_groups = params[0];
uint channels_per_group = params[1];
uint spatial_size = params[2];
uint total_channels = params[3];
uint half_channels = params[4];

uint batch_idx = bg / num_groups;
uint group_idx = bg % num_groups;

uint elems_per_group = channels_per_group * spatial_size;

uint base = batch_idx * total_channels * spatial_size
          + group_idx * channels_per_group * spatial_size;

// Pass 1: Compute mean
float local_sum = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    local_sum += (float)x[base + i];
}
local_sum = simd_sum(local_sum);

threadgroup float shared_sums[32];
if (sid == 0) shared_sums[wid] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean = shared_sums[0] / (float)elems_per_group;
// Every simdgroup reads slot 0 above, and pass 2 below has simdgroup 0 write
// that same slot. Without this barrier a fast simdgroup can clobber the mean
// before a lagging one has loaded it, poisoning the whole (batch, group) slab.
// Widest window at small elems_per_group -- the freq-branch DConv shapes.
threadgroup_barrier(mem_flags::mem_threadgroup);

// Pass 2: Compute variance
float local_var = 0.0f;
for (uint i = tid; i < elems_per_group; i += tg_size) {
    float diff = (float)x[base + i] - mean;
    local_var += diff * diff;
}
local_var = simd_sum(local_var);
if (sid == 0) shared_sums[wid] = local_var;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (wid == 0) {
    float val = (sid < num_simdgroups) ? shared_sums[sid] : 0.0f;
    val = simd_sum(val);
    if (sid == 0) shared_sums[0] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float var = shared_sums[0] / (float)elems_per_group;
float inv_std = metal::rsqrt(var + eps[0]);

// Intermediate buffer for normalized values (stored in shared memory would be
// too large for typical channel counts, so we write to a temp output and read back).
// Instead, we normalize on-the-fly and store in the output buffer.

// Pass 3: Normalize, apply affine, then apply GLU
// GLU: output[c, s] = a[c, s] * sigmoid(b[c, s])
// where a = first half of channels, b = second half
// After GroupNorm+affine, the normalized values are in (B, 2C, S) layout.
// We need to pair channel c with channel c + C.

// For GLU, we iterate over the OUTPUT elements (half the channels).
// Each output element at (c_out, s) reads:
//   a = norm_affine(x[base_a + c_out * S + s])
//   b = norm_affine(x[base_b + c_out * S + s])
// where base_a/base_b account for the group offset.

// But groups span the full 2C channels. We need to handle the case
// where the GLU split crosses group boundaries.
// For standard usage (num_groups divides 2C and C), the first half of groups
// correspond to 'a' channels and second half to 'b' channels.

// Actually for simplicity and correctness, this kernel should handle the common
// case where num_groups divides both 2C and C (e.g., groups=1 or groups=4 with
// 2C divisible by 4). The GLU split is on the channel dimension, independent of groups.

// We compute all normalized values first, then apply GLU.
// Strategy: each thread processes output elements. For each output (c_out, s),
// compute both a and b normalized values on the fly.

// Half channels per group on the output side:
uint half_cpg = channels_per_group / 2;
uint out_epg = half_cpg * spatial_size;
uint out_base = batch_idx * half_channels * spatial_size
              + group_idx * half_cpg * spatial_size;

for (uint i = tid; i < out_epg; i += tg_size) {
    uint c_local = i / spatial_size;
    uint s = i % spatial_size;

    // 'a' channel: first half of the group
    uint c_a = c_local;
    uint c_a_global = group_idx * channels_per_group + c_a;
    float val_a = ((float)x[base + c_a * spatial_size + s] - mean) * inv_std;
    val_a = val_a * (float)weight[c_a_global] + (float)bias[c_a_global];

    // 'b' channel: second half of the group
    uint c_b = c_local + half_cpg;
    uint c_b_global = group_idx * channels_per_group + c_b;
    float val_b = ((float)x[base + c_b * spatial_size + s] - mean) * inv_std;
    val_b = val_b * (float)weight[c_b_global] + (float)bias[c_b_global];

    // GLU: a * sigmoid(b)
    float sig_b = 1.0f / (1.0f + metal::exp(-val_b));
    out[out_base + i] = (T)(val_a * sig_b);
}
"""

_groupnorm_glu_kernel = None


def _get_groupnorm_glu_kernel():
    global _groupnorm_glu_kernel
    if _groupnorm_glu_kernel is None:
        _groupnorm_glu_kernel = mx.fast.metal_kernel(
            name="fused_groupnorm_glu",
            input_names=["x", "weight", "bias", "eps", "params"],
            output_names=["out"],
            source=_GROUPNORM_GLU_SOURCE,
        )
    return _groupnorm_glu_kernel


@mx.compile
def _groupnorm_glu_fallback(
    x: mx.array, weight: mx.array, bias: mx.array,
    num_groups: int, eps: float = 1e-5,
) -> mx.array:
    """Pure-MLX GroupNorm + GLU (no Metal required)."""
    B, C_full = x.shape[0], x.shape[1]
    cpg = C_full // num_groups
    x_r = x.reshape(B, num_groups, cpg, *x.shape[2:])
    axes = tuple(range(2, x_r.ndim))
    mean = x_r.mean(axis=axes, keepdims=True)
    var = mx.var(x_r, axis=axes, keepdims=True)
    x_norm = (x_r - mean) * mx.rsqrt(var + eps)
    x_out = x_norm.reshape(x.shape)
    w_shape = [1, C_full] + [1] * (x.ndim - 2)
    normed = x_out * weight.reshape(w_shape) + bias.reshape(w_shape)
    a, b = mx.split(normed, 2, axis=1)
    return a * mx.sigmoid(b)


def fused_groupnorm_glu(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    num_groups: int,
    eps: float = 1e-5,
) -> mx.array:
    """Fused GroupNorm + GLU for NCL or NCHW layout.

    GroupNorm normalizes over 2C channels, then GLU splits in half on axis=1.

    Equivalent to:
        x = groupnorm(x, weight, bias, num_groups, eps)
        a, b = split(x, 2, axis=1)
        return a * sigmoid(b)

    Input shape: (B, 2C, ...) -> Output shape: (B, C, ...)

    Note: Only supported when num_groups=1 or when channels_per_group is even
    (so the GLU split aligns with group boundaries). When num_groups > 1 and
    channels_per_group is odd, falls back to separate GroupNorm + GLU.
    """
    orig_shape = x.shape
    B = x.shape[0]
    C_full = x.shape[1]  # 2C
    C_half = C_full // 2  # C

    spatial_size = 1
    for d in x.shape[2:]:
        spatial_size *= d

    if C_full % num_groups != 0:
        raise ValueError(f"channels {C_full} not divisible by num_groups {num_groups}")
    if C_full % 2 != 0:
        raise ValueError(f"channels {C_full} must be even for GLU")
    channels_per_group = C_full // num_groups

    elems_per_group = channels_per_group * spatial_size

    # Pure-MLX fallback for cases where the Metal kernel underperforms:
    # - No Metal available
    # - Large groups (few threadgroups -> GPU underutilization)
    # - num_groups > 1 (GLU split crosses group boundaries)
    if not HAS_METAL or num_groups > 1 or elems_per_group > _HYBRID_THRESHOLD:
        return _groupnorm_glu_fallback(x, weight, bias, num_groups, eps)

    x_contig = mx.contiguous(x.reshape(B, C_full, spatial_size))
    weight = mx.contiguous(weight.astype(mx.float32))
    bias = mx.contiguous(bias.astype(mx.float32))
    eps_arr = mx.array([eps], dtype=mx.float32)
    params = mx.array(
        [num_groups, channels_per_group, spatial_size, C_full, C_half], dtype=mx.int32
    )

    total_groups = B * num_groups
    tg = min(1024, max(32, ((elems_per_group + 31) // 32) * 32))

    out_shape = list(orig_shape)
    out_shape[1] = C_half

    result = _get_groupnorm_glu_kernel()(
        inputs=[x_contig, weight, bias, eps_arr, params],
        template=[("T", x.dtype)],
        grid=(total_groups * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, C_half, spatial_size)],
        output_dtypes=[x.dtype],
    )[0]

    return result.reshape(out_shape)


# ==============================================================================
# Fused complex-to-real magnitude (CAC mode)
# ==============================================================================

_COMPLEX_TO_INTERLEAVED_SOURCE = r"""
uint i = thread_position_in_grid.x;
uint total = params[3];
if (i >= total) return;

uint T_dim = params[0];
uint Fr_dim = params[1];
uint C_dim = params[2];
uint FrT = Fr_dim * T_dim;
uint CFrT = C_dim * FrT;

uint b = i / (2 * CFrT);
uint rem = i % (2 * CFrT);
uint c2 = rem / FrT;
uint c = c2 / 2;
uint is_imag = c2 % 2;
uint ft = rem % FrT;

uint in_idx = (b * CFrT + c * FrT + ft) * 2 + is_imag;
out[i] = x[in_idx];
"""

_complex_to_interleaved_kernel = None


def _get_complex_to_interleaved_kernel():
    global _complex_to_interleaved_kernel
    if _complex_to_interleaved_kernel is None:
        _complex_to_interleaved_kernel = mx.fast.metal_kernel(
            name="complex_to_interleaved",
            input_names=["x", "params"],
            output_names=["out"],
            source=_COMPLEX_TO_INTERLEAVED_SOURCE,
        )
    return _complex_to_interleaved_kernel


def fused_complex_to_interleaved(z: mx.array) -> mx.array:
    """Convert complex (B, C, Fr, T) to interleaved real (B, 2*C, Fr, T).

    Equivalent to:
        real = mx.real(z)
        imag = mx.imag(z)
        m = mx.stack([real, imag], axis=2).reshape(B, C*2, Fr, T)
    """
    B, C, Fr, T = z.shape

    if not HAS_METAL:
        real = mx.real(z)
        imag = mx.imag(z)
        return mx.stack([real, imag], axis=2).reshape(B, C * 2, Fr, T)

    z_real = mx.view(z, mx.float32)  # (B, C, Fr, T, 2)
    z_flat = mx.contiguous(z_real).reshape(-1)

    total = B * 2 * C * Fr * T
    params = mx.array([T, Fr, C, total], dtype=mx.int32)

    result = _get_complex_to_interleaved_kernel()(
        inputs=[z_flat, params],
        template=[],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.float32],
    )[0]

    return result.reshape(B, 2 * C, Fr, T)


# ==============================================================================
# Fused parallel overlap-add across overlapping audio chunks
# ==============================================================================

_OVERLAP_ADD_SOURCE = r"""
uint t = thread_position_in_grid.x;
uint ch = thread_position_in_grid.y;
uint total_samples = params[0];
uint chunk_len = params[1];
uint step = params[2];
uint num_chunks = params[3];
uint num_channels = params[4];

if (t >= total_samples || ch >= num_channels) return;

int last_k = min((int)(num_chunks - 1), (int)(t / step));
int first_k = max(0, (int)((int)t - (int)chunk_len + (int)step) / (int)step);

float acc = 0.0f;
float wsum = 0.0f;

for (int k = first_k; k <= last_k; ++k) {
    int offset = k * step;
    int j = (int)t - offset;
    if (j >= 0 && j < (int)chunk_len) {
        float w = (float)window[j];
        uint frame_idx = (uint)k * num_channels * chunk_len + ch * chunk_len + (uint)j;
        acc += (float)frames[frame_idx] * w;
        wsum += w;
    }
}
out[ch * total_samples + t] = (T)((wsum > 1e-11f) ? (acc / wsum) : 0.0f);
"""

_overlap_add_kernel = None


def _get_overlap_add_kernel():
    global _overlap_add_kernel
    if _overlap_add_kernel is None:
        _overlap_add_kernel = mx.fast.metal_kernel(
            name="waveform_chunk_overlap_add",
            input_names=["frames", "window", "params"],
            output_names=["out"],
            source=_OVERLAP_ADD_SOURCE,
        )
    return _overlap_add_kernel


def _overlap_add_fallback(
    frames: mx.array,
    window: mx.array,
    step: int,
    total_samples: int,
) -> mx.array:
    orig_shape = frames.shape
    num_chunks = orig_shape[0]
    prefix_shape = orig_shape[1:-1]
    chunk_len = orig_shape[-1]
    num_channels = 1
    for s in prefix_shape:
        num_channels *= s
    frames_flat = frames.reshape(num_chunks, num_channels, chunk_len).astype(mx.float32)
    window_f32 = window.astype(mx.float32)
    out = mx.zeros((num_channels, total_samples), dtype=mx.float32)
    sum_weight = mx.zeros((total_samples,), dtype=mx.float32)
    for k in range(num_chunks):
        off = k * step
        this_len = min(chunk_len, total_samples - off)
        if this_len <= 0:
            continue
        end = off + this_len
        w = window_f32[:this_len]
        chunk_val = frames_flat[k, :, :this_len] * w.reshape(1, -1)
        out = mx.slice_update(out, out[:, off:end] + chunk_val, mx.array([off]), axes=(1,))
        sum_weight = sum_weight.at[off:end].add(w)
    out = out / mx.maximum(sum_weight.reshape(1, -1), 1e-11)
    return out.reshape(*prefix_shape, total_samples).astype(frames.dtype)



def fused_overlap_add(
    frames: mx.array,
    window: mx.array,
    step: int,
    total_samples: int,
) -> mx.array:
    """Fused parallel overlap-add across overlapping audio chunks.

    Args:
        frames: Shape (num_chunks, ..., chunk_len) containing processed audio chunks.
        window: Shape (chunk_len,) weighting / analysis-synthesis window.
        step: Hop size / stride between consecutive chunk offsets (must be > 0).
        total_samples: Target output length in samples along the time axis.

    Returns:
        Array of shape (..., total_samples) normalized by the accumulated window.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    orig_shape = frames.shape
    if len(orig_shape) < 2:
        raise ValueError(f"frames must have at least 2 dimensions, got {orig_shape}")

    num_chunks = orig_shape[0]
    prefix_shape = orig_shape[1:-1]
    chunk_len = orig_shape[-1]

    if num_chunks == 0 or total_samples == 0:
        return mx.zeros((*prefix_shape, total_samples), dtype=frames.dtype)

    if not HAS_METAL:
        return _overlap_add_fallback(frames, window, step, total_samples)

    num_channels = 1
    for s in prefix_shape:
        num_channels *= s

    frames_flat = mx.contiguous(frames.reshape(num_chunks, num_channels, chunk_len))
    window_contig = mx.contiguous(window)
    params = mx.array([total_samples, chunk_len, step, num_chunks, num_channels], dtype=mx.int32)

    tg_x = min(256, total_samples)
    kernel = _get_overlap_add_kernel()
    result = kernel(
        inputs=[frames_flat, window_contig, params],
        template=[("T", frames.dtype)],
        grid=(total_samples, num_channels, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(num_channels * total_samples,)],
        output_dtypes=[frames.dtype],
    )[0]

    return result.reshape(*prefix_shape, total_samples)

