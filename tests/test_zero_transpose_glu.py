"""Test and benchmark zero-transpose 2D-coalesced Vectorized Metal GLU kernel."""
import time

import mlx.core as mx
import numpy as np

_ZERO_TRANSPOSE_GLU_SOURCE = r"""
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

_ZERO_TRANSPOSE_GLU_VEC4_SOURCE = r"""
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

_zero_transpose_glu_kernel = None
_zero_transpose_glu_vec4_kernel = None

def get_zero_transpose_glu_kernel():
    global _zero_transpose_glu_kernel
    if _zero_transpose_glu_kernel is None:
        _zero_transpose_glu_kernel = mx.fast.metal_kernel(
            name="zero_transpose_glu",
            input_names=["x", "params"],
            output_names=["out"],
            source=_ZERO_TRANSPOSE_GLU_SOURCE,
        )
    return _zero_transpose_glu_kernel

def get_zero_transpose_glu_vec4_kernel():
    global _zero_transpose_glu_vec4_kernel
    if _zero_transpose_glu_vec4_kernel is None:
        _zero_transpose_glu_vec4_kernel = mx.fast.metal_kernel(
            name="zero_transpose_glu_vec4",
            input_names=["x", "params"],
            output_names=["out"],
            source=_ZERO_TRANSPOSE_GLU_VEC4_SOURCE,
        )
    return _zero_transpose_glu_vec4_kernel

def zero_transpose_fused_glu(x: mx.array, axis: int = 1) -> mx.array:
    ndim = x.ndim
    axis = axis % ndim
    shape = list(x.shape)
    if shape[axis] % 2 != 0:
        raise ValueError(f"Axis {axis} size must be even, got {shape[axis]}")

    M = 1
    for s in shape[:axis]:
        M *= s
    C = shape[axis] // 2
    K = 1
    for s in shape[axis + 1:]:
        K *= s

    MC = M * C
    total_out = MC * K
    out_shape = shape[:axis] + [C] + shape[axis + 1:]

    # Check vector-4 alignment
    if K % 4 == 0 and x.size % 4 == 0:
        K4 = K // 4
        params = mx.array([K4, C, MC], dtype=mx.int32)
        tg_x = min(256, K4)
        if tg_x >= 32:
            tg_x = (tg_x // 32) * 32
        tg_y = min(256 // max(1, tg_x), MC)
        tg_y = max(1, tg_y)
        kernel = get_zero_transpose_glu_vec4_kernel()
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
        kernel = get_zero_transpose_glu_kernel()
        result = kernel(
            inputs=[x, params],
            template=[("T", x.dtype)],
            grid=(K, MC, 1),
            threadgroup=(max(1, tg_x), tg_y, 1),
            output_shapes=[(total_out,)],
            output_dtypes=[x.dtype],
        )[0]

    return result.reshape(out_shape)

def main():
    from demucs_mlx.metal_kernels import fused_glu as old_fused_glu

    shapes = [
        ("NCL (DConv waveform)", (8, 96, 344), 1),
        ("NCL (DConv mid)", (8, 192, 172), 1),
        ("NCL (DConv deep)", (8, 384, 86), 1),
        ("NCHW (Spectral mid)", (8, 96, 256, 43), 1),
        ("NCL (Encoder tail)", (8, 96, 85995), 1),
    ]

    @mx.compile
    def compiled_glu(x, axis):
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    def native_glu(x, axis):
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    print("=" * 80)
    print("VECTORIZED ZERO-TRANSPOSE GLU vs COMPILED vs NATIVE vs OLD FUSED")
    print("=" * 80)

    for desc, shape, axis in shapes:
        x = mx.random.normal(shape).astype(mx.float32)
        mx.eval(x)

        # Warmup all
        _ = native_glu(x, axis)
        mx.eval(_)
        _ = compiled_glu(x, axis)
        mx.eval(_)
        _ = old_fused_glu(x, axis)
        mx.eval(_)
        _ = zero_transpose_fused_glu(x, axis)
        mx.eval(_)

        t_native = []
        for _ in range(25):
            t0 = time.perf_counter()
            out_nat = native_glu(x, axis)
            mx.eval(out_nat)
            t_native.append((time.perf_counter() - t0) * 1000)

        t_comp = []
        for _ in range(25):
            t0 = time.perf_counter()
            out_comp = compiled_glu(x, axis)
            mx.eval(out_comp)
            t_comp.append((time.perf_counter() - t0) * 1000)

        t_old = []
        for _ in range(25):
            t0 = time.perf_counter()
            out_old = old_fused_glu(x, axis)
            mx.eval(out_old)
            t_old.append((time.perf_counter() - t0) * 1000)

        t_new = []
        for _ in range(25):
            t0 = time.perf_counter()
            out_new = zero_transpose_fused_glu(x, axis)
            mx.eval(out_new)
            t_new.append((time.perf_counter() - t0) * 1000)

        med_nat = np.median(t_native)
        med_comp = np.median(t_comp)
        med_old = np.median(t_old)
        med_new = np.median(t_new)

        err_new = float(mx.max(mx.abs(out_nat - out_new)))

        print(f"\nShape {shape} - {desc}:")
        print(f"  Native (split+sigmoid): {med_nat:6.3f} ms")
        print(f"  Compiled (@mx.compile):  {med_comp:6.3f} ms")
        print(f"  Old Fused (Transposed):  {med_old:6.3f} ms")
        print(
            f"  Vectorized Zero-Trans:   {med_new:6.3f} ms  <-- {med_old/med_new:.2f}x vs old, "
            f"{med_nat/med_new:.2f}x vs native, {med_comp/med_new:.2f}x vs comp"
        )
        print(f"  Max Diff vs Reference:   {err_new:.2e}")

if __name__ == "__main__":
    main()
