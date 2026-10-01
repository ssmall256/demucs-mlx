"""Microbenchmark attention head layout and SDPA latency."""
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from demucs_mlx.api import Separator

def main():
    print("=" * 78)
    print("HEADROOM 3: FAST MULTI-HEAD ATTENTION HEAD PROJECTION BENCHMARK")
    print("=" * 78)

    sep = Separator(seed=481, batch_size=8)
    m = sep.model.models[0]
    layer = m.crosstransformer.layers[0] # TransformerEncoderLayer
    attn = layer.attn # FastMultiHeadAttention
    attn._ensure_fused()

    B = 8
    T = 2688 # 336 * 8
    C = 512
    heads = 8
    head_dim = C // heads # 64

    rng = np.random.default_rng(481)
    x_np = rng.standard_normal((B, T, C)).astype(np.float32)
    x = mx.array(x_np)
    mx.eval(x)

    # Warmup
    for _ in range(3):
        out = attn(x, x, x)
        mx.eval(out)

    times_attn = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = attn(x, x, x)
        mx.eval(out)
        times_attn.append((time.perf_counter() - t0) * 1000)

    # Isolate QKV projection vs Transpose vs SDPA vs OutProj
    q_fp16 = x.astype(mx.float16)
    mx.eval(q_fp16)

    times_qkv = []
    for _ in range(20):
        t0 = time.perf_counter()
        qkv = attn.qkv_proj(q_fp16)
        mx.eval(qkv)
        times_qkv.append((time.perf_counter() - t0) * 1000)

    qkv = attn.qkv_proj(q_fp16)
    q, k, v = mx.split(qkv, [C, 2 * C], axis=-1)
    mx.eval(q, k, v)

    times_trans = []
    for _ in range(20):
        t0 = time.perf_counter()
        qt = mx.unflatten(q, -1, (heads, -1)).transpose(0, 2, 1, 3)
        kt = mx.unflatten(k, -1, (heads, -1)).transpose(0, 2, 1, 3)
        vt = mx.unflatten(v, -1, (heads, -1)).transpose(0, 2, 1, 3)
        mx.eval(qt, kt, vt)
        times_trans.append((time.perf_counter() - t0) * 1000)

    qt = mx.unflatten(q, -1, (heads, -1)).transpose(0, 2, 1, 3)
    kt = mx.unflatten(k, -1, (heads, -1)).transpose(0, 2, 1, 3)
    vt = mx.unflatten(v, -1, (heads, -1)).transpose(0, 2, 1, 3)
    scale = 1.0 / (head_dim ** 0.5)
    mx.eval(qt, kt, vt)

    times_sdpa = []
    for _ in range(20):
        t0 = time.perf_counter()
        res_sdpa = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale)
        mx.eval(res_sdpa)
        times_sdpa.append((time.perf_counter() - t0) * 1000)

    res_sdpa = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale)
    mx.eval(res_sdpa)

    times_out_trans = []
    for _ in range(20):
        t0 = time.perf_counter()
        res_t = res_sdpa.transpose(0, 2, 1, 3).flatten(-2, -1)
        mx.eval(res_t)
        times_out_trans.append((time.perf_counter() - t0) * 1000)

    res_t = res_sdpa.transpose(0, 2, 1, 3).flatten(-2, -1)
    mx.eval(res_t)

    times_out_proj = []
    for _ in range(20):
        t0 = time.perf_counter()
        final_res = attn.out_proj_fp16(res_t)
        mx.eval(final_res)
        times_out_proj.append((time.perf_counter() - t0) * 1000)

    med_attn = np.median(times_attn)
    med_qkv = np.median(times_qkv)
    med_trans = np.median(times_trans)
    med_sdpa = np.median(times_sdpa)
    med_out_trans = np.median(times_out_trans)
    med_out_proj = np.median(times_out_proj)

    print(f"Per-Layer MultiHeadAttention Latency Breakdown (B={B}, T={T}, C={C}):")
    print(f"  Total Attention Layer: {med_attn:6.2f} ms")
    print(f"  - QKV Fused GEMM:      {med_qkv:6.2f} ms ({med_qkv/med_attn*100:4.1f}%)")
    print(f"  - Input Head Transpose:{med_trans:6.2f} ms ({med_trans/med_attn*100:4.1f}%)")
    print(f"  - Scaled Dot-Product:  {med_sdpa:6.2f} ms ({med_sdpa/med_attn*100:4.1f}%)")
    print(f"  - Output Head Transpose:{med_out_trans:6.2f} ms ({med_out_trans/med_attn*100:4.1f}%)")
    print(f"  - Out Proj GEMM:       {med_out_proj:6.2f} ms ({med_out_proj/med_attn*100:4.1f}%)")
    
    total_trans_ms = (med_trans + med_out_trans) * 10 * 3
    print("-" * 78)
    print(f"  Transpose overhead across 120s (10 layers x 3 batches): {total_trans_ms:6.2f} ms")

if __name__ == "__main__":
    main()
