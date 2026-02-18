"""Run a quick benchmark suite for the example MLX Metal kernels.

Prints a consistent table:
- kernel name
- dtype
- shape
- median ms / iter (after warmup)

Notes:
- Uses mx.eval() + mx.synchronize() to avoid lazy-eval timing artifacts.
- This is intentionally "quick" (small iters). For more rigorous benchmarking,
  increase iters and pin CPU frequency where possible.
"""

import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx

from scripts.rmsnorm_kernel import rmsnorm as rmsnorm_fn
from scripts.softmax_kernel import softmax as softmax_fn
from scripts.layernorm_kernel import layernorm as layernorm_fn
from scripts.batch_elementwise_kernel import batch_silu as batch_silu_fn
from scripts.attention_kernel import attention as attn_fn
from scripts.attention_variants_kernel import causal_attention as causal_attn_fn
from scripts.multihead_rope_kernel import multihead_rope as rope_fn
from scripts.dequant_matvec_kernel import quantize_4bit, dequant_matvec as dequant_mv_fn


def _timeit(fn, warmup=3, iters=20):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
        mx.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _row(kernel, dtype, shape, ms):
    return f"{kernel:<22} {dtype:<10} {shape:<18} {ms:>8.3f}"


def main():
    if not mx.metal.is_available():
        raise SystemExit("Metal is not available. Run on macOS with a Metal-capable device.")

    print("=== mlx-metal-kernels-skill: quick bench ===")
    print(f"Device: {mx.default_device()}")
    print("")
    print(f"{'kernel':<22} {'dtype':<10} {'shape':<18} {'p50 ms':>8}")
    print("-" * 62)

    D = 4096
    x = mx.random.normal((2, 8, D)).astype(mx.float16)
    w = mx.ones((D,), dtype=mx.float16)
    b = mx.zeros((D,), dtype=mx.float16)

    ms = _timeit(lambda: rmsnorm_fn(x, w, eps=1e-5))
    print(_row("rmsnorm", "float16", str(tuple(x.shape)), ms))

    ms = _timeit(lambda: softmax_fn(x.reshape(-1, D)))
    print(_row("softmax", "float16", str((x.shape[0]*x.shape[1], D)), ms))

    ms = _timeit(lambda: layernorm_fn(x, w, b, eps=1e-5))
    print(_row("layernorm", "float16", str(tuple(x.shape)), ms))

    ms = _timeit(lambda: batch_silu_fn(x))
    print(_row("batch_silu", "float16", str(tuple(x.shape)), ms))

    seq = 256
    d = 128
    q = mx.random.normal((seq, d)).astype(mx.float16)
    k = mx.random.normal((seq, d)).astype(mx.float16)
    v = mx.random.normal((seq, d)).astype(mx.float16)
    ms = _timeit(lambda: attn_fn(q, k, v))
    print(_row("attn_single", "float16", str((seq, d)), ms))

    # Optional: batched single-head attention. Note: attention() loops over batch
    # in Python, so this is mainly for correctness/testing.
    B_attn = 2
    qb = mx.random.normal((B_attn, seq, d)).astype(mx.float16)
    kb = mx.random.normal((B_attn, seq, d)).astype(mx.float16)
    vb = mx.random.normal((B_attn, seq, d)).astype(mx.float16)
    ms = _timeit(lambda: attn_fn(qb, kb, vb))
    print(_row("attn_single_b", "float16", str((B_attn, seq, d)), ms))

    H = 8
    qh = mx.random.normal((H, seq, d)).astype(mx.float16)
    kh = mx.random.normal((H, seq, d)).astype(mx.float16)
    vh = mx.random.normal((H, seq, d)).astype(mx.float16)
    ms = _timeit(lambda: causal_attn_fn(qh, kh, vh))
    print(_row("attn_causal_mh", "float16", str((H, seq, d)), ms))

    B = 2
    xrope = mx.random.normal((B, H, seq, d)).astype(mx.float16)
    ms = _timeit(lambda: rope_fn(xrope, offset=0))
    print(_row("rope_mh", "float16", str((B, H, seq, d)), ms))

    out_f, in_f = 2048, 2048
    wq = mx.random.normal((out_f, in_f)).astype(mx.float16)
    w_packed, scales, biases = quantize_4bit(wq, group_size=64)
    xv = mx.random.normal((in_f,)).astype(mx.float16)
    ms = _timeit(lambda: dequant_mv_fn(w_packed, scales, biases, xv, group_size=64))
    print(_row("dequant_matvec", "float16", str((out_f, in_f)), ms))

    print("-" * 62)
    if not os.environ.get("MLX_METAL_AUTOTUNE"):
        print("Tip: set MLX_METAL_AUTOTUNE=1 to autotune threadgroup sizes (cached on disk).")
    else:
        print("Autotuning enabled (MLX_METAL_AUTOTUNE=1). Threadgroup sizes cached on disk.")


if __name__ == "__main__":
    main()
