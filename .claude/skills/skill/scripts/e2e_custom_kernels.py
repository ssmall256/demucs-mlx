"""End-to-end example: patch an mlx-lm model with custom Metal kernels.

Demonstrates:
- Loading a quantized model from mlx-community
- Benchmarking baseline inference speed
- Patching RMSNorm modules with a custom Metal kernel
- Measuring patched performance
- Printing a comparison table

Requires: pip install mlx-lm
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time

import mlx.core as mx

from kernels.utils import scalar_f32
from kernels.autotune_cache import pick_threadgroup
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Custom RMSNorm kernel (same as in rmsnorm_kernel.py)
# ---------------------------------------------------------------------------
RMSNORM_SOURCE = """
uint row = threadgroup_position_in_grid.x;
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tg_size = threads_per_threadgroup.x;
uint D = x_shape[x_ndim - 1];
float eps_val = eps[0];

float sum_sq = 0.0f;
for (uint i = tid; i < D; i += tg_size) {
    float val = (float)x[row * D + i];
    sum_sq += val * val;
}
sum_sq = simd_sum(sum_sq);

threadgroup float shared[32];
if (lane == 0) shared[sg] = sum_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    uint num_sg = (tg_size + 31) / 32;
    float p = (lane < num_sg) ? shared[lane] : 0.0f;
    shared[0] = simd_sum(p);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float rms = metal::rsqrt(shared[0] / float(D) + eps_val);

for (uint i = tid; i < D; i += tg_size) {
    out[row * D + i] = (T)((float)x[row * D + i] * rms * (float)w[i]);
}
"""

_rmsnorm_kernel = mx.fast.metal_kernel(
    name="custom_rmsnorm_e2e",
    input_names=["x", "w", "eps"],
    output_names=["out"],
    source=RMSNORM_SOURCE,
)


class CustomRMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm using a custom Metal kernel."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims
        self._eps_arr = scalar_f32(eps)

    def __call__(self, x: mx.array) -> mx.array:
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)
        N = x_2d.shape[0]
        def _run(tgx: int):
            tgx = max(32, (tgx // 32) * 32)
            return _rmsnorm_kernel(
                inputs=[x_2d, self.weight, self._eps_arr],
                template=[("T", x.dtype)],
                grid=(N * tgx, 1, 1),
                threadgroup=(tgx, 1, 1),
                output_shapes=[(N, D)],
                output_dtypes=[x.dtype],
            )[0]

        tg = pick_threadgroup(
            kernel_name="rmsnorm_e2e",
            shape_sig=f"N={N},D={D}",
            dtype_sig=str(x.dtype),
            candidates=[32, 64, 128, 256, 512, 1024],
            run=_run,
            default=min(256, D),
        )

        out = _run(tg)
        return out.reshape(orig_shape)


def patch_rmsnorm(model: nn.Module) -> int:
    """Replace all nn.RMSNorm modules with CustomRMSNorm."""
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "RMSNorm":
            dims = module.weight.shape[0]
            eps = getattr(module, "eps", 1e-5)
            replacement = CustomRMSNorm(dims, eps)
            replacement.weight = module.weight
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            count += 1
    return count


def benchmark_generation(model, tokenizer, prompt: str, max_tokens: int, warmup: int = 2, trials: int = 3):
    """Measure tokens per second for text generation."""
    from mlx_lm import generate

    # Warmup
    for _ in range(warmup):
        generate(model, tokenizer, prompt=prompt, max_tokens=16)

    # Timed trials
    times = []
    for _ in range(trials):
        mx.synchronize()
        t0 = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = max_tokens / avg_time
    return tokens_per_sec, avg_time


def main():
    parser = argparse.ArgumentParser(description="E2E custom kernel benchmark")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="Model path or HuggingFace repo")
    parser.add_argument("--prompt", default="Explain the concept of attention in transformers:",
                        help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("ERROR: mlx-lm is required for this example.")
        print("Install it with: pip install mlx-lm")
        return

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load(args.model)
    print("Model loaded.\n")

    # Baseline benchmark
    print("Benchmarking baseline...")
    baseline_tps, baseline_time = benchmark_generation(
        model, tokenizer, args.prompt, args.max_tokens
    )
    print(f"  Baseline: {baseline_tps:.1f} tokens/sec ({baseline_time:.2f}s)\n")

    # Patch RMSNorm modules
    n_patched = patch_rmsnorm(model)
    print(f"Patched {n_patched} RMSNorm modules with custom kernel.\n")

    # Patched benchmark
    print("Benchmarking patched model...")
    patched_tps, patched_time = benchmark_generation(
        model, tokenizer, args.prompt, args.max_tokens
    )
    print(f"  Patched:  {patched_tps:.1f} tokens/sec ({patched_time:.2f}s)\n")

    # Results table
    print("=" * 50)
    print(f"{'':>20s}  {'tok/s':>8s}  {'time':>8s}")
    print("-" * 50)
    print(f"{'Baseline':>20s}  {baseline_tps:>8.1f}  {baseline_time:>7.2f}s")
    print(f"{'Custom RMSNorm':>20s}  {patched_tps:>8.1f}  {patched_time:>7.2f}s")
    speedup = patched_tps / baseline_tps
    print("-" * 50)
    print(f"{'Speedup':>20s}  {speedup:>8.2f}x")
    print("=" * 50)

    if speedup < 1.0:
        print("\nNote: Custom kernel is slower than the built-in.")
        print("This is expected — MLX's built-in RMSNorm is already optimized.")
        print("Custom kernels shine for fused operations not available as built-ins.")


if __name__ == "__main__":
    main()
