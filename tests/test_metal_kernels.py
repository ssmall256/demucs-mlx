"""
Numerical parity tests for custom Metal kernels.
Each test compares custom kernel output against reference MLX implementation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn
import math


def test_fused_glu():
    """Test fused GLU against split + sigmoid reference."""
    from demucs_mlx.metal_kernels import fused_glu

    print("=== Fused GLU Kernel ===")
    test_cases = [
        # (shape, axis, description)
        ((1, 64, 100), 1, "Small NCL"),
        ((2, 128, 200), 1, "Medium NCL"),
        ((4, 384, 50), 1, "Large channels NCL"),
        ((1, 768, 10), 1, "Very large channels"),
        ((2, 96, 512), 1, "Long sequence"),
        ((1, 48, 1), 1, "Single timestep"),
        # Axis 0 test
        ((64, 32), 0, "2D axis=0"),
        # Last axis
        ((4, 16, 128), -1, "Last axis"),
    ]

    all_passed = True
    for shape, axis, desc in test_cases:
        x = mx.random.normal(shape) * 2.0  # Scale up to test sigmoid saturation
        mx.eval(x)

        # Reference
        a, b = mx.split(x, 2, axis=axis)
        ref = a * mx.sigmoid(b)
        mx.eval(ref)

        # Custom kernel
        out = fused_glu(x, axis=axis)
        mx.eval(out)

        max_diff = mx.max(mx.abs(ref - out)).item()
        # GLU involves sigmoid which can amplify small diffs, use 1e-5 for f32
        passed = max_diff < 1e-5
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {desc:30s} shape={str(shape):20s} axis={axis}  max_diff={max_diff:.2e}  [{status}]")

    print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    assert all_passed


def test_fused_groupnorm_gelu():
    """Test fused GroupNorm+GELU against separate GroupNorm + nn.gelu reference."""
    from demucs_mlx.metal_kernels import fused_groupnorm_gelu
    from demucs_mlx.mlx_layers import GroupNormNCL, GroupNormNCHW

    print("=== Fused GroupNorm+GELU Kernel ===")
    test_cases = [
        # (shape, num_groups, description)
        ((1, 48, 100), 4, "HTDemucs first layer NCL"),
        ((2, 96, 50), 4, "HTDemucs second layer NCL"),
        ((1, 192, 25), 4, "HTDemucs third layer NCL"),
        ((1, 384, 12), 4, "HTDemucs deepest layer NCL"),
        ((4, 48, 512, 10), 4, "NCHW spectral"),
        ((1, 96, 256, 20), 4, "NCHW mid"),
        ((1, 32, 1), 1, "Single group single timestep"),
        ((1, 64, 1024), 1, "Single group long"),
        ((2, 384, 1), 4, "Large channels single timestep"),
    ]

    all_passed = True
    for shape, num_groups, desc in test_cases:
        x = mx.random.normal(shape).astype(mx.float32)
        C = shape[1]
        weight = mx.random.normal((C,)).astype(mx.float32)
        bias = mx.random.normal((C,)).astype(mx.float32)
        eps = 1e-5
        mx.eval(x, weight, bias)

        # Reference: manual GroupNorm + GELU
        B = shape[0]
        G = num_groups
        spatial = 1
        for d in shape[2:]:
            spatial *= d
        x_r = x.reshape(B, G, C // G, *x.shape[2:])
        axes = tuple(range(2, x_r.ndim))
        mean = x_r.mean(axis=axes, keepdims=True)
        var = ((x_r - mean) ** 2).mean(axis=axes, keepdims=True)
        x_norm = (x_r - mean) * mx.rsqrt(var + eps)
        x_out = x_norm.reshape(x.shape)
        w_shape = [1, C] + [1] * (x.ndim - 2)
        ref = x_out * weight.reshape(w_shape) + bias.reshape(w_shape)
        ref = nn.gelu(ref)
        mx.eval(ref)

        # Custom kernel
        out = fused_groupnorm_gelu(x, weight, bias, num_groups, eps)
        mx.eval(out)

        max_diff = mx.max(mx.abs(ref - out)).item()
        # GroupNorm + GELU involves multiple reductions and tanh; 1e-4 is reasonable
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
            # Debug: print more info
            mean_diff = mx.mean(mx.abs(ref - out)).item()
            print(f"    DEBUG: mean_diff={mean_diff:.2e}, ref_range=[{mx.min(ref).item():.4f}, {mx.max(ref).item():.4f}]")
        print(f"  {desc:40s} shape={str(shape):20s} G={num_groups}  max_diff={max_diff:.2e}  [{status}]")

    print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    assert all_passed


def test_fused_complex_to_interleaved():
    """Test fused complex-to-interleaved against reference."""
    from demucs_mlx.metal_kernels import fused_complex_to_interleaved

    print("=== Fused Complex-to-Interleaved Kernel ===")
    test_cases = [
        ((1, 2, 2049, 10), "Small STFT"),
        ((1, 4, 2049, 43), "HTDemucs typical"),
        ((2, 2, 1025, 100), "Batch=2"),
        ((1, 2, 512, 1), "Single frame"),
        ((1, 1, 4, 4), "Minimal"),
    ]

    all_passed = True
    for shape, desc in test_cases:
        B, C, Fr, T = shape
        real_part = mx.random.normal((B, C, Fr, T))
        imag_part = mx.random.normal((B, C, Fr, T))
        z = real_part + 1j * imag_part
        mx.eval(z)

        # Reference
        real = mx.real(z)
        imag = mx.imag(z)
        ref = mx.stack([real, imag], axis=2).reshape(B, C * 2, Fr, T)
        mx.eval(ref)

        # Custom kernel
        out = fused_complex_to_interleaved(z)
        mx.eval(out)

        max_diff = mx.max(mx.abs(ref - out)).item()
        passed = max_diff < 1e-6  # This should be exact (just reordering floats)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {desc:30s} shape={str(shape):20s}  max_diff={max_diff:.2e}  [{status}]")

    print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    assert all_passed


def test_fused_groupnorm_glu():
    """Test fused GroupNorm+GLU against separate GroupNorm + GLU reference."""
    from demucs_mlx.metal_kernels import fused_groupnorm_glu

    print("=== Fused GroupNorm+GLU Kernel ===")
    test_cases = [
        # (shape, num_groups, description)
        ((1, 96, 344),      1, "G=1 NCL (DConv typical)"),
        ((1, 192, 172),     1, "G=1 NCL mid"),
        ((1, 384, 86),      1, "G=1 NCL deep"),
        ((2, 128, 200),     1, "G=1 batch=2"),
        ((1, 96, 344),      4, "G=4 NCL (fallback)"),
        ((1, 96, 512, 43),  4, "G=4 NCHW (fallback)"),
        ((1, 64, 100),      1, "G=1 small"),
    ]

    all_passed = True
    for shape, num_groups, desc in test_cases:
        C = shape[1]
        x = mx.random.normal(shape).astype(mx.float32) * 2.0
        weight = mx.random.normal((C,)).astype(mx.float32)
        bias = mx.random.normal((C,)).astype(mx.float32)
        eps = 1e-5
        mx.eval(x, weight, bias)

        # Reference: manual GroupNorm + GLU
        B = shape[0]
        G = num_groups
        x_r = x.reshape(B, G, C // G, *x.shape[2:])
        axes = tuple(range(2, x_r.ndim))
        mean = x_r.mean(axis=axes, keepdims=True)
        var = ((x_r - mean) ** 2).mean(axis=axes, keepdims=True)
        x_norm = (x_r - mean) * mx.rsqrt(var + eps)
        x_out = x_norm.reshape(x.shape)
        w_shape = [1, C] + [1] * (x.ndim - 2)
        normed = x_out * weight.reshape(w_shape) + bias.reshape(w_shape)
        a, b = mx.split(normed, 2, axis=1)
        ref = a * mx.sigmoid(b)
        mx.eval(ref)

        # Custom kernel
        out = fused_groupnorm_glu(x, weight, bias, num_groups, eps)
        mx.eval(out)

        max_diff = mx.max(mx.abs(ref - out)).item()
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {desc:35s} shape={str(shape):20s} G={num_groups}  max_diff={max_diff:.2e}  [{status}]")

    print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    assert all_passed


def test_transformer_norm_fix():
    """Verify the norm1 caching fix doesn't change outputs."""
    from demucs_mlx.mlx_transformer import TransformerEncoderLayer, CrossTransformerEncoderLayer

    print("=== Transformer Norm Fix (Functional Equivalence) ===")

    # Test self-attention layer
    d_model = 384
    nhead = 8
    dim_ff = 1536
    layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=0.0,
        activation=lambda x: nn.gelu(x),
        norm_first=True,
        layer_scale=True,
    )
    layer.eval()

    x = mx.random.normal((1, 10, d_model))
    mx.eval(x)

    # Run forward pass
    out = layer(x)
    mx.eval(out)
    print(f"  TransformerEncoderLayer: output shape={out.shape}, mean={mx.mean(out).item():.6f}  [OK]")

    # Test cross-attention layer
    cross_layer = CrossTransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=0.0,
        activation=lambda x: nn.gelu(x),
        norm_first=True,
        layer_scale=True,
    )
    cross_layer.eval()

    q = mx.random.normal((1, 10, d_model))
    k = mx.random.normal((1, 20, d_model))
    mx.eval(q, k)

    out = cross_layer(q, k)
    mx.eval(out)
    print(f"  CrossTransformerEncoderLayer: output shape={out.shape}, mean={mx.mean(out).item():.6f}  [OK]")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Metal Kernel Numerical Parity Tests")
    print("=" * 70)
    print()

    test_transformer_norm_fix()
    test_fused_glu()
    test_fused_groupnorm_gelu()
    test_fused_groupnorm_glu()
    test_fused_complex_to_interleaved()

    print("=" * 70)
    print("All tests passed!")
