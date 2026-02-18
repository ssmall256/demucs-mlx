# Testing Patterns for Metal Kernels

## Edge-Case Categories

Custom Metal kernels often work on "normal" inputs but fail on edge cases. Test these categories systematically:

### Shape Edge Cases

| Category | Example Shapes | Why It Breaks |
|----------|---------------|---------------|
| Single element | `(1, 1)` | Grid has 1 thread; reduction logic may assume multiple lanes |
| Small D (< SIMD width) | `(4, 16)` | Threads stride past D in first iteration; uninitialized accumulator lanes |
| D = 1 | `(8, 1)` | Degenerate reduction; division by D may cause issues |
| Large D | `(1, 16384)` | Many loop iterations; floating-point accumulation drift |
| Empty batch | `(0, 128)` | Grid size 0 causes Metal error |
| Large batch | `(4096, 128)` | Many threadgroups; tests scaling |
| Non-power-of-2 D | `(2, 127)` | Threadgroup size doesn't divide D evenly |
| 3D input | `(2, 8, 4096)` | Reshape to 2D may have bugs |

### Data Edge Cases

| Category | How to Create | Why It Breaks |
|----------|--------------|---------------|
| All zeros | `mx.zeros(shape)` | Division by zero in normalization (RMS=0) |
| Large values | `mx.ones(shape) * 1e4` | `exp()` overflow in softmax |
| Negative values | `-mx.ones(shape) * 1e4` | `exp()` underflow; test numerical stability |
| NaN in input | `mx.array([float('nan')])` | NaN propagation through reductions |
| Inf in input | `mx.array([float('inf')])` | Inf arithmetic edge cases |
| Mixed signs | `mx.array([-1e3, 1e3])` | Tests range handling in softmax |

### Memory Layout Edge Cases

| Category | How to Create | Why It Breaks |
|----------|--------------|---------------|
| Non-contiguous (transpose) | `x.T` | Strides are non-trivial; `ensure_row_contiguous` must handle it |
| Non-contiguous (slice) | `x[:, ::2]` | Stride > 1 on last dim |
| Broadcast input | `mx.ones((1, D))` with batch ops | Shape/stride mismatch |

## Reusable Test Runner

Extend the `validate()` pattern used in the example scripts:

```python
def run_test_suite(custom_fn, reference_fn, test_cases, atol_f32=1e-5, atol_f16=1e-2):
    """Run a custom kernel against a reference across test cases.

    Args:
        custom_fn: The custom kernel function.
        reference_fn: The reference (MLX built-in) function.
        test_cases: List of (description, input_arrays) tuples.
            Each input_arrays is a tuple of mx.arrays to pass to both functions.
        atol_f32: Absolute tolerance for float32.
        atol_f16: Absolute tolerance for float16.

    Returns:
        True if all tests pass.
    """
    all_passed = True
    for desc, inputs in test_cases:
        dtype = inputs[0].dtype
        atol = atol_f16 if dtype == mx.float16 else atol_f32

        expected = reference_fn(*inputs)
        actual = custom_fn(*inputs)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        passed = max_diff < atol
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  {desc:>40s}  max_diff={max_diff:.2e}  [{status}]")

    return all_passed
```

### Building Test Cases

```python
def make_standard_test_cases(make_inputs_fn, dtype=mx.float32):
    """Generate standard edge-case test inputs.

    Args:
        make_inputs_fn: Callable(shape, dtype) -> tuple of mx.arrays.
            For example, for RMSNorm: lambda s, d: (mx.random.normal(s).astype(d), mx.ones((s[-1],)).astype(d))
    """
    shapes = [
        (1, 1),            # Single element
        (1, 16),           # Small D (< SIMD width)
        (4, 32),           # D = SIMD width
        (2, 127),          # Non-power-of-2 D
        (1, 128),          # Standard small
        (4, 1024),         # Standard medium
        (2, 4096),         # Standard large
        (1, 16384),        # Very large D
        (2, 8, 4096),      # 3D input
    ]
    cases = []
    for shape in shapes:
        inputs = make_inputs_fn(shape, dtype)
        cases.append((f"{dtype} shape={shape}", inputs))
    return cases
```

## Testing Non-Contiguous Inputs

Kernels that use `ensure_row_contiguous=True` (the default) should work with non-contiguous inputs because MLX copies them to contiguous layout. But this copy adds overhead, and if you set `ensure_row_contiguous=False`, non-contiguous inputs will produce wrong results unless you handle strides manually.

Test both:

```python
# Contiguous (normal)
x = mx.random.normal((4, 128))
test_kernel(x)  # Should work

# Non-contiguous via transpose
x_t = mx.random.normal((128, 4)).T   # Shape (4, 128) but non-contiguous strides
test_kernel(x_t)  # Should work with ensure_row_contiguous=True

# Non-contiguous via slicing
x_sliced = mx.random.normal((4, 256))[:, ::2]  # Shape (4, 128), stride 2
test_kernel(x_sliced)  # Should work with ensure_row_contiguous=True
```

## Testing Numerical Extremes

```python
def test_numerical_extremes(kernel_fn, reference_fn, shape=(4, 128), dtype=mx.float32):
    """Test kernel behavior with extreme numerical inputs."""
    D = shape[-1]

    cases = [
        ("zeros", mx.zeros(shape, dtype=dtype)),
        ("ones", mx.ones(shape, dtype=dtype)),
        ("large positive", mx.ones(shape, dtype=dtype) * 100),
        ("large negative", mx.ones(shape, dtype=dtype) * -100),
        ("mixed extremes", mx.concatenate([
            mx.ones((shape[0], D // 2), dtype=dtype) * 100,
            mx.ones((shape[0], D - D // 2), dtype=dtype) * -100,
        ], axis=-1)),
    ]

    for desc, x in cases:
        expected = reference_fn(x)
        actual = kernel_fn(x)
        mx.eval(expected, actual)

        max_diff = mx.max(mx.abs(expected - actual)).item()
        has_nan = mx.any(mx.isnan(actual)).item()
        status = "PASS" if max_diff < 1e-3 and not has_nan else "FAIL"
        print(f"  {desc:>20s}  max_diff={max_diff:.2e}  nan={has_nan}  [{status}]")
```

## Extending the validate() Pattern

The existing scripts (`rmsnorm_kernel.py`, `softmax_kernel.py`) use a `validate(shapes, dtype)` function. To add edge-case coverage, add shapes to the default list:

```python
def validate(shapes=None, dtype=mx.float32):
    if shapes is None:
        shapes = [
            # Original shapes
            (1, 128, 4096),
            (4, 256, 1024),
            # Edge cases
            (1, 1),             # Single element
            (1, 16),            # D < SIMD width
            (1, 16384),         # Very large D
            (2, 127),           # Non-power-of-2
        ]
    # ... rest of validation
```

Also add a non-contiguous test:

```python
# After the shape loop, test transposed input
x_t = mx.random.normal((D, 4)).T.astype(dtype)  # Non-contiguous (4, D)
expected = reference_fn(x_t)
actual = custom_fn(x_t)
# ... compare
```
