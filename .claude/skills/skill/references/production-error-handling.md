# Production Error Handling for MLX Metal Kernels

## Device Availability Check

Always guard Metal kernel paths with an availability check. Provide a pure-MLX fallback so your code works on machines without a Metal GPU (e.g., CI, Linux):

```python
import mlx.core as mx

def create_rmsnorm_op():
    """Factory: returns kernel-backed op if Metal is available, else fallback."""
    if not mx.metal.is_available():
        def fallback(x, weight, eps=1e-5):
            return mx.fast.rms_norm(x, weight, eps)
        return fallback

    kernel = mx.fast.metal_kernel(
        name="rmsnorm", input_names=["x", "w", "eps"], output_names=["out"],
        source=RMSNORM_SOURCE,
    )
    def kernel_op(x, weight, eps=1e-5):
        D = x.shape[-1]
        x_2d = x.reshape(-1, D)
        N = x_2d.shape[0]
        tg = min(256, max(32, (D // 32) * 32))
        eps_buf = mx.array([eps], dtype=mx.float32)
        return kernel(
            inputs=[x_2d, weight, eps_buf], template=[("T", x.dtype)],
            grid=(N * tg, 1, 1), threadgroup=(tg, 1, 1),
            output_shapes=[(N, D)], output_dtypes=[x.dtype],
        )[0].reshape(x.shape)
    return kernel_op

rmsnorm = create_rmsnorm_op()
```

## Input Validation

Validate inputs **before** dispatching the kernel. Errors inside a Metal kernel are hard to diagnose — they may silently produce garbage or crash the GPU command buffer.

```python
def validate_inputs(x, weight, bias=None):
    D = x.shape[-1]

    # Shape checks
    if x.ndim < 1:
        raise ValueError(f"Input must be at least 1-D, got shape {x.shape}")
    if weight.shape != (D,):
        raise ValueError(f"Weight shape {weight.shape} doesn't match last dim {D}")
    if bias is not None and bias.shape != (D,):
        raise ValueError(f"Bias shape {bias.shape} doesn't match last dim {D}")

    # Empty tensor guard — no kernel launch needed
    if x.size == 0:
        return x

    # dtype validation — bfloat16 requires M3+
    if x.dtype == mx.bfloat16:
        info = mx.device_info()
        device_name = info.get("device_name", "")
        if not any(chip in device_name for chip in ["M3", "M4"]):
            raise ValueError(
                f"bfloat16 not supported on {device_name}; use float16 instead"
            )

    return None  # Proceed to kernel
```

## Defensive Kernel Patterns

### Bounds Checking in Source

When grid size may exceed the output size (e.g., grid rounded up for alignment), add bounds checks:

```metal
uint i = thread_position_in_grid.x;
if (i >= out_shape[0]) return;  // Guard excess threads
out[i] = (T)((float)x[i] * 2.0f);
```

### Safe Output Initialization

Use `init_value` to zero-fill outputs before the kernel runs. This prevents uninitialized memory from leaking through if some threads exit early:

```python
out = kernel(
    inputs=[x],
    template=[("T", x.dtype)],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(M,)],
    output_dtypes=[x.dtype],
    init_value=0,  # All output bytes zeroed before kernel executes
)
```

## Graceful Degradation Pattern

Wrap kernel dispatch in a try/except and fall back to pure-MLX ops on failure. Validate once on first call to catch kernel bugs early:

```python
class RobustKernelOp:
    """Wraps a kernel with automatic fallback and one-time validation."""

    def __init__(self, kernel_fn, fallback_fn, atol=1e-3):
        self._kernel_fn = kernel_fn
        self._fallback_fn = fallback_fn
        self._atol = atol
        self._use_kernel = True
        self._validated = False

    def __call__(self, *args, **kwargs):
        if not self._validated and self._use_kernel:
            self._validated = True
            try:
                kernel_out = self._kernel_fn(*args, **kwargs)
                ref_out = self._fallback_fn(*args, **kwargs)
                mx.eval(kernel_out, ref_out)
                max_diff = mx.max(mx.abs(kernel_out - ref_out)).item()
                if max_diff > self._atol:
                    print(f"[WARN] Kernel validation failed (diff={max_diff:.2e}), "
                          f"using fallback")
                    self._use_kernel = False
                    return ref_out
                return kernel_out
            except Exception as e:
                print(f"[WARN] Kernel failed ({e}), using fallback")
                self._use_kernel = False
                return self._fallback_fn(*args, **kwargs)

        if self._use_kernel:
            try:
                return self._kernel_fn(*args, **kwargs)
            except Exception as e:
                print(f"[WARN] Kernel runtime error ({e}), switching to fallback")
                self._use_kernel = False
                return self._fallback_fn(*args, **kwargs)

        return self._fallback_fn(*args, **kwargs)
```

Usage:

```python
rmsnorm = RobustKernelOp(
    kernel_fn=my_kernel_rmsnorm,
    fallback_fn=lambda x, w, eps=1e-5: mx.fast.rms_norm(x, w, eps),
)
```

## Error Messages in Production

### Errors Surface at `mx.eval()`, Not at Kernel Call

MLX uses lazy evaluation. When you call `kernel(...)`, it only records the operation in the compute graph. The Metal kernel doesn't actually execute until `mx.eval()` or another synchronization point:

```python
out = kernel(inputs=[x], ...)   # No error yet — just queued
mx.eval(out)                     # Error happens HERE if kernel is invalid
```

This means the Python traceback at eval time may not point to the kernel call. Use this logging pattern to track which kernel was last dispatched:

```python
import logging
log = logging.getLogger(__name__)

def safe_kernel_call(name, kernel, **kwargs):
    log.debug(f"Dispatching kernel '{name}' grid={kwargs.get('grid')}")
    try:
        result = kernel(**kwargs)
        mx.eval(result)
        return result
    except RuntimeError as e:
        log.error(f"Kernel '{name}' failed: {e}")
        raise
```

### Common Runtime Errors

| Error | Likely Cause |
|-------|-------------|
| `Abort trap: 6` or GPU hang | Out-of-bounds memory access in kernel |
| `Command buffer execution failed` | Kernel timeout (> ~2 seconds) or GPU fault |
| Silent wrong output | Grid/threadgroup math error, or missing barrier |
| `libc++abi: terminating` | Accessing negative index via unsigned int underflow |
