"""
Very small smoke tests for local runs / CI.

Run:
  python -m skill.tests.smoke_test
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

def _run_snippet(snippet: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def _mlx_is_usable() -> bool:
    rc, output = _run_snippet("import mlx.core as mx; print('ok')")
    if rc == 0:
        return True
    detail = output.splitlines()[-1] if output else f"exit code {rc}"
    print("mlx is not usable in this environment; skipping smoke tests:", detail)
    return False


def test_rmsnorm():
    snippet = textwrap.dedent(
        """
        import mlx.core as mx
        from skill.scripts.rmsnorm_kernel import rmsnorm

        x = mx.random.normal((4, 128)).astype(mx.float16)
        w = mx.ones((128,), dtype=mx.float16)
        y = rmsnorm(x, w, eps=1e-5)
        y_ref = mx.fast.rms_norm(x, w, eps=1e-5)
        mx.eval(y, y_ref)

        diff = mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item()
        tol = 1e-3 + 1e-3 * max(1.0, float(mx.max(mx.abs(y_ref)).item()))
        if diff > tol:
            raise AssertionError(f"rmsnorm max|diff|={diff}")
        """
    )
    rc, output = _run_snippet(snippet)
    if rc != 0:
        raise AssertionError(f"rmsnorm smoke test failed:\n{output}")


def test_softmax():
    snippet = textwrap.dedent(
        """
        import mlx.core as mx
        from skill.scripts.softmax_kernel import softmax

        x = mx.random.normal((8, 257)).astype(mx.float16)
        y = softmax(x)
        y_ref = mx.softmax(x, axis=-1)
        mx.eval(y, y_ref)

        diff = mx.max(mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))).item()
        tol = 2e-3 + 2e-3 * max(1.0, float(mx.max(mx.abs(y_ref)).item()))
        if diff > tol:
            raise AssertionError(f"softmax max|diff|={diff}")
        """
    )
    rc, output = _run_snippet(snippet)
    if rc != 0:
        raise AssertionError(f"softmax smoke test failed:\n{output}")


def main():
    if not _mlx_is_usable():
        return
    test_rmsnorm()
    test_softmax()
    print("smoke tests passed")

if __name__ == "__main__":
    main()
