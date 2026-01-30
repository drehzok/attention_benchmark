"""Tests for pallas_matmul kernel (forward + backward).

Tests forward and backward correctness against pure JAX reference,
gradient checks, JIT compatibility, and various shapes.

Run:  python tests/test_pallas_matmul.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from src.kernels import pallas_matmul


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def reference_linear(x, weight, bias):
    """Pure JAX: x @ weight + bias."""
    return x @ weight + bias


def make_inputs(key, batch_shape, dim, out_dim):
    keys = jax.random.split(key, 3)
    x      = jax.random.normal(keys[0], (*batch_shape, dim))
    weight = jax.random.normal(keys[1], (dim, out_dim)) * 0.1
    bias   = jax.random.normal(keys[2], (out_dim,)) * 0.1
    return x, weight, bias


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_forward_correctness():
    print("=" * 60)
    print("TEST: Forward correctness")
    print("=" * 60)

    shapes = [
        ((2, 16),   64,  128,  "small – JAX fallback"),
        ((4, 32),   128, 256,  "medium"),
        ((128,),    128, 128,  "2-D input (M=128)"),
        ((2, 4, 8), 64,  128,  "3-D batch"),
        ((1, 128),  128, 128,  "TPU-aligned minimal"),
        ((4, 128),  256, 512,  "TPU-aligned large"),
    ]

    all_passed = True
    for batch_shape, dim, out_dim, desc in shapes:
        key = jax.random.PRNGKey(42)
        x, weight, bias = make_inputs(key, batch_shape, dim, out_dim)

        out_kernel = pallas_matmul(x, weight, bias)
        out_ref    = reference_linear(x, weight, bias)

        max_err = float(jnp.max(jnp.abs(out_kernel - out_ref)))
        passed  = bool(jnp.allclose(out_kernel, out_ref, atol=1e-4, rtol=1e-4))
        status  = "PASS" if passed else "FAIL"

        print(f"  [{status}] {desc:30s}  "
              f"shape={batch_shape}x{dim}->{out_dim}  max_err={max_err:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def test_backward_correctness():
    print("=" * 60)
    print("TEST: Backward correctness (gradients)")
    print("=" * 60)

    shapes = [
        ((2, 16),  64,  128,  "small"),
        ((4, 32),  128, 256,  "medium"),
        ((128,),   128, 128,  "2-D input"),
        ((1, 128), 128, 128,  "TPU-aligned"),
    ]

    names = ["x", "weight", "bias"]
    all_passed = True

    for batch_shape, dim, out_dim, desc in shapes:
        key = jax.random.PRNGKey(7)
        args = make_inputs(key, batch_shape, dim, out_dim)

        def loss_kernel(*a):
            return pallas_matmul(*a).sum()

        def loss_ref(*a):
            return reference_linear(*a).sum()

        grads_kernel = jax.grad(loss_kernel, argnums=(0, 1, 2))(*args)
        grads_ref    = jax.grad(loss_ref,    argnums=(0, 1, 2))(*args)

        shape_passed = True
        for gk, gr, name in zip(grads_kernel, grads_ref, names):
            max_err = float(jnp.max(jnp.abs(gk - gr)))
            ok = bool(jnp.allclose(gk, gr, atol=1e-4, rtol=1e-4))
            if not ok:
                print(f"  [FAIL] {desc} grad_{name}  max_err={max_err:.2e}")
                shape_passed = False
                all_passed = False

        if shape_passed:
            print(f"  [PASS] {desc:30s}  all 3 gradients match")

    return all_passed


def test_finite_gradients():
    print("=" * 60)
    print("TEST: Gradient finiteness")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    args = make_inputs(key, (4, 32), 128, 256)

    grads = jax.grad(
        lambda *a: pallas_matmul(*a).sum(),
        argnums=(0, 1, 2),
    )(*args)

    names = ["x", "weight", "bias"]
    all_ok = True
    for g, name in zip(grads, names):
        if not bool(jnp.all(jnp.isfinite(g))):
            print(f"  [FAIL] grad_{name} has non-finite values")
            all_ok = False

    if all_ok:
        print("  [PASS] All gradients are finite")
    return all_ok


def test_jit_compatibility():
    print("=" * 60)
    print("TEST: JIT compatibility")
    print("=" * 60)

    key = jax.random.PRNGKey(99)
    args = make_inputs(key, (2, 16), 64, 128)

    @jax.jit
    def fwd(*a):
        return pallas_matmul(*a)

    @jax.jit
    def grad_fn(*a):
        return jax.grad(lambda *a: pallas_matmul(*a).sum(),
                        argnums=(0, 1, 2))(*a)

    out   = fwd(*args)
    grads = grad_fn(*args)

    fwd_ok  = bool(jnp.all(jnp.isfinite(out)))
    grad_ok = all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    ok = fwd_ok and grad_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] JIT forward + backward")
    return ok


def test_value_and_grad():
    print("=" * 60)
    print("TEST: value_and_grad")
    print("=" * 60)

    key = jax.random.PRNGKey(0)
    args = make_inputs(key, (2, 16), 64, 128)

    def loss(*a):
        return pallas_matmul(*a).sum()

    val, grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(*args)
    ref_val = reference_linear(*args).sum()

    val_ok  = bool(jnp.allclose(val, ref_val, atol=1e-4))
    grad_ok = all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    ok = val_ok and grad_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] value_and_grad consistency")
    return ok


def test_numerical_gradient():
    """Finite-difference gradient check for d_x."""
    print("=" * 60)
    print("TEST: Numerical gradient check (finite differences)")
    print("=" * 60)

    key = jax.random.PRNGKey(77)
    x, weight, bias = make_inputs(key, (2, 4), 8, 16)

    def f(x_flat):
        return pallas_matmul(x_flat.reshape(x.shape), weight, bias).sum()

    x_flat = x.ravel()
    analytic = jax.grad(f)(x_flat)

    eps = 1e-4
    numerical = jnp.zeros_like(x_flat)
    for i in range(x_flat.shape[0]):
        e = jnp.zeros_like(x_flat).at[i].set(eps)
        numerical = numerical.at[i].set((f(x_flat + e) - f(x_flat - e)) / (2 * eps))

    max_err = float(jnp.max(jnp.abs(analytic - numerical)))
    ok = max_err < 1e-2

    print(f"  [{'PASS' if ok else 'FAIL'}] d_x finite-diff  max_err={max_err:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    results = [
        ("Forward correctness",      test_forward_correctness()),
        ("Backward correctness",     test_backward_correctness()),
        ("Gradient finiteness",      test_finite_gradients()),
        ("JIT compatibility",        test_jit_compatibility()),
        ("value_and_grad",           test_value_and_grad()),
        ("Numerical gradient check", test_numerical_gradient()),
    ]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
