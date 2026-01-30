"""Test passing a scalar into a Pallas matmul kernel on TPU.

Computes:  (alpha * x) @ w
where alpha: scalar, x: (M, K), w: (K, N) -> (M, N)

Two approaches tested:
  A) scalar_prefetch (SMEM) — alpha packed into a (1,) array
  B) BlockSpec broadcast   — alpha reshaped to (1, 1), BlockSpec((1, 1), ...)

Run:  python tests/test_scalar_matmul.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MATMUL_DIMS = (((1,), (0,)), ((), ()))
_BM = 128
_BK = 128
_BN = 128


# ===========================================================================
# Approach A: scalar_prefetch (SMEM)
# ===========================================================================

def _scalar_prefetch_kernel(scalar_ref, x_ref, w_ref, out_ref):
    k_iter = pl.program_id(2)

    alpha = scalar_ref[0]
    x_blk = x_ref[...].astype(jnp.float32)
    w_blk = w_ref[...].astype(jnp.float32)

    scaled = alpha * x_blk

    partial = lax.dot_general(
        scaled, w_blk,
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def scalar_prefetch_matmul(x, w, alpha):
    M, K = x.shape
    N = w.shape[1]

    scalars = jnp.array([alpha], dtype=jnp.float32)

    return pl.pallas_call(
        _scalar_prefetch_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec((_BM, _BK), lambda m, n, k, _: (m, k)),
                pl.BlockSpec((_BK, _BN), lambda m, n, k, _: (k, n)),
            ],
            out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k, _: (m, n)),
            grid=(M // _BM, N // _BN, K // _BK),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(scalars, x, w)


# ===========================================================================
# Approach B: BlockSpec broadcast — scalar as (1, 1) array
# ===========================================================================

def _blockspec_scalar_kernel(x_ref, w_ref, alpha_ref, out_ref):
    k_iter = pl.program_id(2)

    x_blk = x_ref[...].astype(jnp.float32)
    w_blk = w_ref[...].astype(jnp.float32)
    alpha = alpha_ref[...].astype(jnp.float32)  # (1, 1)

    # (1, 1) * (BM, BK) broadcasts to (BM, BK)
    scaled = alpha * x_blk

    partial = lax.dot_general(
        scaled, w_blk,
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def blockspec_scalar_matmul(x, w, alpha):
    M, K = x.shape
    N = w.shape[1]

    alpha_2d = jnp.array([[alpha]], dtype=jnp.float32)  # (1, 1)

    return pl.pallas_call(
        _blockspec_scalar_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((_BM, _BK), lambda m, n, k: (m, k)),
                pl.BlockSpec((_BK, _BN), lambda m, n, k: (k, n)),
                pl.BlockSpec((1, 1),     lambda m, n, k: (0, 0)),  # alpha
            ],
            out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k: (m, n)),
            grid=(M // _BM, N // _BN, K // _BK),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(x, w, alpha_2d)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def reference(x, w, alpha):
    return (alpha * x) @ w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_approach(name, fn):
    print("=" * 60)
    print(f"TEST: {name}")
    print("=" * 60)

    shapes = [
        (128,  128, 128,  "minimal"),
        (256,  256, 256,  "256 all"),
        (512,  256, 512,  "large"),
    ]

    all_passed = True
    for M, K, N, desc in shapes:
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key, 2)

        x     = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w     = jax.random.normal(k2, (K, N), dtype=jnp.float32) * 0.1
        alpha = 0.5

        try:
            out_kernel = fn(x, w, alpha)
            out_ref    = reference(x, w, alpha)

            max_err = float(jnp.max(jnp.abs(out_kernel - out_ref)))
            passed = bool(jnp.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-3))
            status = "PASS" if passed else "FAIL"

            print(f"  [{status}] {desc:15s}  ({M},{K},{N})  max_err={max_err:.2e}")
        except Exception as e:
            print(f"  [ERROR] {desc:15s}  ({M},{K},{N})  {type(e).__name__}: {e}")
            passed = False

        if not passed:
            all_passed = False

    return all_passed


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    results = [
        ("A: scalar_prefetch (SMEM)",     test_approach("A: scalar_prefetch", scalar_prefetch_matmul)),
        ("B: BlockSpec broadcast (1,1)",   test_approach("B: BlockSpec (1,1)", blockspec_scalar_matmul)),
    ]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if all(p for _, p in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
