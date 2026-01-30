"""Test broadcasting a 1D vector into a Pallas matmul kernel on TPU.

Computes:  (gamma * x) @ w
where gamma: (K,), x: (M, K), w: (K, N) -> (M, N)

gamma is reshaped to (1, K) and tiled via BlockSpec((1, BK), ...).
The (1, BK) block broadcasts naturally with the (BM, BK) x_block.

Run:  python tests/test_broadcast_matmul.py
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


# ---------------------------------------------------------------------------
# Kernel: (gamma * x) @ w
# ---------------------------------------------------------------------------

def _broadcast_matmul_kernel(x_ref, w_ref, gamma_ref, out_ref):
    """gamma_ref is (1, BK), tiled along K by BlockSpec."""
    k_iter = pl.program_id(2)

    x_blk     = x_ref[...].astype(jnp.float32)      # (BM, BK)
    w_blk     = w_ref[...].astype(jnp.float32)      # (BK, BN)
    gamma_blk = gamma_ref[...].astype(jnp.float32)  # (1, BK)

    # Broadcasting: (1, BK) * (BM, BK) -> (BM, BK)
    scaled = gamma_blk * x_blk

    partial = lax.dot_general(
        scaled, w_blk,
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )                                                 # (BM, BN)

    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def broadcast_matmul(x, w, gamma):
    """(gamma * x) @ w  using Pallas on TPU.

    x: (M, K), w: (K, N), gamma: (K,) -> (M, N).
    All of M, K, N must be multiples of 128.
    """
    M, K = x.shape
    N = w.shape[1]

    # Reshape gamma to (1, K) so BlockSpec can tile along K.
    # Block (1, BK) satisfies TPU alignment: dim[-2]=1 == array dim[-2]=1.
    gamma_2d = gamma.reshape(1, K)

    return pl.pallas_call(
        _broadcast_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((_BM, _BK), lambda m, n, k: (m, k)),  # x
                pl.BlockSpec((_BK, _BN), lambda m, n, k: (k, n)),  # w
                pl.BlockSpec((1, _BK),   lambda m, n, k: (0, k)),  # gamma
            ],
            out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k: (m, n)),
            grid=(M // _BM, N // _BN, K // _BK),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(x, w, gamma_2d)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def reference(x, w, gamma):
    return (gamma * x) @ w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_broadcast_matmul():
    print("=" * 60)
    print("TEST: (gamma * x) @ w — broadcast vector into Pallas matmul")
    print("=" * 60)

    shapes = [
        (128,  128, 128,  "minimal 128x128x128"),
        (256,  128, 128,  "M=256"),
        (128,  256, 128,  "K=256"),
        (128,  128, 256,  "N=256"),
        (256,  256, 256,  "256 all"),
        (512,  256, 512,  "large"),
    ]

    all_passed = True
    for M, K, N, desc in shapes:
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        x     = jax.random.normal(k1, (M, K), dtype=jnp.float32)
        w     = jax.random.normal(k2, (K, N), dtype=jnp.float32) * 0.1
        gamma = jax.random.normal(k3, (K,),   dtype=jnp.float32) * 0.5 + 1.0

        out_kernel = broadcast_matmul(x, w, gamma)
        out_ref    = reference(x, w, gamma)

        max_err = float(jnp.max(jnp.abs(out_kernel - out_ref)))
        passed = bool(jnp.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-3))
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] {desc:25s}  ({M},{K},{N})  max_err={max_err:.2e}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    ok = test_broadcast_matmul()

    print()
    if ok:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
