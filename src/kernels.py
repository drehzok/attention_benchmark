"""Fused Derf normalization + linear projection via JAX Pallas kernels (TPU).

The main public API is `fused_derf_linear(x, weight, bias, alpha, s, gamma, beta)`
which computes:
    derf_out = gamma * erf(alpha * x + s) + beta
    return derf_out @ weight + bias

On TPU with sufficiently large tensors, this runs as a single Pallas kernel
that keeps the Derf intermediate in VMEM, avoiding an HBM roundtrip.
On CPU or for small tensors, it falls back to pure JAX ops.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

# Block sizes for TPU MXU alignment
_BM = 128
_BK = 128
_BN = 128
_MIN_M = 128  # fallback to JAX if M < this


# ---------------------------------------------------------------------------
# Pure JAX reference (fallback and backward)
# ---------------------------------------------------------------------------
def _derf_linear_ref(x, weight, bias, alpha, s, gamma, beta):
    """Pure JAX: derf(x) @ weight + bias."""
    normed = gamma * lax.erf(alpha * x + s) + beta
    return normed @ weight + bias


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------
def _make_kernel(alpha, s):
    """Create kernel function with alpha/s captured via closure."""
    def kernel(x_ref, w_ref, bias_ref, gamma_ref, beta_ref, out_ref):
        k_iter = jax.lax.axis_index("k") if False else None  # placeholder
        from jax.experimental import pallas as pl

        k_iter = pl.program_id(2)

        x_block = x_ref[...]
        w_block = w_ref[...]
        g_block = gamma_ref[jnp.newaxis, :]
        b_block = beta_ref[jnp.newaxis, :]

        normed = g_block * lax.erf(alpha * x_block + s) + b_block
        result = pl.dot(normed, w_block)

        @pl.when(k_iter == 0)
        def _init():
            out_ref[...] = jnp.broadcast_to(bias_ref[jnp.newaxis, :], result.shape) + result

        @pl.when(k_iter != 0)
        def _accum():
            out_ref[...] = out_ref[...] + result

    return kernel


def _pallas_fused(x_2d, weight, bias, alpha, s, gamma, beta):
    """Invoke Pallas kernel. x_2d: (M, K), weight: (K, N)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    M, K = x_2d.shape
    N = weight.shape[1]

    kernel_fn = _make_kernel(alpha, s)

    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((M, N), x_2d.dtype),
        grid=(M // _BM, N // _BN, K // _BK),
        in_specs=[
            pl.BlockSpec((_BM, _BK), lambda m, n, k: (m, k)),
            pl.BlockSpec((_BK, _BN), lambda m, n, k: (k, n)),
            pl.BlockSpec((_BN,), lambda m, n, k: (n,)),
            pl.BlockSpec((_BK,), lambda m, n, k: (k,)),
            pl.BlockSpec((_BK,), lambda m, n, k: (k,)),
        ],
        out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k: (m, n)),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(x_2d, weight, bias, gamma, beta)


# ---------------------------------------------------------------------------
# Public API with custom_vjp
# ---------------------------------------------------------------------------
def _can_use_pallas(M, K, N):
    return (
        jax.default_backend() == "tpu"
        and M >= _MIN_M
        and M % _BM == 0
        and K % _BK == 0
        and N % _BN == 0
    )


def _impl(x, weight, bias, alpha, s, gamma, beta):
    """Forward implementation: Pallas on TPU, JAX fallback otherwise."""
    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    x_2d = x.reshape(M, K)

    if _can_use_pallas(M, K, N):
        out_2d = _pallas_fused(x_2d, weight, bias, alpha, s, gamma, beta)
    else:
        out_2d = _derf_linear_ref(x_2d, weight, bias, alpha, s, gamma, beta)

    return out_2d.reshape(*leading, N)


@functools.partial(jax.custom_vjp)
def fused_derf_linear(x, weight, bias, alpha, s, gamma, beta):
    """Fused Derf normalization + linear projection.

    Computes: (gamma * erf(alpha * x + s) + beta) @ weight + bias

    Args:
        x: (..., dim) input
        weight: (dim, out_dim) projection weight
        bias: (out_dim,) projection bias
        alpha: scalar Derf parameter
        s: scalar Derf shift parameter
        gamma: (dim,) Derf per-dim scale
        beta: (dim,) Derf per-dim bias

    Returns:
        (..., out_dim) output
    """
    return _impl(x, weight, bias, alpha, s, gamma, beta)


def _fwd(x, weight, bias, alpha, s, gamma, beta):
    out = _impl(x, weight, bias, alpha, s, gamma, beta)
    return out, (x, weight, alpha, s, gamma, beta)


def _bwd(res, g):
    x, weight, alpha, s, gamma, beta = res

    # Recompute intermediates
    u = alpha * x + s
    erf_u = lax.erf(u)
    normed = gamma * erf_u + beta

    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    normed_2d = normed.reshape(M, K)
    g_2d = g.reshape(M, N)

    d_bias = g_2d.sum(axis=0)
    d_weight = normed_2d.T @ g_2d
    d_normed = (g_2d @ weight.T).reshape(x.shape)

    reduce_axes = tuple(range(len(leading)))
    d_beta = d_normed.sum(axis=reduce_axes)
    d_gamma = (d_normed * erf_u).sum(axis=reduce_axes)

    erf_deriv = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-u * u)

    d_x = d_normed * gamma * erf_deriv * alpha
    d_alpha = (d_normed * gamma * erf_deriv * x).sum()
    d_s = (d_normed * gamma * erf_deriv).sum()

    return (d_x, d_weight, d_bias, d_alpha, d_s, d_gamma, d_beta)


fused_derf_linear.defvjp(_fwd, _bwd)
