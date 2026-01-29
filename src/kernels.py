"""Fused Derf normalization + linear projection via JAX Pallas kernels (TPU).

Computes: output = (gamma * erf(alpha * x + s) + beta) @ weight + bias

The forward pass runs as a single fused Pallas kernel on TPU that keeps the
Derf intermediate in VMEM, avoiding an HBM round-trip.  Falls back to pure
JAX on non-TPU backends or when shapes are not aligned to block sizes.

The backward pass uses pure JAX with recomputation of intermediates.

Block sizes (128, 128) are the LCM of VPU-optimal (8, 128) and MXU-optimal
(128, 128), giving good utilisation on both processing units.

erf  -> computed on VPU  (optimal tile: multiple of (8, 128))
matmul -> computed on MXU (optimal tile: (128, 128))
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MATMUL_DIMS = (((1,), (0,)), ((), ()))  # contract dim 1 of lhs with dim 0 of rhs
_TWO_OVER_SQRT_PI = 1.1283791670955126   # 2/sqrt(pi)
_ERF_COEFF        = 0.08943              # cubic coefficient for erf(x) approx

# Block sizes — LCM of VPU (8,128) and MXU (128,128) optimal tiles.
_BM = 128
_BK = 128
_BN = 128


# ---------------------------------------------------------------------------
# Pure JAX reference (fallback & gradient computation)
# ---------------------------------------------------------------------------
def _derf_linear_ref(x, weight, bias, alpha, s, gamma, beta):
    """Pure JAX: (gamma * erf(alpha*x + s) + beta) @ weight + bias."""
    normed = gamma * lax.erf(alpha * x + s) + beta
    return normed @ weight + bias


# ---------------------------------------------------------------------------
# Pallas forward kernel
# ---------------------------------------------------------------------------
def _fused_kernel(x_ref, w_ref, bias_ref, gamma_ref, beta_ref,
                  alpha_ref, s_ref, out_ref):
    """Fused derf + matmul kernel body.

    Each grid cell (m, n, k) computes a (BM, BN) partial result:
        normed_block = gamma[k] * erf(alpha * x[m,k] + s) + beta[k]
        partial      = normed_block @ weight[k,n]
    and accumulates into out[m, n].  Bias is added once on the first k iter.
    """
    k_iter = pl.program_id(2)

    # Load tiles into VMEM, upcast to f32 for numerical stability.
    x_blk     = x_ref[...].astype(jnp.float32)      # (BM, BK)
    w_blk     = w_ref[...].astype(jnp.float32)      # (BK, BN)
    gamma_blk = gamma_ref[...].astype(jnp.float32)  # (BK,)
    beta_blk  = beta_ref[...].astype(jnp.float32)   # (BK,)
    bias_blk  = bias_ref[...].astype(jnp.float32)   # (BN,)
    alpha_blk = alpha_ref[...].astype(jnp.float32)  # (BK,) broadcast scalar
    s_blk     = s_ref[...].astype(jnp.float32)      # (BK,) broadcast scalar

    # ---- Fused Derf (runs on VPU) ----
    # Approximate erf via tanh: erf(x) ≈ tanh(√(2/π) · (x + 0.044715·x³))
    # lax.erf has no Pallas TPU lowering; tanh does.
    u = alpha_blk[jnp.newaxis, :] * x_blk + s_blk[jnp.newaxis, :]
    erf_approx = lax.tanh(_TWO_OVER_SQRT_PI * (u + _ERF_COEFF * u * u * u))
    normed = (gamma_blk[jnp.newaxis, :] * erf_approx
              + beta_blk[jnp.newaxis, :])            # (BM, BK)

    # ---- Matrix multiply (runs on MXU) ----
    partial = lax.dot_general(
        normed, w_blk,
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )                                                # (BM, BN)

    # ---- Accumulate into output ----
    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial + bias_blk[jnp.newaxis, :]

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def _pallas_forward(x_2d, weight, bias, alpha, s, gamma, beta):
    """Invoke the Pallas kernel.  x_2d: (M, K), weight: (K, N)."""
    M, K = x_2d.shape
    N = weight.shape[1]

    # Pallas kernels cannot capture traced values — pass alpha/s as (K,) arrays.
    alpha_arr = jnp.full((K,), alpha, dtype=jnp.float32)
    s_arr     = jnp.full((K,), s,     dtype=jnp.float32)

    return pl.pallas_call(
        _fused_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((_BM, _BK), lambda m, n, k: (m, k)),  # x
                pl.BlockSpec((_BK, _BN), lambda m, n, k: (k, n)),  # weight
                pl.BlockSpec((_BN,),     lambda m, n, k: (n,)),     # bias
                pl.BlockSpec((_BK,),     lambda m, n, k: (k,)),     # gamma
                pl.BlockSpec((_BK,),     lambda m, n, k: (k,)),     # beta
                pl.BlockSpec((_BK,),     lambda m, n, k: (k,)),     # alpha
                pl.BlockSpec((_BK,),     lambda m, n, k: (k,)),     # s
            ],
            out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k: (m, n)),
            grid=(M // _BM, N // _BN, K // _BK),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(x_2d, weight, bias, gamma, beta, alpha_arr, s_arr)


# ---------------------------------------------------------------------------
# Dispatch: Pallas on TPU when shapes align, JAX fallback otherwise
# ---------------------------------------------------------------------------
def _can_use_pallas(M, K, N):
    return (
        jax.default_backend() == "tpu"
        and M % _BM == 0
        and K % _BK == 0
        and N % _BN == 0
    )


def _forward_impl(x, weight, bias, alpha, s, gamma, beta):
    """Run forward pass, choosing Pallas or JAX fallback."""
    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    x_2d = x.reshape(M, K)

    if _can_use_pallas(M, K, N):
        out_2d = _pallas_forward(x_2d, weight, bias, alpha, s, gamma, beta)
    else:
        out_2d = _derf_linear_ref(x_2d, weight, bias, alpha, s, gamma, beta)

    return out_2d.reshape(*leading, N)


# ---------------------------------------------------------------------------
# Public API with custom_vjp for backpropagation
# ---------------------------------------------------------------------------
@functools.partial(jax.custom_vjp)
def fused_derf_linear(x, weight, bias, alpha, s, gamma, beta):
    """Fused Derf + linear projection.

    Computes: (gamma * erf(alpha * x + s) + beta) @ weight + bias

    Args:
        x:      (..., dim) input tensor.
        weight: (dim, out_dim) projection matrix.
        bias:   (out_dim,) projection bias.
        alpha:  scalar Derf scale parameter.
        s:      scalar Derf shift parameter.
        gamma:  (dim,) per-channel Derf scale.
        beta:   (dim,) per-channel Derf bias.

    Returns:
        (..., out_dim) output tensor.
    """
    return _forward_impl(x, weight, bias, alpha, s, gamma, beta)


def _fwd(x, weight, bias, alpha, s, gamma, beta):
    out = _forward_impl(x, weight, bias, alpha, s, gamma, beta)
    # Save residuals for backward — recompute intermediates to save memory.
    return out, (x, weight, alpha, s, gamma, beta)


def _bwd(res, g):
    x, weight, alpha, s, gamma, beta = res

    # Recompute intermediates (cheaper than saving from forward).
    u = alpha * x + s
    erf_u = lax.erf(u)
    normed = gamma * erf_u + beta                                # (..., K)

    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    normed_2d = normed.reshape(M, K)
    g_2d = g.reshape(M, N)

    # ---- Gradients ----
    # d(output)/d(bias) = I
    d_bias = g_2d.sum(axis=0)                                    # (N,)

    # d(output)/d(weight) = normed^T @ g
    d_weight = normed_2d.T @ g_2d                                # (K, N)

    # d(output)/d(normed) = g @ weight^T
    d_normed = (g_2d @ weight.T).reshape(x.shape)                # (..., K)

    # erf'(u) = 2/sqrt(pi) * exp(-u^2)
    erf_deriv = (2.0 / jnp.sqrt(jnp.pi)) * jnp.exp(-(u * u))

    reduce_axes = tuple(range(len(leading)))
    d_gamma = (d_normed * erf_u).sum(axis=reduce_axes)           # (K,)
    d_beta  = d_normed.sum(axis=reduce_axes)                     # (K,)
    d_x     = d_normed * gamma * erf_deriv * alpha               # (..., K)
    d_alpha = (d_normed * gamma * erf_deriv * x).sum()           # scalar
    d_s     = (d_normed * gamma * erf_deriv).sum()               # scalar

    return (d_x, d_weight, d_bias, d_alpha, d_s, d_gamma, d_beta)


fused_derf_linear.defvjp(_fwd, _bwd)
