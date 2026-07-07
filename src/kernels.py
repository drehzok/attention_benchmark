"""Pallas TPU kernels: matmul and fused Derf+linear.

Block sizes (128, 128) are the LCM of VPU-optimal (8, 128) and MXU-optimal
(128, 128), giving good utilisation on both processing units.

The fused derf kernel uses a 2D grid (M, K) with full-N-width weight tiles,
so derf is computed once per (m, k) pair and the MXU uses bf16 inputs with
f32 accumulation for ~2x throughput.

Public API:
    pallas_matmul       — y = x @ w + bias   (Pallas fwd + bwd on TPU)
    fused_derf_linear   — y = (γ·erf(α·x+s)+β) @ w + bias  (Pallas fwd, JAX bwd)
"""

import functools
import logging

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


# ===========================================================================
# Pallas matmul kernel (fwd + bwd)
# ===========================================================================

def _matmul_kernel(x_ref, w_ref, out_ref):
    """Tiled matmul: accumulates x_block @ w_block over K tiles."""
    k_iter = pl.program_id(2)

    x_blk = x_ref[...].astype(jnp.float32)   # (BM, BK)
    w_blk = w_ref[...].astype(jnp.float32)   # (BK, BN)

    partial = lax.dot_general(
        x_blk, w_blk,
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )                                          # (BM, BN)

    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def _pallas_matmul_2d(a, b):
    """a: (M, K) @ b: (K, N) -> (M, N).  All dims must be multiples of 128."""
    M, K = a.shape
    N = b.shape[1]

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((_BM, _BK), lambda m, n, k: (m, k)),
                pl.BlockSpec((_BK, _BN), lambda m, n, k: (k, n)),
            ],
            out_specs=pl.BlockSpec((_BM, _BN), lambda m, n, k: (m, n)),
            grid=(M // _BM, N // _BN, K // _BK),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)


def _can_use_pallas(M, K, N):
    return (
        jax.default_backend() == "tpu"
        and M % _BM == 0   # M must be divisible by smallest BM (128)
        and K % _BK == 0   # K must be divisible by smallest BK (128)
        and N % _BN == 0   # N must be 128-aligned for MXU
    )


# ---------------------------------------------------------------------------
# Public API: pallas_matmul  —  y = x @ weight + bias
# ---------------------------------------------------------------------------

def _matmul_ref(x_2d, weight, bias):
    """Pure JAX fallback."""
    return x_2d @ weight + bias


@functools.partial(jax.custom_vjp)
def pallas_matmul(x, weight, bias):
    """Linear projection y = x @ weight + bias.

    Uses a Pallas matmul kernel on TPU when all dimensions are multiples of 128.
    Falls back to pure JAX otherwise.

    Args:
        x:      (..., K) input.
        weight: (K, N) projection matrix.
        bias:   (N,) bias vector.

    Returns:
        (..., N) output.
    """
    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    x_2d = x.reshape(M, K)

    if _can_use_pallas(M, K, N):
        out_2d = _pallas_matmul_2d(x_2d, weight) + bias
    else:
        logging.warning(
            f"pallas_matmul silent fallback to JAX. "
            f"Shapes M={M}, K={K}, N={N} not multiples of 128."
        )
        out_2d = _matmul_ref(x_2d, weight, bias)

    return out_2d.reshape(*leading, N)


def _matmul_fwd(x, weight, bias):
    out = pallas_matmul(x, weight, bias)
    return out, (x, weight)


def _matmul_bwd(res, g):
    x, weight = res

    leading = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[-1]
    M = 1
    for d in leading:
        M *= d

    x_2d = x.reshape(M, K)
    g_2d = g.reshape(M, N)

    if _can_use_pallas(M, K, N):
        # dx = g @ weight^T   (M,N) @ (N,K) -> (M,K)
        dx_2d = _pallas_matmul_2d(g_2d, weight.T)
        # dw = x^T @ g        (K,M) @ (M,N) -> (K,N)
        dw = _pallas_matmul_2d(x_2d.T, g_2d)
    else:
        dx_2d = g_2d @ weight.T
        dw = x_2d.T @ g_2d

    dx = dx_2d.reshape(x.shape)
    db = g_2d.sum(axis=0)

    return (dx, dw, db)


pallas_matmul.defvjp(_matmul_fwd, _matmul_bwd)


# ===========================================================================
# Fused Derf + linear kernel
# ===========================================================================

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
def _fused_derf_kernel(x_ref, w_ref, alpha_ref, s_ref, gamma_ref, beta_ref,
                       bias_ref, out_ref):
    """Fused derf + matmul kernel body.

    Grid: (M//BM, K//BK) — N is NOT a grid dimension.
    Weight and output tiles span the full N width, so derf is computed
    only once per (m, k) pair.  The MXU uses bf16 inputs with f32
    accumulation for ~2x throughput vs the previous f32-only version.

    Block shapes:
        x_ref:     (BM, BK) — tiled along M and K
        w_ref:     (BK, N)  — full N width, tiled along K only
        alpha_ref: (1, 1)   — scalar broadcast
        s_ref:     (1, 1)   — scalar broadcast
        gamma_ref: (1, BK)  — tiled along K, broadcasts along M
        beta_ref:  (1, BK)  — tiled along K, broadcasts along M
        bias_ref:  (1, N)   — full N width
        out_ref:   (BM, N)  — full N width, tiled along M only
    """
    k_iter = pl.program_id(1)

    x_blk = x_ref[...].astype(jnp.float32)          # (BM, BK)
    w_blk = w_ref[...]                               # (BK, N)
    alpha = alpha_ref[...].astype(jnp.float32)       # (1, 1)
    s     = s_ref[...].astype(jnp.float32)           # (1, 1)
    gamma = gamma_ref[...].astype(jnp.float32)       # (1, BK)
    beta  = beta_ref[...].astype(jnp.float32)        # (1, BK)
    bias  = bias_ref[...].astype(jnp.float32)        # (1, N)

    # ---- Fused Derf (VPU, f32) ----
    # erf(x) ≈ tanh(2/√π · (x + 0.08943·x³))  (lax.erf has no Pallas lowering)
    u = alpha * x_blk + s                            # (BM, BK)
    erf_approx = lax.tanh(_TWO_OVER_SQRT_PI * (u + _ERF_COEFF * u * u * u))
    normed = gamma * erf_approx + beta               # (BM, BK)

    # ---- Matmul (MXU, bf16 inputs → f32 accumulation) ----
    partial = lax.dot_general(
        normed.astype(jnp.bfloat16),
        w_blk.astype(jnp.bfloat16),
        dimension_numbers=_MATMUL_DIMS,
        preferred_element_type=jnp.float32,
    )                                                # (BM, N)

    # ---- Accumulate ----
    @pl.when(k_iter == 0)
    def _():
        out_ref[...] = partial + bias

    @pl.when(k_iter != 0)
    def _():
        out_ref[...] += partial


def _vmem_estimate(bm, bk, N):
    """Estimate total VMEM usage for (bm, bk, N) tile config.

    Mosaic double-buffers all tiled inputs/outputs for DMA pipelining,
    and the kernel body creates intermediate arrays (normed, u, bf16 casts).
    """
    B = 4  # bytes per f32 element
    # Double-buffered tiles: x, weight, output
    double_buf = 2 * (bm * bk + bk * N + bm * N) * B
    # Kernel intermediates: u, erf_approx, normed, bf16 casts (~3 x-sized)
    intermediates = 3 * bm * bk * B
    # Compiler scratch (params, small buffers, alignment padding)
    scratch = 1 * 1024 * 1024
    return double_buf + intermediates + scratch


_VMEM_CAPACITY = 16 * 1024 * 1024   # TPU v3+ VMEM per core


def _choose_block_sizes(M, K, N):
    """Pick (BM, BK) to maximise tile size while fitting in VMEM.

    Strategy: use BK = K (full reduction in one tile) when possible,
    then pick the largest BM that keeps total VMEM under capacity.
    Larger tiles → fewer grid points, better MXU utilisation,
    and fewer K tiles means derf is computed fewer times per M-tile.
    """
    # Try BK from largest to smallest, BM from largest to smallest.
    for bk in [K, 512, 384, 256, _BK]:
        if K % bk != 0:
            continue
        for bm in [512, 256, _BM]:
            if M % bm != 0:
                continue
            if _vmem_estimate(bm, bk, N) <= _VMEM_CAPACITY:
                return bm, bk
    return _BM, _BK


def _pallas_forward(x_2d, weight, bias, alpha, s, gamma, beta):
    """Invoke the Pallas kernel.  x_2d: (M, K), weight: (K, N).

    Uses a 2D grid (M, K) with adaptive tile sizes.  Prefers BK = K
    (full reduction in one tile, no K loop) and the largest BM that
    fits in VMEM.  Weight and output tiles span the full N width.
    """
    M, K = x_2d.shape
    N = weight.shape[1]
    bm, bk = _choose_block_sizes(M, K, N)

    alpha_2d = jnp.array([[alpha]], dtype=jnp.float32)   # (1, 1)
    s_2d     = jnp.array([[s]],     dtype=jnp.float32)   # (1, 1)
    gamma_2d = gamma.reshape(1, K)                        # (1, K)
    beta_2d  = beta.reshape(1, K)                         # (1, K)
    bias_2d  = bias.reshape(1, N)                         # (1, N)

    k_tiles = K // bk
    # When K fits in a single tile, no reduction loop — use "parallel".
    k_sem = "parallel" if k_tiles == 1 else "arbitrary"

    return pl.pallas_call(
        _fused_derf_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda m, k: (m, k)),  # x
                pl.BlockSpec((bk, N),  lambda m, k: (k, 0)),  # weight (full N)
                pl.BlockSpec((1, 1),   lambda m, k: (0, 0)),   # alpha
                pl.BlockSpec((1, 1),   lambda m, k: (0, 0)),   # s
                pl.BlockSpec((1, bk),  lambda m, k: (0, k)),   # gamma
                pl.BlockSpec((1, bk),  lambda m, k: (0, k)),   # beta
                pl.BlockSpec((1, N),   lambda m, k: (0, 0)),   # bias (full N)
            ],
            out_specs=pl.BlockSpec((bm, N), lambda m, k: (m, 0)),
            grid=(M // bm, k_tiles),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", k_sem),
        ),
    )(x_2d, weight, alpha_2d, s_2d, gamma_2d, beta_2d, bias_2d)


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
        logging.warning(
            f"fused_derf_linear silent fallback to JAX. "
            f"Shapes M={M}, K={K}, N={N} not multiples of 128."
        )
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

    # Recompute intermediates matching the forward approximation.
    u = alpha * x + s
    inner = _TWO_OVER_SQRT_PI * (u + _ERF_COEFF * u * u * u)
    erf_approx = lax.tanh(inner)
    normed = gamma * erf_approx + beta                                # (..., K)

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

    # erf'(u) approximation derivative:
    # d(tanh(inner))/du = (1 - tanh^2(inner)) * d(inner)/du
    erf_deriv = (1.0 - erf_approx * erf_approx) * _TWO_OVER_SQRT_PI * (1.0 + 3.0 * _ERF_COEFF * u * u)

    reduce_axes = tuple(range(len(leading)))
    d_gamma = (d_normed * erf_approx).sum(axis=reduce_axes)           # (K,)
    d_beta  = d_normed.sum(axis=reduce_axes)                     # (K,)
    d_x     = d_normed * gamma * erf_deriv * alpha               # (..., K)
    d_alpha = (d_normed * gamma * erf_deriv * x).sum()           # scalar
    d_s     = (d_normed * gamma * erf_deriv).sum()               # scalar

    return (d_x, d_weight, d_bias, d_alpha, d_s, d_gamma, d_beta)


fused_derf_linear.defvjp(_fwd, _bwd)
