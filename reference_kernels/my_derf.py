import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np

def matmul_kernel(
    x_ref, y_ref, gamma_ref, beta_ref, alpha_ref, s_ref, acc_ref,
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)


  acc_ref[...] += jax.lax.dot_general(
      gamma_ref[...]*jax.lax.erf(
          alpha_ref[...]*x_ref[...]+s_ref[...] #I am not sure if alpha_ref and s_ref should be done this way
      ) + beta_ref[...],
      y_ref[...],
      dims,
      preferred_element_type=jnp.float32,
  )



@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    alpha: jax.Array,
    s: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):

  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      matmul_kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
              pl.BlockSpec((bk), lambda i, j, k: (k,)), #is this correct?
              pl.BlockSpec((bk), lambda i, j, k: (k,)), #is this correct?
              pl.BlockSpec((bk), lambda i, j, k: (0,)), #is this correct?
              pl.BlockSpec((bk), lambda i, j, k: (0,)), #is this correct?
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          grid=(m // bm, n // bn, k // bk),
      ),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y, gamma, beta, alpha, s)
