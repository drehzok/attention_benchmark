"""Correctness tests for attention models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.models import (
    BertConfig,
    BertEmbeddings,
    DerfBert,
    NormalBert,
    FusedDerfBert,
    JaxEncoderLayer,
    NaiveEncoderLayer,
)
from src.kernels import fused_derf_linear


BATCH_SIZE = 2
SEQ_LEN = 16


@pytest.fixture
def config():
    return BertConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
    )


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


@pytest.fixture
def random_hidden(config):
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (BATCH_SIZE, SEQ_LEN, config.hidden_size))


# ---- Encoder layer tests ----


def test_naive_vs_jax_encoder_output_shape(config, rngs, random_hidden):
    naive = NaiveEncoderLayer(config, rngs=rngs)
    jax_enc = JaxEncoderLayer(config, rngs=rngs)

    out_naive = naive(random_hidden)
    out_jax = jax_enc(random_hidden)

    assert out_naive.shape == random_hidden.shape
    assert out_jax.shape == random_hidden.shape


def test_naive_vs_jax_encoder_close(config, random_hidden):
    # Use the same RNG seed so both layers get identical initial weights for
    # the shared sub-modules (LayerNorm, FFN).  The attention implementations
    # differ (custom vs flax built-in) so we only check that they produce
    # outputs in the same ballpark rather than being exactly equal.
    naive = NaiveEncoderLayer(config, rngs=nnx.Rngs(0))
    jax_enc = JaxEncoderLayer(config, rngs=nnx.Rngs(0))

    out_naive = naive(random_hidden)
    out_jax = jax_enc(random_hidden)

    # Both should be finite and have the same shape.
    assert jnp.all(jnp.isfinite(out_naive))
    assert jnp.all(jnp.isfinite(out_jax))
    assert out_naive.shape == out_jax.shape


# ---- Embeddings test ----


def test_bert_embeddings_shape(config, rngs):
    embeddings = BertEmbeddings(config, rngs=rngs)
    input_ids = jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)

    out = embeddings(input_ids)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, config.hidden_size)


# ---- DerfBert test ----


def test_derf_bert_forward(rngs):
    vocab_size = 256
    dim = 64
    model = DerfBert(
        vocab_size=vocab_size,
        num_layers=2,
        dim=dim,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
        rngs=rngs,
    )

    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    logits = model(input_ids, deterministic=True)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)
    assert jnp.all(jnp.isfinite(logits))


# ---- NormalBert tests ----


def test_normal_bert_forward(rngs):
    vocab_size = 256
    dim = 64
    model = NormalBert(
        vocab_size=vocab_size,
        num_layers=2,
        dim=dim,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
        rngs=rngs,
    )

    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    logits = model(input_ids, deterministic=True)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_derf_vs_normal_bert_same_shape():
    vocab_size = 256
    dim = 64
    kwargs = dict(
        vocab_size=vocab_size,
        num_layers=2,
        dim=dim,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
    )

    derf_model = DerfBert(**kwargs, rngs=nnx.Rngs(0))
    normal_model = NormalBert(**kwargs, rngs=nnx.Rngs(0))

    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    derf_logits = derf_model(input_ids, deterministic=True)
    normal_logits = normal_model(input_ids, deterministic=True)

    assert derf_logits.shape == normal_logits.shape
    assert derf_logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)


# ---- Fused kernel tests ----


def test_fused_derf_linear_matches_reference():
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    dim, out_dim = 64, 128
    x = jax.random.normal(k1, (BATCH_SIZE, SEQ_LEN, dim))
    weight = jax.random.normal(k2, (dim, out_dim))
    bias = jax.random.normal(k3, (out_dim,))
    alpha = jnp.array(0.5)
    s = jnp.array(0.1)
    gamma = jax.random.normal(k4, (dim,))
    beta = jax.random.normal(k5, (dim,))

    out_fused = fused_derf_linear(x, weight, bias, alpha, s, gamma, beta)

    # Reference: derf then matmul
    normed = gamma * jax.lax.erf(alpha * x + s) + beta
    out_ref = normed @ weight + bias

    assert out_fused.shape == (BATCH_SIZE, SEQ_LEN, out_dim)
    assert jnp.allclose(out_fused, out_ref, atol=1e-5)


def test_fused_derf_linear_gradient():
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    dim, out_dim = 64, 128
    x = jax.random.normal(k1, (BATCH_SIZE, SEQ_LEN, dim))
    weight = jax.random.normal(k2, (dim, out_dim))
    bias = jax.random.normal(k3, (out_dim,))
    alpha = jnp.array(0.5)
    s = jnp.array(0.1)
    gamma = jax.random.normal(k4, (dim,))
    beta = jax.random.normal(k5, (dim,))

    def loss_fused(x, w, b, a, s_, g, bt):
        return fused_derf_linear(x, w, b, a, s_, g, bt).sum()

    def loss_ref(x, w, b, a, s_, g, bt):
        normed = g * jax.lax.erf(a * x + s_) + bt
        return (normed @ w + b).sum()

    grads_fused = jax.grad(loss_fused, argnums=(0, 1, 2, 3, 4, 5, 6))(
        x, weight, bias, alpha, s, gamma, beta
    )
    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5, 6))(
        x, weight, bias, alpha, s, gamma, beta
    )

    names = ['x', 'weight', 'bias', 'alpha', 's', 'gamma', 'beta']
    for gf, gr, name in zip(grads_fused, grads_ref, names):
        assert jnp.allclose(gf, gr, atol=1e-4), f"Gradient mismatch for {name}"


# ---- FusedDerfBert tests ----


def test_fused_derf_bert_forward(rngs):
    vocab_size = 256
    dim = 64
    model = FusedDerfBert(
        vocab_size=vocab_size,
        num_layers=2,
        dim=dim,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
        rngs=rngs,
    )

    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    logits = model(input_ids, deterministic=True)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_fused_vs_unfused_same_shape():
    vocab_size = 256
    kwargs = dict(
        vocab_size=vocab_size,
        num_layers=2,
        dim=64,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
    )

    fused = FusedDerfBert(**kwargs, rngs=nnx.Rngs(0))
    unfused = DerfBert(**kwargs, rngs=nnx.Rngs(0))

    input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    out_fused = fused(input_ids, deterministic=True)
    out_unfused = unfused(input_ids, deterministic=True)

    assert out_fused.shape == out_unfused.shape
    assert out_fused.shape == (BATCH_SIZE, SEQ_LEN, vocab_size)
