from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nnx.Module):
    def __init__(self, d_model, max_len=5000):
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        position = jnp.arange(max_len)[:, jnp.newaxis]

        pe_sin = jnp.sin(position * div_term)
        pe_cos = jnp.cos(position * div_term)

        pe = jnp.stack([pe_sin, pe_cos], axis=-1).reshape(max_len, d_model)
        pe = pe[jnp.newaxis, :, :]

        self.pe = pe

    def __call__(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class BertEmbeddings(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size

        self.word_embeddings = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=self.hidden_size,
            rngs=rngs,
        )

        self.position_embeddings = SinusoidalPositionalEncoding(
            d_model=self.hidden_size,
            max_len=config.max_position_embeddings,
        )

        self.token_type_embeddings = nnx.Embed(
            num_embeddings=config.type_vocab_size,
            features=self.hidden_size,
            rngs=rngs,
        )

        self.LayerNorm = nnx.LayerNorm(self.hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids=None, rngs=None):
        words_emb = self.word_embeddings(input_ids)
        embeddings = self.position_embeddings(words_emb)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        if rngs is not None:
            embeddings = self.dropout(embeddings, rngs=rngs)

        return embeddings


# ---------------------------------------------------------------------------
# Attention & encoder layers
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None, rngs=None):
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(query.shape[-1])

    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)

    p_attn = jax.nn.softmax(scores, axis=-1)

    if dropout is not None and rngs is not None:
        p_attn = dropout(p_attn, rngs=rngs)

    return jnp.matmul(p_attn, value)


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "num_heads should divide d_model"

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.w_o = nnx.Linear(d_model, d_model, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout)

    def __call__(self, query, key, value, mask=None, rngs=None):
        batch_size = query.shape[0]

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        x = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout, rngs=rngs)

        x = x.transpose((0, 2, 1, 3)).reshape(batch_size, -1, self.num_heads * self.head_dim)

        return self.w_o(x)


class PositionwiseFeedForward(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, rngs: nnx.Rngs, dropout=0.1):
        self.w_1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.w_2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout)

    def __call__(self, x, rngs=None):
        return self.w_2(self.dropout(jax.nn.gelu(self.w_1(x)), rngs=rngs))


class NaiveEncoderLayer(nnx.Module):
    def __init__(self, config, rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads, rngs=rngs, dropout=config.hidden_dropout_prob)
        self.dropout1 = nnx.Dropout(config.hidden_dropout_prob)

        self.norm2 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.feed_forward = PositionwiseFeedForward(config.hidden_size, config.intermediate_size, rngs=rngs, dropout=config.hidden_dropout_prob)
        self.dropout2 = nnx.Dropout(config.hidden_dropout_prob)

    def __call__(self, x, mask=None, rngs=None):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, norm_x, norm_x, mask=mask, rngs=rngs)
        x = x + self.dropout1(attn_out, rngs=rngs)

        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x, rngs=rngs)
        x = x + self.dropout2(ff_out, rngs=rngs)
        return x


class JaxEncoderLayer(nnx.Module):
    def __init__(self, config, rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(config.num_attention_heads, config.hidden_size, rngs=rngs, dropout_rate=config.hidden_dropout_prob, decode=False)
        self.dropout1 = nnx.Dropout(config.hidden_dropout_prob)

        self.norm2 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.feed_forward = PositionwiseFeedForward(config.hidden_size, config.intermediate_size, rngs=rngs, dropout=config.hidden_dropout_prob)
        self.dropout2 = nnx.Dropout(config.hidden_dropout_prob)

    def __call__(self, x, mask=None, rngs=None):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, norm_x, norm_x, mask=mask, rngs=rngs)
        x = x + self.dropout1(attn_out, rngs=rngs)

        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x, rngs=rngs)
        x = x + self.dropout2(ff_out, rngs=rngs)
        return x


# ---------------------------------------------------------------------------
# Derf (erf-based normalization) models
# ---------------------------------------------------------------------------
class Derf(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jnp.array(0.5))
        self.s = nnx.Param(jnp.array(0.0))
    
        self.gamma = nnx.Param(jnp.ones((dim,)))
        self.beta = nnx.Param(jnp.zeros((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.gamma * jax.lax.erf(self.alpha * x + self.s) + self.beta


class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = Derf(dim, rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)

        self.norm2 = Derf(dim, rngs)
        self.mlp_fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array, deterministic: bool = False) -> jax.Array:
        resid = x

        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout1(x, deterministic=deterministic)
        x = resid + x

        resid = x

        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_fc2(x)
        x = self.dropout2(x, deterministic=deterministic)
        x = resid + x

        return x


CheckpointBlock = nnx.remat(TransformerBlock)


class NormalTransformerBlock(nnx.Module):
    """Same as TransformerBlock but uses nnx.LayerNorm instead of Derf."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)

        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array, deterministic: bool = False) -> jax.Array:
        resid = x

        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout1(x, deterministic=deterministic)
        x = resid + x

        resid = x

        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_fc2(x)
        x = self.dropout2(x, deterministic=deterministic)
        x = resid + x

        return x


class NormalBert(nnx.Module):
    """Same as DerfBert but uses NormalTransformerBlock + LayerNorm final norm."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.num_layers = num_layers

        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.pos_embed = nnx.Embed(512, dim, rngs=rngs)
        self.final_norm = nnx.LayerNorm(dim, rngs=rngs)

        self.layers = nnx.List([
            NormalTransformerBlock(dim, num_heads, mlp_dim, dropout_rate, rngs)
            for _ in range(num_layers)
        ])

        self.head = nnx.Linear(dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array, mask: jax.Array = None, deterministic: bool = False) -> jax.Array:
        seq_len = input_ids.shape[1]

        pos_ids = jnp.arange(seq_len)[None, :]
        x = self.embed(input_ids) + self.pos_embed(pos_ids)

        for layer in self.layers:
            x = layer(x, mask, deterministic=deterministic)

        x = self.final_norm(x)
        logits = self.head(x)

        return logits


class DerfBert(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.num_layers = num_layers

        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.pos_embed = nnx.Embed(512, dim, rngs=rngs)
        self.final_norm = Derf(dim, rngs)

        self.layers = nnx.List([
            TransformerBlock(dim, num_heads, mlp_dim, dropout_rate, rngs)
            for _ in range(num_layers)
        ])

        self.head = nnx.Linear(dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array, mask: jax.Array = None, deterministic: bool = False) -> jax.Array:
        seq_len = input_ids.shape[1]

        pos_ids = jnp.arange(seq_len)[None, :]
        x = self.embed(input_ids) + self.pos_embed(pos_ids)

        for layer in self.layers:
            x = layer(x, mask, deterministic=deterministic)

        x = self.final_norm(x)
        logits = self.head(x)

        return logits
