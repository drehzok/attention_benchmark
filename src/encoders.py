import jax
import flax

import jax.numpy as jnp
from flax import nnx


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None, rngs=None):
  scores = jnp.matmul(query, jnp.swapaxes(key,-2,-1))/jnp.sqrt(query.shape[-1])

  if mask is not None:
    scores = jnp.where(mask == 0, -1e9, scores)

  p_attn = jax.nn.softmax(scores,axis=-1)

  if dropout is not None and rngs is not None:
    p_attn = dropout(p_attn,rngs=rngs)

  return jnp.matmul(p_attn, value)

class MultiHeadAttention(nnx.Module):
  def __init__(self, d_model:int, num_heads:int, rngs:nnx.Rngs, dropout=0.1):
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

    x = scaled_dot_product_attention(q,k,v,mask=mask,dropout=self.dropout,rngs=rngs)

    x = x.transpose((0,2,1,3)).reshape(batch_size,-1,self.num_heads*self.head_dim)

    return self.w_o(x)




## I need naive, jax multihead, derf implementation


class PositionwiseFeedForward(nnx.Module):
    def __init__(self, d_model:int, d_ff:int, rngs:nnx.Rngs, dropout=0.1):
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
        self.attn = nnx.MultiHeadAttention(config.num_attention_heads, config.hidden_size, rngs=rngs, dropout=config.hidden_dropout_prob)
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
