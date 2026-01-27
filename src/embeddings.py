import jax
import flax

import jax.numpy as jnp
from flax import nnx

class SinusoidalPositionalEncoding(nnx.Module):
  def __init__(self, d_model, max_len=5000):
    # 1. Create the frequency term (10000^(2i/d_model))
    # We do this in log space for numerical stability
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))

    # 2. Create position indices [0, 1, ... max_len-1]
    position = jnp.arange(max_len)[:, jnp.newaxis] # Shape (max_len, 1)

    # 3. Compute Sine and Cosine
    # (max_len, 1) * (d_model/2) -> (max_len, d_model/2)
    pe_sin = jnp.sin(position * div_term)
    pe_cos = jnp.cos(position * div_term)

    # 4. Interleave them efficiently
    # We stack them to shape (max_len, d_model/2, 2) and flatten the last two dims
    pe = jnp.stack([pe_sin, pe_cos], axis=-1).reshape(max_len, d_model)

    # 5. Add a batch dimension for broadcasting
    # Shape: (1, max_len, d_model)
    pe = pe[jnp.newaxis, :, :]

    # Store as a constant (not a trainable parameter!)
    # In NNX, standard attributes are ignored by the optimizer unless they are nnx.Param
    self.pe = pe

  def __call__(self, x):
    """
    Args:
        x: Input embeddings of shape (Batch, SeqLen, Dim)
    Returns:
        x + positional_encoding
    """
    # Slicing is cheap in JAX
    seq_len = x.shape[1]

    # We simply slice our pre-computed table to the current sequence length
    # and add it to the input.
    # Broadcasting handles the batch dimension automatically.
    return x + self.pe[:, :seq_len, :]


class BertEmbeddings(nnx.Module):
  def __init__(self, config, rngs: nnx.Rngs):
    self.hidden_size = config.hidden_size

    # 1. Token Embeddings (Still Learnable)
    self.word_embeddings = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=self.hidden_size,
        rngs=rngs
    )

    # 2. Position Embeddings (Swapped to Sinusoidal)
    # Note: No 'rngs' needed here because there are no random weights!
    self.position_embeddings = SinusoidalPositionalEncoding(
        d_model=self.hidden_size,
        max_len=config.max_position_embeddings
    )

    self.token_type_embeddings = nnx.Embed(
        num_embeddings=config.type_vocab_size,
        features=self.hidden_size,
        rngs=rngs
    )

    self.LayerNorm = nnx.LayerNorm(self.hidden_size, rngs=rngs)
    self.dropout = nnx.Dropout(rate=config.hidden_dropout_prob)

  def __call__(self, input_ids, token_type_ids=None, rngs=None):
    # 1. Get Word Embeddings
    words_emb = self.word_embeddings(input_ids)

    # 2. Add Sinusoidal Position Embeddings
    # The class handles the addition logic internally
    embeddings = self.position_embeddings(words_emb)

    # 3. Add Token Types (Segment A/B)
    if token_type_ids is None:
        token_type_ids = jnp.zeros_like(input_ids)
    embeddings = embeddings + self.token_type_embeddings(token_type_ids)

    # 4. Norm + Dropout
    embeddings = self.LayerNorm(embeddings)
    if rngs is not None:
        embeddings = self.dropout(embeddings, rngs=rngs)

    return embeddings