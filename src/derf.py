import jax
import jax.numpy as jnp
from flax import nnx
import optax  # Standard optimizer library for JAX

# -----------------------------------------------------------------------------
# 1. The Derf Layer (Replaces LayerNorm)
# -----------------------------------------------------------------------------
class Derf(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs):
        # Sanity Check
        assert dim > 0, f"Dim must be positive, got {dim}"

        # alpha: Scale (init 1.0 to preserve signal variance initially)
        # s:     Shift (init 0.0)
        self.alpha = nnx.Param(nnx.initializers.ones(rngs.params(), (dim,)))
        self.s = nnx.Param(nnx.initializers.zeros(rngs.params(), (dim,)))
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        # Fused pointwise operation: Derf(x) = erf(alpha * x + s)
        # This fuses perfectly on TPU/GPU VPUs
        return jax.lax.erf(self.alpha * x + self.s)

# -----------------------------------------------------------------------------
# 2. Transformer Block (Checkpointing + NNX)
# -----------------------------------------------------------------------------
class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads

        # 1. Attention Components
        self.norm1 = Derf(dim, rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            decode=False,
            rngs=rngs
        )
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs)

        # 2. MLP Components
        self.norm2 = Derf(dim, rngs)
        self.mlp_fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array, deterministic: bool = False) -> jax.Array:
        resid = x

        # --- Attention Sub-Layer ---
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout1(x, deterministic=deterministic)
        x = resid + x

        resid = x

        # --- MLP Sub-Layer ---
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = jax.nn.gelu(x)
        x = self.mlp_fc2(x)
        x = self.dropout2(x, deterministic=deterministic)
        x = resid + x

        return x

# Apply Gradient Checkpointing (Remat)
# This saves memory by recomputing the block during backward pass
CheckpointBlock = nnx.remat(TransformerBlock)

# -----------------------------------------------------------------------------
# 3. The BERT Backbone (Stacked with Vmap + Scan)
# -----------------------------------------------------------------------------
class DerfBert(nnx.Module):
    def __init__(self,
                 vocab_size: int,
                 num_layers: int,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout_rate: float,
                 rngs: nnx.Rngs):

        self.dim = dim
        self.num_layers = num_layers

        # 1. Embeddings
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.pos_embed = nnx.Embed(512, dim, rngs=rngs) # Fixed max len 512
        self.final_norm = Derf(dim, rngs)

        # 2. Creating the Stack of Layers
        # We use nnx.vmap to create a "Stack" of blocks
        def create_block(rngs_slice):
            return CheckpointBlock(dim, num_heads, mlp_dim, dropout_rate, rngs_slice)

        # --- FIX: Use .fork() to split RNGs for the layer stack ---
        # This creates 'num_layers' independent RNG streams
        layer_rngs = rngs.fork(num_layers)

        self.layers = nnx.vmap(create_block, in_axes=0, out_axes=0)(layer_rngs)

        # 3. Output Head (Simple Masked LM style)
        self.head = nnx.Linear(dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array, mask: jax.Array = None, deterministic: bool = False) -> jax.Array:
        seq_len = input_ids.shape[1]

        # 1. Embeddings
        pos_ids = jnp.arange(seq_len)[None, :]
        x = self.embed(input_ids) + self.pos_embed(pos_ids)

        # 2. SCAN Logic (The Loop)
        # We must split the 'layers' object into (Graph, State) to scan over the State
        graph_def, layer_states = nnx.split(self.layers)

        def scan_fn(carry, state_slice):
            x_curr = carry
            # Reconstruct the block for this specific layer index
            current_layer = nnx.merge(graph_def, state_slice)

            # Run block
            x_next = current_layer(x_curr, mask, deterministic=deterministic)

            return x_next, None

        # Execute scan
        x, _ = jax.lax.scan(scan_fn, x, layer_states)

        # 3. Final Norm & Head
        x = self.final_norm(x)
        logits = self.head(x)

        return logits
