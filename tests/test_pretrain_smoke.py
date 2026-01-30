"""Smoke test: verify all 3 BERT models work with the pretrain training step.

Tests encode(), __call__, SPMD sharding, and gradient updates with synthetic data.
No HuggingFace dependencies required.

Usage (on GCE TPU):
    python tests/test_pretrain_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.models import DerfBert, NormalBert, FusedDerfBert

# Small config for smoke test
CONFIG = dict(num_layers=2, dim=128, num_heads=4, mlp_dim=512)
VOCAB_SIZE = 256
SEQ_LEN = 32
DROPOUT_RATE = 0.0


def make_synthetic_batch(batch_size, seq_len, vocab_size, seed=42):
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
    labels = np.full((batch_size, seq_len), -100, dtype=np.int32)
    mask_pos = rng.random((batch_size, seq_len)) < 0.15
    labels[mask_pos] = input_ids[mask_pos]
    return jnp.array(input_ids), jnp.array(labels)


def make_train_step():
    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(model):
            logits = model(input_ids, deterministic=False)
            vocab_size = logits.shape[-1]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            flat_log_probs = log_probs.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)
            safe_labels = jnp.maximum(flat_labels, 0)
            token_losses = -flat_log_probs[
                jnp.arange(flat_log_probs.shape[0]), safe_labels
            ]
            mask = flat_labels != -100
            loss = jnp.sum(token_losses * mask) / jnp.maximum(jnp.sum(mask), 1)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def test_model(model_type, cls, mesh, replicate, data_shard, batch_size):
    """Run all checks for one model type. Returns (passed: bool, message: str)."""
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        num_layers=CONFIG["num_layers"],
        dim=CONFIG["dim"],
        num_heads=CONFIG["num_heads"],
        mlp_dim=CONFIG["mlp_dim"],
        dropout_rate=DROPOUT_RATE,
        rngs=nnx.Rngs(42),
    )
    if model_type == "fused":
        kwargs["mesh"] = mesh

    model = cls(**kwargs)
    input_ids, labels = make_synthetic_batch(batch_size, SEQ_LEN, VOCAB_SIZE)

    # 1. encode() returns hidden states
    hidden = model.encode(input_ids, deterministic=True)
    assert hidden.shape == (batch_size, SEQ_LEN, CONFIG["dim"]), (
        f"encode() shape: {hidden.shape}"
    )
    assert bool(jnp.all(jnp.isfinite(hidden))), "encode() produced non-finite values"

    # 2. __call__ returns logits via encode()
    logits = model(input_ids, deterministic=True)
    assert logits.shape == (batch_size, SEQ_LEN, VOCAB_SIZE), (
        f"logits shape: {logits.shape}"
    )
    assert bool(jnp.all(jnp.isfinite(logits))), "logits non-finite"

    # 3. SPMD: replicate model + shard data
    state = nnx.state(model)
    state = jax.device_put(state, replicate)
    nnx.update(model, state)

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, 1e-4, 10),
            optax.cosine_decay_schedule(1e-4, 90),
        ],
        boundaries=[10],
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=0.01)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    opt_state = nnx.state(optimizer)
    opt_state = jax.device_put(opt_state, replicate)
    nnx.update(optimizer, opt_state)

    train_step = make_train_step()

    ids_sharded = jax.device_put(input_ids, data_shard)
    labels_sharded = jax.device_put(labels, data_shard)

    # 4. Two training steps — loss must be finite
    loss1 = train_step(model, optimizer, ids_sharded, labels_sharded)
    loss2 = train_step(model, optimizer, ids_sharded, labels_sharded)
    l1, l2 = float(loss1), float(loss2)

    assert jnp.isfinite(loss1), f"loss1 non-finite: {l1}"
    assert jnp.isfinite(loss2), f"loss2 non-finite: {l2}"

    extra = " (loss decreasing)" if l2 < l1 else ""
    return True, f"loss {l1:.4f} -> {l2:.4f}{extra}"


def main():
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=("dp",))
    replicate = NamedSharding(mesh, P())
    data_shard = NamedSharding(mesh, P("dp"))

    batch_size = max(4, num_devices)
    batch_size = (batch_size // num_devices) * num_devices

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {num_devices}")
    print(f"Batch:   {batch_size} ({batch_size // num_devices} per device)")
    print()

    models = [("derf", DerfBert), ("normal", NormalBert), ("fused", FusedDerfBert)]
    passed = 0
    failed = 0

    for model_type, cls in models:
        try:
            ok, msg = test_model(
                model_type, cls, mesh, replicate, data_shard, batch_size
            )
            print(f"  {model_type:>8s}: PASS  {msg}")
            passed += 1
        except Exception as e:
            print(f"  {model_type:>8s}: FAIL  {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\nSmoke test: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
