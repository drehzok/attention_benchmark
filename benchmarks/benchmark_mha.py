"""Benchmark: Naive custom MHA encoder vs Flax built-in MHA encoder."""

import sys
import os

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from flax import nnx

from src.models import BertConfig, NaiveEncoderLayer, JaxEncoderLayer
from src.utils import BenchmarkTimer, warmup_jit

NUM_RUNS = 20
BATCH_SIZE = 8
SEQ_LEN = 128


def main():
    config = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.0,  # deterministic for benchmarking
    )

    rngs = nnx.Rngs(0)
    naive_layer = NaiveEncoderLayer(config, rngs=rngs)
    jax_layer = JaxEncoderLayer(config, rngs=rngs)

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (BATCH_SIZE, SEQ_LEN, config.hidden_size))

    # JIT-compile both forward passes
    @jax.jit
    def run_naive(x):
        return naive_layer(x)

    @jax.jit
    def run_jax(x):
        return jax_layer(x)

    # Warmup
    warmup_jit(run_naive, x)
    warmup_jit(run_jax, x)

    # Benchmark
    timer_naive = BenchmarkTimer()
    timer_jax = BenchmarkTimer()

    for _ in range(NUM_RUNS):
        with timer_naive.measure():
            jax.block_until_ready(run_naive(x))

    for _ in range(NUM_RUNS):
        with timer_jax.measure():
            jax.block_until_ready(run_jax(x))

    print("=" * 60)
    print("MHA Encoder Layer Benchmark")
    print(f"  batch_size={BATCH_SIZE}  seq_len={SEQ_LEN}  hidden={config.hidden_size}")
    print("=" * 60)
    print(f"  Naive (custom MHA) : {timer_naive.summary()}")
    print(f"  Jax   (flax MHA)   : {timer_jax.summary()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
