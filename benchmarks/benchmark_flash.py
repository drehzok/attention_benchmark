"""Benchmark: manual scaled_dot_product_attention vs jax.nn.dot_product_attention."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from src.models import scaled_dot_product_attention
from src.utils import BenchmarkTimer, warmup_jit

NUM_RUNS = 20
BATCH_SIZE = 8
NUM_HEADS = 12
HEAD_DIM = 64
SEQ_LENGTHS = [64, 128, 256, 512]


def main():
    key = jax.random.PRNGKey(0)

    print("=" * 70)
    print("Flash-Attention Kernel Benchmark  (raw Q/K/V attention)")
    print(f"  batch={BATCH_SIZE}  heads={NUM_HEADS}  head_dim={HEAD_DIM}")
    print("=" * 70)
    print(f"{'seq_len':>8}  {'manual (ms)':>14}  {'jax.nn (ms)':>14}  {'speedup':>8}")
    print("-" * 70)

    for seq_len in SEQ_LENGTHS:
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM))
        k = jax.random.normal(k2, (BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM))
        v = jax.random.normal(k3, (BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM))

        @jax.jit
        def run_manual(q, k, v):
            return scaled_dot_product_attention(q, k, v)

        @jax.jit
        def run_jax_nn(q, k, v):
            return jax.nn.dot_product_attention(q, k, v)

        warmup_jit(run_manual, q, k, v)
        warmup_jit(run_jax_nn, q, k, v)

        timer_manual = BenchmarkTimer()
        timer_jax = BenchmarkTimer()

        for _ in range(NUM_RUNS):
            with timer_manual.measure():
                jax.block_until_ready(run_manual(q, k, v))

        for _ in range(NUM_RUNS):
            with timer_jax.measure():
                jax.block_until_ready(run_jax_nn(q, k, v))

        speedup = timer_manual.mean / timer_jax.mean if timer_jax.mean else float("inf")
        print(f"{seq_len:>8}  {timer_manual.mean * 1000:>14.2f}  {timer_jax.mean * 1000:>14.2f}  {speedup:>7.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
