"""Micro-benchmark: Pallas matmul vs XLA matmul on TPU.

Isolates the matmul performance gap to confirm whether the fused Derf+Linear
kernel slowdown is caused by the Pallas matmul itself.

Usage (on TPU):
    python tests/test_matmul_perf.py
"""

import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import _pallas_matmul_2d


def bench(fn, a, b, warmup=5, steps=20):
    for _ in range(warmup):
        fn(a, b).block_until_ready()

    times = []
    for _ in range(steps):
        t = time.perf_counter()
        fn(a, b).block_until_ready()
        times.append(time.perf_counter() - t)

    import numpy as np
    times = np.array(times) * 1000
    return np.median(times), np.std(times)


def main():
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}\n")

    configs = [
        # (M, K, N, description)
        (8192, 768, 2304, "QKV proj  (batch=32, seq=256, dim=768 -> 3*dim)"),
        (8192, 768, 3072, "MLP fc1   (batch=32, seq=256, dim=768 -> mlp=3072)"),
        (8192, 3072, 768, "MLP fc2   (batch=32, seq=256, mlp=3072 -> dim=768)"),
        (16384, 768, 3072, "MLP fc1   (batch=64, seq=256)"),
        (32768, 768, 3072, "MLP fc1   (batch=128, seq=256)"),
        (16384, 768, 2304, "QKV proj  (batch=32, seq=512)"),
    ]

    f_xla = jax.jit(lambda a, b: a @ b)
    f_pallas = jax.jit(_pallas_matmul_2d)

    print(f"{'config':>45s} | {'XLA':>10s} | {'Pallas':>10s} | {'ratio':>7s}")
    print(f"{'-' * 45} | {'-' * 10} | {'-' * 10} | {'-' * 7}")

    for M, K, N, desc in configs:
        a = jnp.ones((M, K), dtype=jnp.float32)
        b = jnp.ones((K, N), dtype=jnp.float32)

        xla_ms, _ = bench(f_xla, a, b)
        pal_ms, _ = bench(f_pallas, a, b)
        ratio = pal_ms / xla_ms

        print(f"{desc:>45s} | {xla_ms:>8.2f}ms | {pal_ms:>8.2f}ms | {ratio:>5.2f}x")


if __name__ == "__main__":
    main()
