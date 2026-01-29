import logging
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp


class BenchmarkTimer:
    """Context manager that measures wall-clock time and records results."""

    def __init__(self):
        self.times: list[float] = []

    @contextmanager
    def measure(self):
        # Block until all JAX computations finish so we time accurately.
        jax.block_until_ready(None) if False else None  # noqa – placeholder
        start = time.perf_counter()
        yield
        # Ensure JAX async dispatch is done before stopping the clock.
        jax.effects_barrier()
        elapsed = time.perf_counter() - start
        self.times.append(elapsed)

    @property
    def mean(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        if not self.times:
            return 0.0
        s = sorted(self.times)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2
        return s[mid]

    def summary(self) -> str:
        if not self.times:
            return "No measurements recorded."
        return (
            f"runs={len(self.times)}  "
            f"mean={self.mean * 1000:.2f}ms  "
            f"median={self.median * 1000:.2f}ms  "
            f"min={min(self.times) * 1000:.2f}ms  "
            f"max={max(self.times) * 1000:.2f}ms"
        )


def generate_random_inputs(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 30522,
    seed: int = 0,
):
    """Create random input_ids and an all-ones attention mask."""
    key = jax.random.PRNGKey(seed)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    return input_ids, attention_mask


def setup_logger(name: str = "benchmark", level: int = logging.INFO) -> logging.Logger:
    """Return a simple console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def warmup_jit(fn, *args, **kwargs):
    """Run *fn* once and block until the result is ready (triggers JIT)."""
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out
