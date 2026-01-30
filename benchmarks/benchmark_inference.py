"""Inference speed benchmark: Derf vs LayerNorm vs Fused Derf.

Measures forward-pass-only throughput to isolate the fused Pallas kernel
speedup (backward uses JAX for all models).

Usage:
    python benchmarks/benchmark_inference.py --size base
    python benchmarks/benchmark_inference.py --size base --wandb
    python benchmarks/benchmark_inference.py --size base --seq-lens 128 256 512
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DerfBert, NormalBert, FusedDerfBert

CONFIGS = {
    "base": dict(num_layers=12, dim=768, num_heads=12, mlp_dim=3072),
    "small": dict(num_layers=4, dim=512, num_heads=8, mlp_dim=2048),
}

MODEL_CLASSES = {
    "derf": DerfBert,
    "normal": NormalBert,
    "fused": FusedDerfBert,
}


def benchmark_model(model_type, config, vocab_size, batch_size, seq_len,
                    warmup_steps, bench_steps, mesh):
    """Benchmark forward pass for one model. Returns dict of results."""
    replicate = NamedSharding(mesh, P())
    data_shard = NamedSharding(mesh, P('dp'))

    rngs = nnx.Rngs(0)
    kwargs = dict(
        vocab_size=vocab_size,
        num_layers=config["num_layers"],
        dim=config["dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=0.0,
        rngs=rngs,
    )
    if model_type == "fused":
        kwargs["mesh"] = mesh
    model = MODEL_CLASSES[model_type](**kwargs)

    # Replicate model across devices
    state = nnx.state(model)
    state = jax.device_put(state, replicate)
    nnx.update(model, state)

    # Shard input across devices
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    input_ids = jax.device_put(input_ids, data_shard)

    @nnx.jit
    def forward(model, input_ids):
        return model(input_ids, deterministic=True)

    # Warmup (JIT compile + cache)
    for _ in range(warmup_steps):
        out = forward(model, input_ids)
        jax.block_until_ready(out)

    # Benchmark
    times = []
    for _ in range(bench_steps):
        start = time.perf_counter()
        out = forward(model, input_ids)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    tokens_per_step = batch_size * seq_len
    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    return {
        "model_type": model_type,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "median_ms": median_time * 1000,
        "mean_ms": mean_time * 1000,
        "std_ms": std_time * 1000,
        "median_tok_per_sec": tokens_per_step / median_time,
        "median_steps_per_sec": 1.0 / median_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference speed benchmark: Derf vs LayerNorm vs Fused"
    )
    parser.add_argument("--size", choices=["base", "small"], default="base")
    parser.add_argument("--models", nargs="+", default=["derf", "normal", "fused"],
                        help="Model types to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64],
                        help="Batch sizes to test")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 256, 512],
                        help="Sequence lengths to test")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Benchmark steps per config (default: 50)")
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="derf-inference")
    args = parser.parse_args()

    config = CONFIGS[args.size]

    # Set up SPMD mesh (same as pretrain.py)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('dp',))

    print(f"Inference Benchmark: {args.size}")
    print(f"  Backend:     {jax.default_backend()}")
    print(f"  Devices:     {len(devices)} ({devices})")
    print(f"  Config:      {config}")
    print(f"  Models:      {args.models}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Seq lengths: {args.seq_lens}")
    print(f"  Steps:       {args.warmup} warmup + {args.steps} bench")

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"inference-{args.size}",
            config={
                "size": args.size,
                **config,
                "models": args.models,
                "batch_sizes": args.batch_sizes,
                "seq_lens": args.seq_lens,
                "warmup": args.warmup,
                "steps": args.steps,
                "backend": jax.default_backend(),
                "num_devices": len(devices),
            },
        )

    all_results = []

    for seq_len in args.seq_lens:
        print(f"\n{'=' * 70}")
        print(f"seq_len={seq_len}")
        print(f"{'=' * 70}")
        print(f"  {'model':>8s} | {'batch':>5s} | {'median':>9s} | {'std':>8s} | {'tok/s':>12s} | {'steps/s':>8s}")
        print(f"  {'-' * 8} | {'-' * 5} | {'-' * 9} | {'-' * 8} | {'-' * 12} | {'-' * 8}")

        for batch_size in args.batch_sizes:
            for model_type in args.models:
                try:
                    r = benchmark_model(
                        model_type, config, args.vocab_size,
                        batch_size, seq_len,
                        args.warmup, args.steps, mesh,
                    )
                    all_results.append(r)

                    print(f"  {model_type:>8s} | {batch_size:>5d} | "
                          f"{r['median_ms']:>7.1f}ms | "
                          f"{r['std_ms']:>6.1f}ms | "
                          f"{r['median_tok_per_sec']:>10,.0f}   | "
                          f"{r['median_steps_per_sec']:>6.1f}")

                    if args.wandb:
                        wandb.log({
                            f"{model_type}/median_ms": r["median_ms"],
                            f"{model_type}/tok_per_sec": r["median_tok_per_sec"],
                            f"{model_type}/batch_size": batch_size,
                            f"{model_type}/seq_len": seq_len,
                        })

                except Exception as e:
                    print(f"  {model_type:>8s} | {batch_size:>5d} | FAILED: {e}")

            # Speedup: fused vs derf (same function, different kernel)
            fused = [r for r in all_results
                     if r["model_type"] == "fused"
                     and r["batch_size"] == batch_size
                     and r["seq_len"] == seq_len]
            derf = [r for r in all_results
                    if r["model_type"] == "derf"
                    and r["batch_size"] == batch_size
                    and r["seq_len"] == seq_len]
            if fused and derf:
                speedup = derf[0]["median_ms"] / fused[0]["median_ms"]
                print(f"  {'':>8s} | {'':>5s} | fused vs derf speedup: {speedup:.2f}x")

            print()

    # Summary table: median across batch sizes
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY (median tok/s across all batch sizes)")
        print(f"{'=' * 70}")
        for model_type in args.models:
            model_results = [r for r in all_results if r["model_type"] == model_type]
            if model_results:
                avg_tok = np.median([r["median_tok_per_sec"] for r in model_results])
                avg_ms = np.median([r["median_ms"] for r in model_results])
                print(f"  {model_type:>8s}: {avg_tok:>12,.0f} tok/s | {avg_ms:>7.1f}ms median")

    if args.wandb:
        # Log summary table
        import wandb
        table = wandb.Table(
            columns=["model", "batch_size", "seq_len", "median_ms", "tok_per_sec"],
        )
        for r in all_results:
            table.add_data(
                r["model_type"], r["batch_size"], r["seq_len"],
                round(r["median_ms"], 2), int(r["median_tok_per_sec"]),
            )
        wandb.log({"results": table})
        wandb.finish()


if __name__ == "__main__":
    main()
