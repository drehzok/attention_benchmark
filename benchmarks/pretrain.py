"""BERT pretraining benchmark: Derf vs LayerNorm.

Usage:
    python benchmarks/pretrain.py --model derf --size base
    python benchmarks/pretrain.py --model normal --size small
    python benchmarks/pretrain.py --model unfused --size base
    python benchmarks/pretrain.py --model both --size base

    # Use --no-shard for the fused model on multi-device TPU
    # (Pallas/Mosaic kernels can't be auto-partitioned by GSPMD)
    python benchmarks/pretrain.py --model fused --size base --no-shard
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DerfBert, NormalBert, FusedDerfBert, UnfusedQKVBert
from src.data import get_tokenizer, create_dataset, create_dataloader

CONFIGS = {
    "base": dict(num_layers=12, dim=768, num_heads=12, mlp_dim=3072),   # ~110M
    "small": dict(num_layers=4, dim=512, num_heads=8, mlp_dim=2048),    # ~30M
}


def create_model(model_type, config, vocab_size, dropout_rate, rngs, mesh=None):
    """Instantiate DerfBert, NormalBert, or FusedDerfBert based on model_type string."""
    cls = {"derf": DerfBert, "normal": NormalBert, "fused": FusedDerfBert, "unfused": UnfusedQKVBert}[model_type]
    kwargs = dict(
        vocab_size=vocab_size,
        num_layers=config["num_layers"],
        dim=config["dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=dropout_rate,
        rngs=rngs,
    )
    if model_type == "fused":
        kwargs["mesh"] = mesh
    return cls(**kwargs)


def count_params(model):
    """Count total number of parameters in an NNX model."""
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def create_optimizer(lr, warmup_steps, total_steps, weight_decay):
    """AdamW with linear warmup + cosine decay."""
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    return optax.adamw(learning_rate=schedule, weight_decay=weight_decay)


def make_train_step(model, optimizer):
    """Create a JIT-compiled training step function."""

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels, rngs):
        def loss_fn(model):
            logits = model(input_ids, deterministic=False, rngs=rngs)
            # Cross-entropy loss only on masked positions
            vocab_size = logits.shape[-1]
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            # Flatten for gathering
            flat_log_probs = log_probs.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)

            # Gather log probs for correct labels (clamp -100 to 0 for indexing)
            safe_labels = jnp.maximum(flat_labels, 0)
            token_losses = -flat_log_probs[jnp.arange(flat_log_probs.shape[0]), safe_labels]

            # Mask out non-masked positions (labels == -100)
            mask = flat_labels != -100
            loss = jnp.sum(token_losses * mask) / jnp.maximum(jnp.sum(mask), 1)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    return train_step


def train_model(model_type, args, config, tokenizer, mesh, print_prefix=""):
    """Train a single model and return results."""
    print(f"\n{'=' * 60}")
    print(f"{print_prefix}Training {model_type.upper()} model ({args.size})")
    print(f"{'=' * 60}")

    replicate = NamedSharding(mesh, P())
    data_shard = NamedSharding(mesh, P('dp'))

    rngs = nnx.Rngs(params=args.seed, dropout=args.seed+1)
    model = create_model(model_type, config, tokenizer.vocab_size, args.dropout, rngs, mesh=mesh)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    # Replicate model state across all devices
    state = nnx.state(model)
    state = jax.device_put(state, replicate)
    nnx.update(model, state)

    tx = create_optimizer(args.lr, args.warmup_steps, args.steps, args.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Replicate optimizer state across all devices
    opt_state = nnx.state(optimizer)
    opt_state = jax.device_put(opt_state, replicate)
    nnx.update(optimizer, opt_state)

    train_step = make_train_step(model, optimizer)

    # Resume from checkpoint if requested
    resume_step = 0
    if args.resume and args.checkpoint_dir:
        resume_step = load_resume_checkpoint(model, optimizer, args.checkpoint_dir, model_type)
        if resume_step > 0:
            # Re-replicate restored state across devices
            opt_state = nnx.state(optimizer)
            opt_state = jax.device_put(opt_state, replicate)
            nnx.update(optimizer, opt_state)

    if resume_step >= args.steps:
        print(f"  Already completed {resume_step} steps (target: {args.steps}). Skipping.")
        return {
            "model_type": model_type,
            "params": n_params,
            "total_time": 0,
            "avg_tokens_per_sec": 0,
            "final_loss": None,
            "losses": [],
        }

    dataset = create_dataset(tokenizer, seq_len=args.seq_len, seed=args.seed)
    dataloader = create_dataloader(
        dataset, args.batch_size, tokenizer, mlm_prob=args.mlm_prob, seed=args.seed
    )

    n_mesh_devices = len(mesh.devices.flat)

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{model_type}-{args.size}",
            config={
                "model_type": model_type,
                "size": args.size,
                **config,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "steps": args.steps,
                "seq_len": args.seq_len,
                "mlm_prob": args.mlm_prob,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "num_devices": n_mesh_devices,
                "params": n_params,
                "resume_step": resume_step,
            },
            resume="allow" if resume_step > 0 else None,
        )

    remaining = args.steps - resume_step
    print(f"Starting training for {remaining} steps "
          f"(step {resume_step + 1} -> {args.steps}, "
          f"{'single device' if n_mesh_devices == 1 else f'data-parallel across {n_mesh_devices} devices'})...")
    total_tokens = 0
    start_time = time.perf_counter()
    step_times = []
    losses = []

    for step, (input_ids, labels) in enumerate(dataloader):
        if step >= args.steps:
            break

        # Skip already-trained steps (fast — just iterates the dataloader)
        if step < resume_step:
            continue

        # Shard batch across devices along batch dimension
        input_ids = jax.device_put(input_ids, data_shard)
        labels = jax.device_put(labels, data_shard)

        step_start = time.perf_counter()
        loss = train_step(model, optimizer, input_ids, labels, rngs)
        jax.block_until_ready(loss)
        step_elapsed = time.perf_counter() - step_start

        step_times.append(step_elapsed)
        loss_val = float(loss)
        losses.append(loss_val)
        total_tokens += args.batch_size * args.seq_len

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = total_tokens / elapsed
            steps_per_sec = (step + 1) / elapsed
            print(
                f"  step {step + 1:>6d}/{args.steps} | "
                f"loss {loss_val:.4f} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"{tokens_per_sec:.0f} tok/s"
            )

            if args.wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "perf/steps_per_sec": steps_per_sec,
                    "perf/tokens_per_sec": tokens_per_sec,
                    "perf/step_time_sec": step_elapsed,
                }, step=step + 1)

        if args.checkpoint_dir and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step + 1, model_type, args)

    total_time = time.perf_counter() - start_time
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    print(f"\n--- {model_type.upper()} Summary ---")
    print(f"  Total steps:     {min(step + 1, args.steps)}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Avg tok/s:       {avg_tokens_per_sec:.0f}")
    print(f"  Final loss:      {losses[-1]:.4f}" if losses else "  No steps completed")

    if args.wandb:
        wandb.finish()

    return {
        "model_type": model_type,
        "params": n_params,
        "total_time": total_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }


def save_checkpoint(model, optimizer, step, model_type, args):
    """Save model state + optimizer state using orbax."""
    try:
        import orbax.checkpoint as ocp

        path = os.path.abspath(os.path.join(args.checkpoint_dir, f"{model_type}_step{step}"))
        os.makedirs(path, exist_ok=True)

        checkpointer = ocp.StandardCheckpointer()

        # Model state (used by finetune_glue.py)
        _, model_state = nnx.split(model)
        checkpointer.save(os.path.join(path, "state"), model_state)

        # Full optimizer state (model params + Adam momentum/variance + LR step counter)
        _, opt_state = nnx.split(optimizer)
        checkpointer.save(os.path.join(path, "optimizer"), opt_state)

        print(f"  Checkpoint saved: {path}")
    except ImportError:
        print("  orbax-checkpoint not installed, skipping checkpoint save")


def _checkpoint_is_valid(checkpoint_path):
    """Check that a checkpoint dir has finalized data (not just .orbax-checkpoint-tmp)."""
    if not os.path.isdir(checkpoint_path):
        return False
    # Valid if it has a finalized 'optimizer' or 'state' subdirectory
    return (os.path.isdir(os.path.join(checkpoint_path, "optimizer"))
            or os.path.isdir(os.path.join(checkpoint_path, "state")))


def find_latest_checkpoint(checkpoint_dir, model_type):
    """Find the latest valid checkpoint step for a model type. Returns step or None."""
    import re
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        return None
    pattern = re.compile(rf"^{re.escape(model_type)}_step(\d+)$")
    steps = []
    for name in os.listdir(checkpoint_dir):
        m = pattern.match(name)
        if m:
            steps.append(int(m.group(1)))
    # Check from newest to oldest, return first valid one
    for step in sorted(steps, reverse=True):
        path = os.path.join(checkpoint_dir, f"{model_type}_step{step}")
        if _checkpoint_is_valid(path):
            return step
        else:
            print(f"  Skipping incomplete checkpoint: {path}")
    return None


def load_resume_checkpoint(model, optimizer, checkpoint_dir, model_type):
    """Load optimizer state (includes model weights) from latest checkpoint.

    Returns the resume step, or 0 if no checkpoint found.
    """
    step = find_latest_checkpoint(checkpoint_dir, model_type)
    if step is None:
        return 0

    import orbax.checkpoint as ocp

    path = os.path.abspath(os.path.join(checkpoint_dir, f"{model_type}_step{step}"))
    checkpointer = ocp.StandardCheckpointer()

    opt_path = os.path.join(path, "optimizer")
    if os.path.exists(opt_path):
        _, ref = nnx.split(optimizer)
        restored = checkpointer.restore(opt_path, ref)
        nnx.update(optimizer, restored)
        print(f"  Resumed from {path} (step {step}, full optimizer state)")
    else:
        # Fallback: old checkpoint without optimizer state
        state_path = os.path.join(path, "state")
        _, ref = nnx.split(model)
        restored = checkpointer.restore(state_path, ref)
        nnx.update(model, restored)
        print(f"  Resumed from {path} (step {step}, model weights only — optimizer reset)")

    return step


def main():
    parser = argparse.ArgumentParser(description="BERT pretraining benchmark: Derf vs LayerNorm")
    parser.add_argument("--model", choices=["derf", "normal", "fused", "unfused", "both", "all"], default="both",
                        help="Which model(s) to train: derf, normal, fused, unfused, both (derf+normal), all")
    parser.add_argument("--size", choices=["base", "small"], default="base",
                        help="Model size (default: base)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps (default: 1000)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (default: 128)")
    parser.add_argument("--mlm-prob", type=float, default=0.15,
                        help="MLM masking probability (default: 0.15)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N steps (default: 100)")
    parser.add_argument("--checkpoint-dir", type=str, default="",
                        help="Checkpoint directory (default: no checkpointing)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Checkpoint every N steps (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-shard", action="store_true",
                        help="Disable SPMD data parallelism (single device). "
                             "Required for --model fused on multi-device TPU "
                             "because Pallas/Mosaic kernels can't be auto-partitioned.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --checkpoint-dir")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="derf-pretrain",
                        help="W&B project name (default: derf-pretrain)")
    parser.add_argument("--wandb-name", type=str, default="",
                        help="W&B run name (default: auto-generated)")
    args = parser.parse_args()

    # Multi-host TPU initialization (skip on single-host Colab/Kaggle)
    try:
        jax.distributed.initialize()
    except Exception:
        pass  # single-host: no coordinator needed

    devices = jax.devices()
    num_devices = len(devices)
    import numpy as np

    if args.no_shard:
        mesh_devices = np.array([devices[0]])
        mesh = Mesh(mesh_devices, axis_names=('dp',))
        num_mesh_devices = 1
    else:
        mesh_devices = np.array(devices)
        mesh = Mesh(mesh_devices, axis_names=('dp',))
        num_mesh_devices = num_devices

    if args.batch_size % num_mesh_devices != 0:
        old_bs = args.batch_size
        args.batch_size = (args.batch_size // num_mesh_devices) * num_mesh_devices
        if args.batch_size == 0:
            args.batch_size = num_mesh_devices
        print(f"  Adjusted batch size {old_bs} -> {args.batch_size} (divisible by {num_mesh_devices} mesh devices)")

    print("BERT Pretraining Benchmark: Derf vs LayerNorm")
    print(f"  JAX devices:   {devices}")
    print(f"  Device count:  {num_devices}")
    if args.no_shard:
        print(f"  Sharding:      DISABLED (single device: {devices[0]})")
    else:
        print(f"  Sharding:      data-parallel across {num_mesh_devices} devices")
    print(f"  Model size:    {args.size}")
    print(f"  Config:        {CONFIGS[args.size]}")
    print(f"  Steps:         {args.steps}")
    print(f"  Batch size:    {args.batch_size} ({args.batch_size // num_mesh_devices} per device)")
    print(f"  Seq length:    {args.seq_len}")
    print(f"  Learning rate: {args.lr}")

    config = CONFIGS[args.size]

    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    model_types = {
        "derf": ["derf"],
        "normal": ["normal"],
        "fused": ["fused"],
        "unfused": ["unfused"],
        "both": ["derf", "normal"],
        "all": ["derf", "normal", "fused", "unfused"],
    }[args.model]

    results = []
    for mt in model_types:
        results.append(train_model(mt, args, config, tokenizer, mesh))

    if len(results) >= 2:
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        for r in results:
            print(f"  {r['model_type'].upper():>8s}: "
                  f"{r['params']:>12,d} params | "
                  f"{r['avg_tokens_per_sec']:>10,.0f} tok/s | "
                  f"final loss {r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
