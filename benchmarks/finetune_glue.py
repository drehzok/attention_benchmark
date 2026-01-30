"""Fine-tune pretrained BERT on GLUE tasks (SST-2, MNLI).

Single model:
    python benchmarks/finetune_glue.py --task sst2 \
        --checkpoint checkpoints/derf_step300000/state \
        --model-type derf --size base

Batch comparison (all models from same pretrain run):
    python benchmarks/finetune_glue.py --task sst2 \
        --checkpoint-dir checkpoints/ --step 300000 --size base \
        --models derf normal fused
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    DerfBert, NormalBert, FusedDerfBert, BertForSequenceClassification,
)
from src.data import get_tokenizer, load_glue_task, create_glue_dataloader, GLUE_TASKS

CONFIGS = {
    "base": dict(num_layers=12, dim=768, num_heads=12, mlp_dim=3072),
    "small": dict(num_layers=4, dim=512, num_heads=8, mlp_dim=2048),
}

MODEL_CLASSES = {
    "derf": DerfBert,
    "normal": NormalBert,
    "fused": FusedDerfBert,
}


def create_bert(model_type, config, vocab_size, dropout_rate, rngs):
    cls = MODEL_CLASSES[model_type]
    return cls(
        vocab_size=vocab_size,
        num_layers=config["num_layers"],
        dim=config["dim"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout_rate=dropout_rate,
        rngs=rngs,
    )


def load_checkpoint(model, checkpoint_path):
    """Load pretrained weights from orbax checkpoint."""
    import orbax.checkpoint as ocp

    _, ref_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path, ref_state)
    nnx.update(model, restored)
    print(f"  Loaded checkpoint: {checkpoint_path}")


def count_params(model):
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def evaluate(model, input_ids, labels, batch_size):
    """Compute accuracy on a dataset."""
    n = len(labels)
    correct = 0
    total = 0

    for start in range(0, n - batch_size + 1, batch_size):
        batch_x = jnp.array(input_ids[start:start + batch_size])
        batch_y = labels[start:start + batch_size]

        logits = model(batch_x, deterministic=True)
        preds = jnp.argmax(logits, axis=-1)
        correct += int(jnp.sum(preds == batch_y))
        total += len(batch_y)

    return correct / total if total > 0 else 0.0


def finetune_one(model_type, args, config, tokenizer, task_info,
                 train_ids, train_labels, val_splits):
    """Fine-tune one model on one task. Returns dict of results."""
    print(f"\n{'=' * 60}")
    print(f"Fine-tuning {model_type.upper()} on {args.task.upper()}")
    print(f"{'=' * 60}")

    rngs = nnx.Rngs(args.seed)
    bert = create_bert(model_type, config, tokenizer.vocab_size, args.dropout, rngs)

    # Load pretrained checkpoint if provided
    ckpt_path = args.checkpoint
    if not ckpt_path and args.checkpoint_dir:
        ckpt_path = os.path.join(
            args.checkpoint_dir, f"{model_type}_step{args.step}", "state"
        )

    if ckpt_path and os.path.exists(ckpt_path):
        load_checkpoint(bert, ckpt_path)
    elif ckpt_path:
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        print(f"  Training from scratch.")
    else:
        print(f"  No checkpoint — training from scratch.")

    num_classes = task_info["num_classes"]
    model = BertForSequenceClassification(bert, num_classes, args.dropout, rngs)
    print(f"  Parameters: {count_params(model):,}")

    # Optimizer
    n_train = len(train_labels)
    steps_per_epoch = n_train // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.06)

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, args.lr, warmup_steps),
            optax.cosine_decay_schedule(args.lr, total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, input_ids, labels):
        def loss_fn(model):
            logits = model(input_ids, deterministic=False)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{model_type}-{args.task}",
            config={
                "model_type": model_type,
                "task": args.task,
                "size": args.size,
                **config,
                "num_classes": num_classes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
                "params": count_params(model),
            },
        )

    # Training loop
    start_time = time.perf_counter()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_x, batch_y in create_glue_dataloader(
            train_ids, train_labels, args.batch_size,
            shuffle=True, seed=args.seed + epoch,
        ):
            loss = train_step(model, optimizer, batch_x, batch_y)
            epoch_loss += float(loss)
            epoch_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)

        # Evaluate after each epoch
        results_str = []
        for val_name, (val_ids, val_labels) in val_splits.items():
            acc = evaluate(model, val_ids, val_labels, args.batch_size)
            results_str.append(f"{val_name}={acc:.4f}")

        elapsed = time.perf_counter() - start_time
        print(f"  Epoch {epoch + 1}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"{' | '.join(results_str)} | "
              f"{elapsed:.1f}s")

        if args.wandb:
            log_dict = {"train/loss": avg_loss}
            for val_name, (val_ids, val_labels) in val_splits.items():
                acc = evaluate(model, val_ids, val_labels, args.batch_size)
                log_dict[f"val/{val_name}"] = acc
            wandb.log(log_dict, step=epoch + 1)

    # Final evaluation
    final_results = {}
    for val_name, (val_ids, val_labels) in val_splits.items():
        acc = evaluate(model, val_ids, val_labels, args.batch_size)
        final_results[val_name] = acc

    if args.wandb:
        wandb.log({f"final/{k}": v for k, v in final_results.items()})
        wandb.finish()

    return {
        "model_type": model_type,
        "task": args.task,
        **final_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained BERT on GLUE (SST-2, MNLI)"
    )
    parser.add_argument("--task", choices=list(GLUE_TASKS.keys()), required=True,
                        help="GLUE task: sst2, mnli")
    parser.add_argument("--size", choices=["base", "small"], default="base",
                        help="Model size")

    # Checkpoint loading (two modes)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to single checkpoint (orbax state dir)")
    parser.add_argument("--checkpoint-dir", type=str, default="",
                        help="Base checkpoint dir (used with --step and --models)")
    parser.add_argument("--step", type=int, default=0,
                        help="Pretrain step for checkpoint-dir mode")

    # Model selection
    parser.add_argument("--model-type", type=str, default="",
                        help="Single model type: derf, normal, fused")
    parser.add_argument("--models", nargs="+", default=[],
                        help="Multiple model types for batch comparison")

    # Fine-tuning hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="derf-glue",
                        help="W&B project name (default: derf-glue)")
    args = parser.parse_args()

    # Determine which models to run
    model_types = args.models if args.models else (
        [args.model_type] if args.model_type else ["derf"]
    )

    task_info = GLUE_TASKS[args.task]

    print(f"GLUE Fine-tuning: {args.task.upper()}")
    print(f"  Backend:  {jax.default_backend()}")
    print(f"  Devices:  {jax.devices()}")
    print(f"  Size:     {args.size}")
    print(f"  Models:   {model_types}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  LR:       {args.lr}")

    config = CONFIGS[args.size]
    tokenizer = get_tokenizer()

    # Load GLUE data
    print(f"\nLoading {args.task} data...")
    train_ids, train_labels = load_glue_task(
        args.task, tokenizer, args.seq_len, split="train"
    )
    print(f"  Train: {len(train_labels)} examples")

    val_splits = {}
    if args.task == "mnli":
        for split_name in ["validation_matched", "validation_mismatched"]:
            ids, labels = load_glue_task(
                args.task, tokenizer, args.seq_len, split=split_name
            )
            val_splits[split_name] = (ids, labels)
            print(f"  {split_name}: {len(labels)} examples")
    else:
        ids, labels = load_glue_task(
            args.task, tokenizer, args.seq_len, split="validation"
        )
        val_splits["validation"] = (ids, labels)
        print(f"  Validation: {len(labels)} examples")

    # Fine-tune each model
    all_results = []
    for mt in model_types:
        result = finetune_one(
            mt, args, config, tokenizer, task_info,
            train_ids, train_labels, val_splits,
        )
        all_results.append(result)

    # Summary
    if len(all_results) >= 1:
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {args.task.upper()}")
        print(f"{'=' * 60}")
        for r in all_results:
            metrics = " | ".join(
                f"{k}={v:.4f}" for k, v in r.items()
                if k not in ("model_type", "task")
            )
            print(f"  {r['model_type']:>8s}: {metrics}")


if __name__ == "__main__":
    main()
