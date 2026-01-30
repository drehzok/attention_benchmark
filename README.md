# Attention Benchmark

Benchmarking different attention mechanism implementations in JAX/Flax NNX.

## Overview

This project compares:

- **Naive custom multi-head attention** vs **Flax built-in `nnx.MultiHeadAttention`** at the encoder-layer level
- **Manual `scaled_dot_product_attention`** vs **`jax.nn.dot_product_attention`** (XLA flash-attention kernel) for raw Q/K/V computation
- An experimental **Derf** normalization layer (erf-based, replacing LayerNorm)
- A **fused Derf+Linear Pallas kernel** for TPU (keeps Derf intermediate in VMEM, avoids HBM roundtrip)
- **BERT pretraining benchmark**: Derf vs LayerNorm vs Fused Derf on Wikipedia + BookCorpusOpen
- **GLUE fine-tuning**: SST-2 and MNLI evaluation for all model variants

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

For pretraining (includes HuggingFace transformers/datasets):

```bash
pip install -e ".[train]"
```

## Usage

### Run benchmarks

```bash
python benchmarks/benchmark_mha.py
python benchmarks/benchmark_flash.py
```

### Run tests

```bash
pytest tests/
```

### BERT Pretraining Benchmark

Train BERT models with Derf vs LayerNorm normalization on Wikipedia + BookCorpusOpen:

```bash
# Train both models (BERT-Base) for comparison
python benchmarks/pretrain.py --model both --size base

# Train only Derf variant (BERT-Small)
python benchmarks/pretrain.py --model derf --size small

# Train only standard LayerNorm variant
python benchmarks/pretrain.py --model normal --size base

# Quick smoke test on CPU
python benchmarks/pretrain.py --model both --size small --steps 5 --batch-size 2 --seq-len 32
```

#### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `both` | Which model(s): `derf`, `normal`, `fused`, `both`, `all` |
| `--size` | `base` | Model size: `base` (~110M) or `small` (~30M) |
| `--steps` | `10000` | Number of training steps |
| `--batch-size` | `64` | Batch size |
| `--lr` | `1e-4` | Peak learning rate |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup-steps` | `1000` | Linear warmup steps |
| `--seq-len` | `128` | Sequence length |
| `--mlm-prob` | `0.15` | MLM masking probability |
| `--dropout` | `0.1` | Dropout rate |
| `--log-interval` | `100` | Log every N steps |
| `--checkpoint-dir` | *(none)* | Directory for checkpoints |
| `--checkpoint-interval` | `1000` | Checkpoint every N steps |
| `--seed` | `42` | Random seed |
| `--no-shard` | off | Disable SPMD (single device) |
| `--resume` | off | Resume from latest checkpoint |
| `--wandb` | off | Enable W&B logging |
| `--wandb-project` | `derf-pretrain` | W&B project name |

#### Data

Uses the original BERT paper data sources via HuggingFace datasets (streaming):
- **Wikipedia** (`wikipedia`, `20220301.en`)
- **BookCorpusOpen** (`bookcorpusopen`)

Documents are tokenized, concatenated, and chunked into fixed-length sequences. MLM masking (80% [MASK], 10% random, 10% unchanged) is applied on-the-fly.

#### Model Configurations

| Size | Layers | Dim | Heads | MLP Dim | ~Params |
|---|---|---|---|---|---|
| `base` | 12 | 768 | 12 | 3072 | 110M |
| `small` | 4 | 512 | 8 | 2048 | 30M |

### TPU Setup

#### 1. Create and connect to TPU VM

```bash
gcloud compute tpus tpu-vm create my-bert-bench \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base

gcloud compute tpus tpu-vm ssh my-bert-bench --zone=us-central2-b
```

#### 2. Install dependencies

```bash
git clone <repo-url> && cd attention_benchmark
bash setup_tpu.sh
source .venv/bin/activate
```

The setup script installs `jax[tpu]`, project dependencies, and verifies TPU access.

#### 3. Run tests

```bash
# Unit tests
pytest tests/test_correctness.py -v

# Smoke test (few steps, tiny config)
python benchmarks/pretrain.py --model both --size small --steps 5 --batch-size 2 --seq-len 32 --log-interval 1
```

#### 4. Run benchmarks

```bash
# BERT-Small quick comparison (~30M params)
python benchmarks/pretrain.py --model both --size small --steps 1000 --batch-size 64

# BERT-Base full comparison (~110M params)
python benchmarks/pretrain.py --model both --size base --steps 10000 --batch-size 64

# With checkpointing
python benchmarks/pretrain.py --model both --size base --steps 10000 --batch-size 64 \
  --checkpoint-dir checkpoints/ --checkpoint-interval 2000
```

When using `--model both`, both models train sequentially with identical data order and a comparison table prints at the end.

### GLUE Fine-tuning

Fine-tune pretrained checkpoints on SST-2 or MNLI:

```bash
# Single model
python benchmarks/finetune_glue.py --task sst2 \
    --checkpoint checkpoints/derf_step50000/state \
    --model-type derf --size base

# All models from same pretrain run
python benchmarks/finetune_glue.py --task sst2 \
    --checkpoint-dir checkpoints/ --step 50000 --size base \
    --models derf normal fused --wandb
```

### Running on Spot/Preemptible TPUs

Spot TPUs can be preempted at any time. Use this setup to survive preemptions:

#### 1. Start training in tmux

```bash
gcloud compute tpus tpu-vm ssh YOUR_TPU --zone=YOUR_ZONE
tmux new -s train

# Run pretraining with checkpoints + wandb
python benchmarks/pretrain.py --model all --size base \
    --steps 50000 --batch-size 64 --seq-len 256 \
    --log-interval 500 --wandb \
    --checkpoint-dir /home/$USER/checkpoints/ \
    --checkpoint-interval 10000
```

#### 2. Back up checkpoints to GCS (second tmux window)

```bash
# Ctrl+B, C to open new tmux window
watch -n 300 gsutil -m rsync -r /home/$USER/checkpoints/ gs://YOUR_BUCKET/checkpoints/
```

Detach tmux with `Ctrl+B, D`. Reattach later with `tmux attach -t train`.

#### 3. After preemption — resume

```bash
# Re-create spot TPU, SSH in, then:

# Pull checkpoints from GCS
gsutil -m rsync -r gs://YOUR_BUCKET/checkpoints/ /home/$USER/checkpoints/

# Resume (automatically finds latest checkpoint per model, restores optimizer state)
tmux new -s train
python benchmarks/pretrain.py --model all --size base \
    --steps 50000 --batch-size 64 --seq-len 256 \
    --log-interval 500 --wandb --resume \
    --checkpoint-dir /home/$USER/checkpoints/ \
    --checkpoint-interval 10000
```

`--resume` restores full optimizer state (weights + Adam momentum + LR schedule position). Models that already reached `--steps` are skipped automatically.

#### 4. After pretraining — fine-tune on GLUE

```bash
python benchmarks/finetune_glue.py --task sst2 \
    --checkpoint-dir /home/$USER/checkpoints/ --step 50000 --size base \
    --models derf normal fused --seq-len 256 --wandb \
&& python benchmarks/finetune_glue.py --task mnli \
    --checkpoint-dir /home/$USER/checkpoints/ --step 50000 --size base \
    --models derf normal fused --seq-len 256 --wandb
```

## Project structure

```
src/
  models.py   -- Model definitions (DerfBert, NormalBert, FusedDerfBert, classification wrapper)
  kernels.py  -- Pallas TPU kernels (fused Derf+Linear, matmul)
  utils.py    -- BenchmarkTimer, input generation, logging helpers
  data.py     -- Data pipeline (streaming MLM, GLUE task loading)
benchmarks/
  benchmark_mha.py    -- Naive vs Flax MHA encoder comparison
  benchmark_flash.py  -- Manual vs XLA flash-attention comparison
  pretrain.py         -- BERT pretraining (Derf vs LayerNorm vs Fused, with resume)
  finetune_glue.py    -- GLUE fine-tuning + evaluation (SST-2, MNLI)
tests/
  test_correctness.py      -- Shape and correctness tests
  test_pretrain_smoke.py   -- Smoke test (all 3 models, SPMD sharding, training step)
  test_fused_derf_linear.py -- Fused kernel correctness + backward tests
  test_pallas_matmul.py    -- Pallas matmul kernel tests
setup_tpu.sh               -- TPU VM setup script
```
