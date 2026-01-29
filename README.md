# Attention Benchmark

Benchmarking different attention mechanism implementations in JAX/Flax NNX.

## Overview

This project compares:

- **Naive custom multi-head attention** vs **Flax built-in `nnx.MultiHeadAttention`** at the encoder-layer level
- **Manual `scaled_dot_product_attention`** vs **`jax.nn.dot_product_attention`** (XLA flash-attention kernel) for raw Q/K/V computation
- An experimental **Derf** normalization layer (erf-based, replacing LayerNorm)
- **BERT pretraining benchmark**: Derf normalization vs standard LayerNorm on Wikipedia + BookCorpusOpen

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
| `--model` | `both` | Which model(s): `derf`, `normal`, or `both` |
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

On a Google Cloud TPU VM:

```bash
git clone <repo-url> && cd attention_benchmark
bash setup_tpu.sh
source .venv/bin/activate
python benchmarks/pretrain.py --model both --size base --steps 1000
```

The setup script installs `jax[tpu]`, project dependencies, and verifies TPU access.

## Project structure

```
src/
  models.py   -- All model definitions (encoders, embeddings, Derf, DerfBert, NormalBert)
  utils.py    -- BenchmarkTimer, input generation, logging helpers
  data.py     -- HuggingFace streaming data pipeline (tokenizer, MLM, dataloader)
benchmarks/
  benchmark_mha.py    -- Naive vs Flax MHA encoder comparison
  benchmark_flash.py  -- Manual vs XLA flash-attention comparison
  pretrain.py         -- BERT pretraining benchmark (Derf vs LayerNorm)
tests/
  test_correctness.py -- Shape and correctness tests
setup_tpu.sh          -- TPU VM setup script
```
