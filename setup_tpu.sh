#!/bin/bash
# Setup script for Google Cloud TPU VM
# Usage: bash setup_tpu.sh

set -e

# Install system deps
sudo apt-get update && sudo apt-get install -y python3-venv git

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install JAX for TPU
pip install -U pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install project deps
pip install -e ".[train]"

# Verify TPU
python3 -c "import jax; print(f'Devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"

echo ""
echo "Setup complete. Run benchmarks with:"
echo "  source .venv/bin/activate"
echo "  python benchmarks/pretrain.py --model both --size base"
