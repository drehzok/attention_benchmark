# --- 1. Install Python 3.11 (Required for v5e Pallas) ---
sudo apt-get update -y
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt-get install python3.11-full python3.11-dev python3.11-venv -y

# --- 2. Create Virtual Env ---
python3.11 -m venv ~/jax-tpu
echo "source ~/jax-tpu/bin/activate" >> ~/.bashrc
source ~/jax-tpu/bin/activate

# --- 3. Install JAX + Pallas ---
pip install --upgrade pip
# Installs JAX with the correct libtpu for v5e
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# --- 4. Install Benchmarking Tools ---
pip install transformers flax optax datasets
