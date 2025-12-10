#!/bin/bash
# Cloud Training Setup Script
# Run this on a fresh GPU instance (RunPod, Vast.ai, Lambda Labs)
#
# ════════════════════════════════════════════════════════════════════
# GPU RECOMMENDATIONS FOR PPO TRADING (CPU-BOUND WORKLOAD)
# ════════════════════════════════════════════════════════════════════
#
# | GPU          | $/hr  | vCPUs | RAM   | num_envs | Est. Steps/sec |
# |--------------|-------|-------|-------|----------|----------------|
# | RTX 3090     | $0.22 | 16    | 125GB | 12       | 200-400        |
# | RTX A5000    | $0.16 | 9     | 25GB  | 6        | 100-200        |
# | RTX 4090     | $0.34 | 6     | 41GB  | 4        | 80-150         |
# | L4           | $0.44 | 12    | 50GB  | 8        | 150-300        |
# | A40          | $0.35 | 9     | 50GB  | 6        | 120-240        |
#
# BEST VALUE: RTX 3090 @ $0.22/hr (16 vCPUs = most parallel envs)
#
# Your training is CPU-BOUND, not GPU-bound!
# - PPO network is small (~500MB VRAM)
# - Environment stepping is pure Python
# - More vCPUs = more parallel environments = faster training
# ════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "  AI Trading System - Cloud Training Setup"
echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────
# 1. System Check
# ─────────────────────────────────────────────────────────────────────
echo "[1/6] Checking system..."
nvidia-smi || { echo "WARNING: No GPU detected (will use CPU)"; }

# Get CPU count for optimal num_envs
VCPUS=$(nproc)
OPTIMAL_ENVS=$((VCPUS - 2))
if [ $OPTIMAL_ENVS -lt 1 ]; then OPTIMAL_ENVS=1; fi
if [ $OPTIMAL_ENVS -gt 16 ]; then OPTIMAL_ENVS=16; fi

echo "vCPUs: $VCPUS"
echo "Recommended num_envs: $OPTIMAL_ENVS"

# ─────────────────────────────────────────────────────────────────────
# 2. Install Python Dependencies
# ─────────────────────────────────────────────────────────────────────
echo "[2/6] Installing Python dependencies..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements (use cloud requirements if available)
if [ -f "cloud_training/requirements_cloud.txt" ]; then
    pip install -r cloud_training/requirements_cloud.txt
else
    pip install -r requirements.txt
fi

# ─────────────────────────────────────────────────────────────────────
# 3. Verify GPU is accessible
# ─────────────────────────────────────────────────────────────────────
echo "[3/6] Verifying PyTorch GPU access..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ─────────────────────────────────────────────────────────────────────
# 4. Prepare data directory
# ─────────────────────────────────────────────────────────────────────
echo "[4/6] Setting up directories..."
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs/tensorboard
mkdir -p models/best

# ─────────────────────────────────────────────────────────────────────
# 5. Check for training data
# ─────────────────────────────────────────────────────────────────────
echo "[5/6] Checking for training data..."
DATA_FILES=$(find data/processed -name "*.csv" 2>/dev/null | wc -l)
if [ "$DATA_FILES" -gt 0 ]; then
    echo "Found $DATA_FILES CSV data files!"
else
    echo "WARNING: No training data found in data/processed/"
    echo "Please upload your data files (EUR_USD_M15.csv, XAU_USD_M15.csv, etc.)"
fi

# ─────────────────────────────────────────────────────────────────────
# 6. Print training command
# ─────────────────────────────────────────────────────────────────────
echo "[6/6] Setup complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Ready to train! Commands:"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  # Activate environment"
echo "  source venv/bin/activate"
echo ""
echo "  # FAST training with optimal parallel envs (recommended)"
echo "  python train/train_ppo_hybrid.py \\"
echo "      --mode offline \\"
echo "      --preset training_fast \\"
echo "      --timesteps 5000000 \\"
echo "      --num_envs $OPTIMAL_ENVS \\"
echo "      --auto-pretrained \\"
echo "      --no-dashboard"
echo ""
echo "  # Monitor with TensorBoard"
echo "  tensorboard --logdir logs/tensorboard --port 6006 &"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  EXPECTED PERFORMANCE:"
echo "═══════════════════════════════════════════════════════════════════"
echo "  num_envs=$OPTIMAL_ENVS on $VCPUS vCPUs"
echo "  Expected: ~${OPTIMAL_ENVS}00-${OPTIMAL_ENVS}50 steps/sec"
echo "  5M steps: ~$((5000000 / (OPTIMAL_ENVS * 100) / 60)) minutes"
echo "═══════════════════════════════════════════════════════════════════"
