#!/bin/bash

# Training script for GLU variants
# This script launches pretraining for all GLU variant models

set -e

echo "Starting GLU variants pretraining..."

# Base directory for the project
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Create output directory for models
mkdir -p checkpoints/glu_variants

# Array of GLU variants to train
variants=(
    "siglu"
    "tanhglu" 
    "capped_swiglu"
    "normglu"
    "additive_gate"
    "smooth_capped_swiglu"
)

# Function to train a single variant
train_variant() {
    local variant=$1
    echo "Training GLU variant: $variant"
    
    # Create variant-specific checkpoint directory
    mkdir -p "checkpoints/glu_variants/$variant"
    
    # Launch training with the appropriate config
    python pretrain.py \
        --config "configs/glu_variants/tinyllama_60m_${variant}.yaml" \
        --out_dir "checkpoints/glu_variants/$variant" \
        --devices 4 \
        --precision bf16-mixed \
        2>&1 | tee "logs/train_${variant}.log"
    
    echo "Completed training for $variant"
}

# Create logs directory
mkdir -p logs

# Train each variant
for variant in "${variants[@]}"; do
    echo "========================================="
    echo "Starting training for: $variant"
    echo "========================================="
    
    train_variant "$variant"
    
    echo "Finished training for: $variant"
    echo ""
done

echo "All GLU variant training completed!"
echo "Models saved in: checkpoints/glu_variants/"
echo "Logs saved in: logs/"