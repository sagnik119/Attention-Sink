#!/bin/bash

# Individual training scripts for GLU variants
# This creates separate scripts for each variant that can be run independently

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Create individual training scripts
mkdir -p scripts/individual

# Function to create individual script
create_individual_script() {
    local variant=$1
    local script_path="scripts/individual/train_${variant}.sh"
    
    cat > "$script_path" << EOF
#!/bin/bash

# Training script for ${variant} GLU variant
set -e

echo "Starting ${variant} GLU variant pretraining..."

# Base directory for the project
BASE_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)"
cd "\$BASE_DIR"

# Create output directory for this variant
mkdir -p "checkpoints/glu_variants/${variant}"
mkdir -p logs

echo "Training ${variant} variant..."
echo "Config: configs/glu_variants/tinyllama_60m_${variant}.yaml"
echo "Output: checkpoints/glu_variants/${variant}"
echo "Log: logs/train_${variant}.log"

# Launch training
python pretrain.py \\
    --config "configs/glu_variants/tinyllama_60m_${variant}.yaml" \\
    --out_dir "checkpoints/glu_variants/${variant}" \\
    --devices 4 \\
    --precision bf16-mixed \\
    2>&1 | tee "logs/train_${variant}.log"

echo "Completed training for ${variant}"
echo "Model saved in: checkpoints/glu_variants/${variant}"
echo "Log saved in: logs/train_${variant}.log"
EOF

    chmod +x "$script_path"
    echo "Created: $script_path"
}

# Array of GLU variants
variants=(
    "siglu"
    "tanhglu" 
    "capped_swiglu"
    "normglu"
    "additive_gate"
    "smooth_capped_swiglu"
)

echo "Creating individual training scripts..."

for variant in "${variants[@]}"; do
    create_individual_script "$variant"
done

echo ""
echo "Individual training scripts created in: scripts/individual/"
echo ""
echo "To train a specific variant, run:"
echo "  bash scripts/individual/train_<variant>.sh"
echo ""
echo "Available variants:"
for variant in "${variants[@]}"; do
    echo "  - $variant"
done