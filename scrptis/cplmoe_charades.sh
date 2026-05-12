#!/bin/bash
set -euo pipefail

# Determine repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate cpl

python "${REPO_ROOT}/cplmoe-main/train_moe.py" \
    --config-path "${REPO_ROOT}/cplmoe-main/config/charades/main_moe.json" \
    --resume "${REPO_ROOT}/cplmoe-main/checkpoints/charades_moe/model-best.pt" \
    --eval --vote
