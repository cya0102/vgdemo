#!/bin/bash
set -euo pipefail

# Determine repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate pps

python "${REPO_ROOT}/pps-main/train.py" \
    --config-path "${REPO_ROOT}/pps-main/config/charades/config.json" \
    --ckpt-path "${REPO_ROOT}/pps-main/checkpoints/charades/model.pt" \
    --eval
