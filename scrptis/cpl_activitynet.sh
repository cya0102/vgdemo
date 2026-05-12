#!/bin/bash
set -euo pipefail

# Determine repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate cpl

python "${REPO_ROOT}/cpl-main/train.py" \
    --config-path "${REPO_ROOT}/cpl-main/config/activitynet/main.json" \
    --resume "${REPO_ROOT}/cpl-main/checkpoints/activitynet/model-best.pt" \
    --eval --vote
