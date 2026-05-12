# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research repository for **Weakly Supervised Temporal Video Grounding (WSTVG)** — locating video segments matching a text query, trained with only video-level labels. It contains three related PyTorch projects, each in its own subdirectory:

| Directory | Paper | Description |
|-----------|-------|-------------|
| `cpl-main/` | CVPR 2022 / TPAMI 2025 | CPL: Contrastive Proposal Learning with Gaussian proposals |
| `cplmoe-main/` | — | CPL-MoE: Extends CPL with query-guided Mixture of Experts for proposal generation |
| `pps-main/` | AAAI 2024 | PPS: Gaussian Mixture Proposals with Pull-Push Learning |
| `scrptis/` | — | Evaluation shell scripts for all three models |

All three solve the same task on the same two benchmarks: **ActivityNet Captions** (C3D features) and **Charades-STA** (I3D features).

## Common architecture pattern

Every project follows the same structure:

- `train.py` — single entry point; parses CLI args, loads a JSON config, instantiates a Runner, calls `train()` or `eval()`
- `runner/` — contains a Runner class that builds datasets/model/optimizer, runs the train/eval loop, and reports metrics (R@1, R@5 at IoU thresholds [0.1, 0.3, 0.5, 0.7])
- `model/` — the core PyTorch `nn.Module` plus loss functions and sub-modules (transformer, attention, etc.)
- `dataset/` — `BaseDataset` + dataset-specific subclasses (ActivityNet, CharadesSTA) that load HDF5 video features and GloVe text embeddings
- `config/` — JSON files controlling all hyperparameters (model dims, loss weights, training schedule, dataset paths)
- `optimizers/` — Adam optimizer and LR schedulers adapted from fairseq

**Shared architecture across models:**
1. Video frame features (C3D/I3D) and sentence word embeddings (GloVe) are projected to a common hidden dim
2. A `DualTransformer` (two decoder stacks, no encoder) generates Gaussian proposal parameters (center + width per proposal) from video-text cross-attention
3. Negative proposals are mined (easy-to-hard schedule) for contrastive learning
4. A reconstruction loss (masked word prediction) and a contrastive loss (positive vs. negative proposals) are combined
5. At inference, proposals are ranked by NLL; the best one is selected (optionally with IoU-weighted voting)

Key differences:
- **CPL**: single FFN generates proposals; rec_loss + ivc_loss
- **CPL-MoE**: replaces the proposal FFN with a query-guided MoE; adds auxiliary load-balancing loss
- **PPS**: uses Gaussian mixture proposals instead of single Gaussians; adds pull-push losses for diversity

## Conda environments

- `cpl` — used by `cpl-main/` and `cplmoe-main/`
- `pps` — used by `pps-main/`

## Commands

All commands are run from the repo root (`/Users/chenyuan/Documents/develop/vgdemo/`).

### CPL

```bash
# Train
python cpl-main/train.py --config-path cpl-main/config/activitynet/main.json --log_dir LOG_DIR --tag TAG
python cpl-main/train.py --config-path cpl-main/config/charades/main.json --log_dir LOG_DIR --tag TAG

# Evaluate (loss-based)
python cpl-main/train.py --config-path cpl-main/config/activitynet/main.json --resume CHECKPOINT_FILE --eval

# Evaluate (vote-based — usually better)
python cpl-main/train.py --config-path cpl-main/config/activitynet/main.json --resume CHECKPOINT_FILE --eval --vote
```

### CPL-MoE

```bash
# Train (uses train_moe.py instead of train.py)
python cplmoe-main/train_moe.py --config-path cplmoe-main/config/activitynet/main_moe.json --tag TAG

# Evaluate
python cplmoe-main/train_moe.py --config-path cplmoe-main/config/activitynet/main_moe.json --resume CHECKPOINT_FILE --eval --vote
```

### PPS

```bash
# Train
bash pps-main/script/train_activitynet.sh
bash pps-main/script/train_charades.sh

# Evaluate
bash pps-main/script/eval_activitynet.sh          # paper model
bash pps-main/script/eval_activitynet_refact.sh   # refactored model
bash pps-main/script/eval_charades.sh
bash pps-main/script/eval_charades_refact.sh

# Or directly:
python pps-main/train.py --config-path pps-main/config/activitynet/config_refact.json --ckpt-path CKPT_FILE --eval
```

### scrptis/ evaluation scripts

These use absolute data paths (`/data/chenyuan/vgdemo/`) and assume a shared filesystem setup:

```bash
bash scrptis/cpl_activitynet.sh
bash scrptis/cplmoe_charades.sh
bash scrptis/pps_activitynet.sh
# etc.
```

### Single-video inference server

```bash
# Requires cpl conda environment
conda activate cpl
python inference_server.py   # serves at http://localhost:8000
```

FastAPI server with inline HTML UI. Upload a video file + enter a query text to get predicted time segments. Auto-detects dataset by filename (`v_` prefix → ActivityNet vote-based, otherwise → Charades loss-based). Uses ffprobe to read video duration. Model loads at startup (ActivityNet + Charades CPL models).

## Data

Feature files (C3D/I3D HDF5) and GloVe embeddings are not stored in this repo. They live at `/data/chenyuan/vgdemo/` (referenced by scrptis scripts and config files). Each project's `config/` JSON files contain dataset paths that may need adjustment for local use.

## Dependencies

- PyTorch (with CUDA)
- h5py
- nltk (requires `punkt` and `averaged_perceptron_tagger` downloads)
- fairseq (used for optimizer/scheduler base classes and softmax in attention)
- tqdm
- wandb (PPS only; disable via `use_wandb: false` in config)

No `requirements.txt` or `setup.py` exists in any project. fairseq may auto-install a conflicting PyTorch version — reinstall PyTorch after fairseq if needed.

## Important quirks

- Hyperparameter `lambda` in CPL configs is sensitive; if results don't reproduce, adjust it in small increments (e.g., 0.125 → 0.135).
- PPS has two model variants (paper/original and refactored) with slightly different hyperparameters and checkpoints due to environment changes during development.
- PPS uses `--ckpt-path` while CPL/CPL-MoE use `--resume` for checkpoint loading.
- CPL-MoE uses `train_moe.py` (not `train.py`); it has multiple runner versions (v2, v3, v4) for different ablation variants.
- The `--vote` flag enables vote-based inference (IoU-weighted voting across proposals), which typically outperforms loss-based selection.
