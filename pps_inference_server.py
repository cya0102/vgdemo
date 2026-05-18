"""
PPS Single Video Inference Server (FastAPI)
Runs on port 8200 in the 'pps' conda environment.
"""
import hashlib
import os
import sys
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Optional

import h5py
import nltk
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
PPS_ROOT = REPO_ROOT / "pps-main"
sys.path.insert(0, str(PPS_ROOT))

from model.pps import PPS
from model.loss import cal_nll_loss

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ──────────────────────────────────────────────────────────────
MAX_NUM_FRAMES = 200
MAX_NUM_WORDS = 20

# ── Helpers ────────────────────────────────────────────────────────────────

def get_video_duration(filepath: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", filepath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def build_vocab_mapping(vocab: dict, vocab_size: int) -> dict:
    keep_vocab = {}
    for w, _ in vocab["counter"].most_common(vocab_size):
        keep_vocab[w] = len(keep_vocab) + 1
    return keep_vocab


def process_query(query: str, vocab: dict, keep_vocab: dict):
    weights = []
    words = []
    for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(query)):
        word = word.lower()
        if word not in keep_vocab:
            continue
        if "NN" in tag:
            weights.append(2)
        elif "VB" in tag:
            weights.append(2)
        elif "JJ" in tag or "RB" in tag:
            weights.append(2)
        else:
            weights.append(1)
        words.append(word)

    words = words[:MAX_NUM_WORDS]
    weights = weights[:MAX_NUM_WORDS]

    if len(words) == 0:
        raise ValueError("No known words in query (all OOV)")

    words_id = [keep_vocab[w] for w in words]
    words_feat = [
        vocab["id2vec"][vocab["w2id"][words[0]]].astype(np.float32)
    ]
    words_feat.extend(
        vocab["id2vec"][vocab["w2id"][w]].astype(np.float32) for w in words
    )
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = np.exp(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    return words_id, np.array(words_feat, dtype=np.float32), weights_arr


def load_and_sample_features(hdf5_path: str, video_id: str, feature_key: Optional[str]):
    with h5py.File(hdf5_path, "r") as fr:
        if feature_key:
            frames_feat = np.asarray(fr[video_id][feature_key]).astype(np.float32)
        else:
            frames_feat = np.asarray(fr[video_id]).astype(np.float32)

    num_frames = len(frames_feat)
    keep_idx = np.arange(0, MAX_NUM_FRAMES + 1) / MAX_NUM_FRAMES * num_frames
    keep_idx = np.round(keep_idx).astype(np.int64)
    keep_idx[keep_idx >= num_frames] = num_frames - 1

    sampled = []
    for j in range(MAX_NUM_FRAMES):
        s, e = keep_idx[j], keep_idx[j + 1]
        if s == e:
            sampled.append(frames_feat[s])
        else:
            sampled.append(frames_feat[s:e].mean(axis=0))
    return np.stack(sampled, 0)


def calculate_IoU(i0, i1):
    """PPS-style IoU: takes tuple of arrays (start_array, end_array)."""
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0),
             np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0),
             np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def select_best_proposal_pps(output: dict, pos_mask_list: list, dataset_name: str):
    """PPS-specific proposal selection. Returns (start, end) in [0,1]."""
    bsz = 1

    # Compute NLL for each mask type
    nll_losses = []
    for mask in pos_mask_list:
        words_logits = output['words_logits'][mask]
        num_props_i = words_logits.size(0) // bsz

        words_mask = (output['words_mask'].unsqueeze(1)
                       .expand(bsz, num_props_i, -1)
                       .contiguous()
                       .view(bsz * num_props_i, -1))
        words_id = (output['words_id'].unsqueeze(1)
                     .expand(bsz, num_props_i, -1)
                     .contiguous()
                     .view(bsz * num_props_i, -1))
        nll_loss, _ = cal_nll_loss(words_logits, words_id, words_mask)
        nll_losses.append(nll_loss.view(bsz, num_props_i))

    nll_losses_cat = torch.cat(nll_losses, 1)  # (1, total_props)
    nll_losses_sort, nll_loss_idx = nll_losses_cat.sort(dim=-1)
    nll_losses_sort_np = nll_losses_sort.cpu().numpy()

    # Gather boundaries in sorted order
    left = torch.cat(list(output['prop_lefts'].values()), 1).gather(index=nll_loss_idx, dim=-1)
    right = torch.cat(list(output['prop_rights'].values()), 1).gather(index=nll_loss_idx, dim=-1)

    selected_props = torch.stack([left, right], dim=-1).cpu().numpy()  # (1, total_props, 2)
    num_all_props = selected_props.shape[1]

    # Vote-based selection
    if dataset_name == 'ActivityNet':
        c = np.ones((bsz, num_all_props))
    else:  # CharadesSTA — NLL-weighted voting
        c = 1 - nll_losses_sort_np / nll_losses_sort_np.max(axis=1, keepdims=True)

    votes = np.zeros((bsz, num_all_props))
    for i in range(num_all_props):
        for j in range(num_all_props):
            iou = calculate_IoU(
                (selected_props[:, i, 0], selected_props[:, i, 1]),
                (selected_props[:, j, 0], selected_props[:, j, 1]),
            )
            votes[:, i] += iou * c[:, j]

    best_idx = int(np.argmax(votes, axis=1)[0])
    return float(selected_props[0, best_idx, 0]), float(selected_props[0, best_idx, 1])


# ── Model loader ───────────────────────────────────────────────────────────

def load_pps_model(config_path: str):
    """Load PPS model, vocab, and keep_vocab mapping."""
    with open(config_path) as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    # Resolve paths
    vocab_path = dataset_cfg["vocab_path"]
    if not os.path.isabs(vocab_path):
        vocab_path = str(PPS_ROOT / vocab_path)

    feature_path = dataset_cfg["feature_path"]
    if not os.path.isabs(feature_path):
        feature_path = str(PPS_ROOT / feature_path)

    checkpoint_dir = train_cfg["save_path"]
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = str(PPS_ROOT / checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model-best.pt")
    if not os.path.exists(checkpoint_path):
        # Try alternate path
        checkpoint_path = str(PPS_ROOT / "checkpoints" / dataset_cfg["name"].lower() / "model-best.pt")

    dataset_name = dataset_cfg.get("name", dataset_cfg.get("dataset", "ActivityNet"))

    # Load vocab
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    keep_vocab = build_vocab_mapping(vocab, dataset_cfg["vocab_size"])

    # Inject runtime config fields (matching PPS train.py)
    model_cfg["vocab_size"] = len(keep_vocab) + 1
    model_cfg["max_epoch"] = train_cfg["num_epochs"]
    model_cfg["max_num_words"] = dataset_cfg["max_num_words"]

    # Build model (PPS config is flat, no nested "config" key)
    model = PPS(model_cfg)
    model = model.to(DEVICE)
    model.eval()

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict["model_parameters"])

    # Determine feature key
    feature_key = "c3d_features" if dataset_name == "ActivityNet" else None

    return {
        "model": model,
        "vocab": vocab,
        "keep_vocab": keep_vocab,
        "feature_path": feature_path,
        "feature_key": feature_key,
        "dataset_name": dataset_name,
    }


# ── Cache dirs ─────────────────────────────────────────────────────────────

CACHE_VIDEOS = REPO_ROOT / "cache" / "videos"
CACHE_RESULTS = REPO_ROOT / "cache" / "results"


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(title="PPS Video Grounding Inference", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}
GT_TABLE = {}


def _build_gt_table():
    """Build {video_id: [(sentence, start, end), ...]} from data JSONs."""
    gt = {}
    data_root = PPS_ROOT / "data"
    for dataset_key, json_names in [
        ("activitynet", ["train_data.json", "test_data.json"]),
        ("charades", ["train.json", "test.json"]),
    ]:
        data_dir = data_root / dataset_key
        for jn in json_names:
            jp = data_dir / jn
            if not jp.exists():
                continue
            with open(jp) as f:
                for item in json.load(f):
                    vid, duration, timestamps, sentence = item
                    entry = (sentence.strip().lower(), float(timestamps[0]), float(timestamps[1]))
                    if vid not in gt:
                        gt[vid] = []
                    if entry not in gt[vid]:
                        gt[vid].append(entry)
    print(f"GT table: {len(gt)} videos indexed")
    return gt


def _find_best_gt(video_id: str, query: str):
    entries = GT_TABLE.get(video_id, [])
    if not entries:
        return None
    query_words = set(query.lower().split())
    best, best_overlap = None, 0
    for sentence, start, end in entries:
        overlap = len(query_words & set(sentence.split()))
        if overlap > best_overlap:
            best_overlap = overlap
            best = (sentence, start, end)
    return best


@app.on_event("startup")
def startup():
    global GT_TABLE
    print(f"Using device: {DEVICE} | PPS server on port 8200")
    CACHE_VIDEOS.mkdir(parents=True, exist_ok=True)
    CACHE_RESULTS.mkdir(parents=True, exist_ok=True)
    GT_TABLE = _build_gt_table()

    # ActivityNet
    print("Loading PPS ActivityNet model...")
    MODELS["activitynet"] = load_pps_model(str(PPS_ROOT / "config/activitynet/config_refact.json"))
    print("  PPS ActivityNet loaded.")

    # Charades
    print("Loading PPS Charades model...")
    MODELS["charades"] = load_pps_model(str(PPS_ROOT / "config/charades/config_refact.json"))
    print("  PPS Charades loaded.")
    print("Server ready.")


def identify_dataset(filename: str) -> str:
    name = Path(filename).stem
    if name.startswith("v_"):
        return "activitynet"
    return "charades"


@app.post("/predict")
async def predict(video: UploadFile = File(...), query: str = Form(...)):
    filename = video.filename or "unknown.mp4"
    dataset = identify_dataset(filename)
    video_id = Path(filename).stem

    model_info = MODELS[dataset]
    torch_model = model_info["model"]
    vocab = model_info["vocab"]
    keep_vocab = model_info["keep_vocab"]
    feature_path = model_info["feature_path"]
    feature_key = model_info["feature_key"]

    # Cache key
    video_bytes = await video.read()
    cache_key = hashlib.md5(video_bytes + query.encode() + b"pps").hexdigest()

    cached_json = CACHE_RESULTS / f"{cache_key}.json"
    if cached_json.exists():
        with open(cached_json) as f:
            return json.load(f)

    cached_video_path = CACHE_VIDEOS / f"{cache_key}_{filename}"
    with open(cached_video_path, "wb") as f:
        f.write(video_bytes)

    try:
        duration = get_video_duration(str(cached_video_path))
    except RuntimeError:
        raise HTTPException(400, "Failed to read video duration from file")

    try:
        frames_feat = load_and_sample_features(feature_path, video_id, feature_key)
    except KeyError:
        raise HTTPException(400, f"Video '{video_id}' not found in feature file")

    try:
        words_id, words_feat, weights = process_query(query, vocab, keep_vocab)
    except ValueError as e:
        raise HTTPException(400, str(e))

    words_len_val = len(words_id)
    batch = {
        "frames_feat": torch.from_numpy(frames_feat).unsqueeze(0).to(DEVICE),
        "frames_len": torch.tensor([MAX_NUM_FRAMES]).to(DEVICE),
        "words_feat": torch.from_numpy(words_feat).unsqueeze(0).to(DEVICE),
        "words_id": torch.tensor(words_id).unsqueeze(0).to(DEVICE),
        "words_len": torch.tensor([words_len_val]).to(DEVICE),
        "weights": torch.from_numpy(weights).unsqueeze(0).to(DEVICE),
    }

    with torch.no_grad():
        output = torch_model(epoch=0, **batch)

    start_norm, end_norm = select_best_proposal_pps(
        output, torch_model.pos_mask_list, model_info["dataset_name"]
    )

    start_time = start_norm * duration
    end_time = end_norm * duration

    gt_entry = _find_best_gt(video_id, query)
    gt_interval = [round(gt_entry[1], 2), round(gt_entry[2], 2)] if gt_entry else None

    result = {
        "success": True,
        "video_name": filename,
        "video_id": video_id,
        "dataset": dataset,
        "model": "pps",
        "query": query,
        "interval": [round(start_time, 2), round(end_time, 2)],
        "duration": round(duration, 2),
        "selection": "vote",
        "gt": gt_interval,
        "gt_sentence": gt_entry[0] if gt_entry else None,
        "cached_video": str(cached_video_path),
    }

    with open(cached_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
