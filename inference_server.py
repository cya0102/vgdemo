"""
Video Grounding Inference Server (FastAPI)
Serves web UI for CPL & CPL-MoE models on port 8100.
PPS model served on port 8200 via pps_inference_server.py.
"""
import hashlib
import os
import sys
import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import h5py
import nltk
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
CPL_ROOT = REPO_ROOT / "cpl-main"
CPLMOE_ROOT = REPO_ROOT / "cplmoe-main"
sys.path.insert(0, str(CPL_ROOT))
sys.path.insert(0, str(CPLMOE_ROOT))

from models.cpl import CPL
from models.cpl_moe import CPL_MoE
from models.loss import cal_nll_loss

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ──────────────────────────────────────────────────────────────
MAX_NUM_FRAMES = 200
MAX_NUM_WORDS = 20
PPS_PORT = 8200

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


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0),
             np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0),
             np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def select_best_proposal(output: dict, use_vote: bool):
    """Select best proposal from CPL/CPL-MoE output. Returns (start, end) in [0,1]."""
    bsz = 1
    num_props = output["center"].shape[0] // bsz

    words_mask = (output["words_mask"].unsqueeze(1)
                  .expand(bsz, num_props, -1)
                  .contiguous()
                  .view(bsz * num_props, -1))
    words_id = (output["words_id"].unsqueeze(1)
                .expand(bsz, num_props, -1)
                .contiguous()
                .view(bsz * num_props, -1))

    nll_loss, _ = cal_nll_loss(output["words_logit"], words_id, words_mask)
    idx = nll_loss.view(bsz, num_props).argsort(dim=-1)

    width = output["width"].view(bsz, num_props).gather(index=idx, dim=-1)
    center = output["center"].view(bsz, num_props).gather(index=idx, dim=-1)

    selected_props = torch.stack([
        torch.clamp(center - width / 2, min=0),
        torch.clamp(center + width / 2, max=1),
    ], dim=-1).cpu().numpy()

    if use_vote:
        c = np.ones((bsz, num_props))
        votes = np.zeros((bsz, num_props))
        for i in range(num_props):
            for j in range(num_props):
                iou = calculate_IoU_batch(
                    (selected_props[:, i, 0], selected_props[:, i, 1]),
                    (selected_props[:, j, 0], selected_props[:, j, 1]),
                )
                votes[:, i] += iou * c[:, j]
        best_idx = int(np.argmax(votes, axis=1)[0])
    else:
        best_idx = 0

    return float(selected_props[0, best_idx, 0]), float(selected_props[0, best_idx, 1])


# ── Model loader (CPL & CPL-MoE share the same logic) ──────────────────────

def load_model(config_path: str, vocab_path: str, checkpoint_path: str,
               model_cls, root: Path):
    """Load a CPL or CPL-MoE model, vocab, and keep_vocab mapping."""
    with open(config_path) as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    # Resolve relative paths (config paths are relative to model root/)
    vocab_path = str(root / vocab_path)
    feature_path = dataset_cfg["feature_path"]
    if not os.path.isabs(feature_path):
        feature_path = str(root / feature_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(root / checkpoint_path)

    # Load vocab
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    keep_vocab = build_vocab_mapping(vocab, dataset_cfg["vocab_size"])

    # Inject runtime config fields
    model_cfg["config"]["vocab_size"] = len(keep_vocab) + 1
    model_cfg["config"]["max_epoch"] = config["train"]["max_num_epochs"]

    # Build model
    model = model_cls(model_cfg["config"])
    model = model.to(DEVICE)
    model.eval()

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict["model_parameters"])

    dataset_name = dataset_cfg.get("dataset", dataset_cfg.get("name", "ActivityNet"))
    return {
        "model": model,
        "vocab": vocab,
        "keep_vocab": keep_vocab,
        "feature_path": feature_path,
        "feature_key": "c3d_features" if dataset_name == "ActivityNet" else None,
        "dataset_name": dataset_name,
        "model_name": model_cfg["name"],
    }


# ── Cache dirs ─────────────────────────────────────────────────────────────

CACHE_VIDEOS = REPO_ROOT / "cache" / "videos"
CACHE_RESULTS = REPO_ROOT / "cache" / "results"


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(title="Video Grounding Inference", version="2.0.0")

MODELS = {}  # key: "cpl_activitynet", "cplmoe_charades", etc.
GT_TABLE = {}  # key: video_id -> {query: [start_sec, end_sec]} (all sentences for that video)


def _model_key(model_type: str, dataset: str) -> str:
    return f"{model_type}_{dataset}"


def _load_if_needed(model_type: str, dataset: str):
    """Lazy-load a model on first request."""
    key = _model_key(model_type, dataset)
    if key in MODELS:
        return

    if model_type == "cpl":
        root = CPL_ROOT
        cls = CPL
        configs = {
            "activitynet": "config/activitynet/main.json",
            "charades": "config/charades/main.json",
        }
        checkpoints = {
            "activitynet": "checkpoints/activitynet/model-best.pt",
            "charades": "checkpoints/charades/model-best.pt",
        }
    else:  # cplmoe
        root = CPLMOE_ROOT
        cls = CPL_MoE
        configs = {
            "activitynet": "config/activitynet/main_moe.json",
            "charades": "config/charades/main_moe.json",
        }
        checkpoints = {
            "activitynet": "checkpoints/activitynet_moe/model-best.pt",
            "charades": "checkpoints/charades_moe/model-best.pt",
        }

    dc = dataset
    vocab_path = f"data/{'activitynet' if dc == 'activitynet' else 'charades'}/glove.pkl"
    print(f"Loading {model_type.upper()} {dc} model...")
    MODELS[key] = load_model(
        config_path=str(root / configs[dc]),
        vocab_path=vocab_path,
        checkpoint_path=checkpoints[dc],
        model_cls=cls,
        root=root,
    )
    print(f"  {model_type.upper()} {dc} loaded.")


def _build_gt_table():
    """Build {video_id: [(sentence, start, end), ...]} from data JSONs."""
    import json as _json
    gt = {}
    for dataset_key, json_names in [
        ("activitynet", ["train_data.json", "test_data.json", "val_data.json"]),
        ("charades", ["train.json", "test.json"]),
    ]:
        data_dir = CPL_ROOT / "data" / dataset_key
        for jn in json_names:
            jp = data_dir / jn
            if not jp.exists():
                continue
            with open(jp) as f:
                for item in _json.load(f):
                    vid, duration, timestamps, sentence = item
                    entry = (sentence.strip().lower(), float(timestamps[0]), float(timestamps[1]))
                    if vid not in gt:
                        gt[vid] = []
                    # Deduplicate
                    if entry not in gt[vid]:
                        gt[vid].append(entry)
    print(f"GT table: {len(gt)} videos indexed")
    return gt


def _find_best_gt(video_id: str, query: str):
    """Find the most relevant GT segment by word overlap with the query."""
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
    print(f"Using device: {DEVICE} | CPL+CPL-MoE on port 8100 | PPS on port {PPS_PORT}")
    CACHE_VIDEOS.mkdir(parents=True, exist_ok=True)
    CACHE_RESULTS.mkdir(parents=True, exist_ok=True)
    GT_TABLE = _build_gt_table()
    print("Models will be loaded on first request (lazy).")
    print("Server ready.")


def identify_dataset(filename: str) -> str:
    name = Path(filename).stem
    if name.startswith("v_"):
        return "activitynet"
    return "charades"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/predict")
async def predict(video: UploadFile = File(...), query: str = Form(...),
                  model: str = Form("cpl")):
    # Validate model
    if model not in ("cpl", "cplmoe"):
        raise HTTPException(400, f"Unknown model '{model}'. Use 'cpl', 'cplmoe', or 'pps'.")

    filename = video.filename or "unknown.mp4"
    dataset = identify_dataset(filename)
    video_id = Path(filename).stem

    # Lazy-load model
    _load_if_needed(model, dataset)

    model_info = MODELS[_model_key(model, dataset)]
    torch_model = model_info["model"]
    vocab = model_info["vocab"]
    keep_vocab = model_info["keep_vocab"]
    feature_path = model_info["feature_path"]
    feature_key = model_info["feature_key"]

    # Cache key = hash(video + query + model)
    video_bytes = await video.read()
    cache_key = hashlib.md5(video_bytes + query.encode() + model.encode()).hexdigest()

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

    use_vote = (dataset == "activitynet")
    start_norm, end_norm = select_best_proposal(output, use_vote=use_vote)

    start_time = start_norm * duration
    end_time = end_norm * duration

    # GT lookup
    gt_entry = _find_best_gt(video_id, query)
    gt_interval = [round(gt_entry[1], 2), round(gt_entry[2], 2)] if gt_entry else None

    result = {
        "success": True,
        "video_name": filename,
        "video_id": video_id,
        "dataset": dataset,
        "model": model,
        "query": query,
        "interval": [round(start_time, 2), round(end_time, 2)],
        "duration": round(duration, 2),
        "selection": "vote" if use_vote else "loss",
        "gt": gt_interval,
        "gt_sentence": gt_entry[0] if gt_entry else None,
        "cached_video": str(cached_video_path),
    }

    with open(cached_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ── HTML Page ──────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>视频时序定位</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 50%, #f0fdf4 100%);
    color: #1e293b; min-height: 100vh; padding: 40px 20px;
  }
  .container { max-width: 1100px; margin: 0 auto; }

  .header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
    border-radius: 12px; padding: 24px 32px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(37,99,235,.25);
  }
  .header h1 { font-size: 1.3rem; color: #fff; font-weight: 700; letter-spacing: -.01em; }
  .header p { color: rgba(255,255,255,.75); font-size: .8rem; margin-top: 4px; }

  /* Two-column layout */
  .main-row { display: flex; gap: 20px; align-items: stretch; }
  .main-row .left-col { flex: 1; min-width: 0; }
  .main-row .right-col { flex: 1; min-width: 0; }

  .card {
    background: #fff; border-radius: 12px; padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    border: 1px solid #e2e8f0;
  }

  /* Model selector */
  .model-selector {
    display: flex; gap: 10px; margin-bottom: 22px;
  }
  .model-btn {
    flex: 1; padding: 8px 0; border: 2px solid #e2e8f0; border-radius: 20px;
    background: #fff; color: #64748b; font-size: .8rem; font-weight: 600;
    cursor: pointer; transition: all .2s; text-align: center;
  }
  .model-btn:hover { border-color: #93c5fd; color: #3b82f6; }
  .model-btn.active {
    background: #2563eb; border-color: #2563eb; color: #fff;
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #cbd5e1; border-radius: 10px; padding: 22px;
    text-align: center; cursor: pointer; transition: all .2s;
    background: #f8fafc; margin-bottom: 16px;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: #3b82f6; background: #eff6ff;
  }
  .upload-zone .icon { font-size: 2rem; margin-bottom: 8px; }
  .upload-zone .text { color: #64748b; font-size: .875rem; }
  .upload-zone .file-name { color: #1e40af; font-weight: 600; font-size: .85rem; margin-top: 6px; }
  .upload-zone input[type="file"] { display: none; }

  label { display: block; font-weight: 600; margin-bottom: 6px; font-size: .8rem;
    color: #64748b; text-transform: uppercase; letter-spacing: .05em; }
  .query-input {
    width: 100%; padding: 12px 16px; border: 1.5px solid #e2e8f0;
    border-radius: 10px; font-size: .95rem; margin-bottom: 18px; outline: none;
    transition: border-color .2s, box-shadow .2s;
  }
  .query-input:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,.15); }

  button.submit-btn {
    width: 100%; padding: 14px; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: all .2s;
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: #fff; box-shadow: 0 2px 8px rgba(37,99,235,.3);
  }
  button.submit-btn:hover { box-shadow: 0 4px 16px rgba(37,99,235,.4); transform: translateY(-1px); }
  button.submit-btn:active { transform: translateY(0); }
  button.submit-btn:disabled {
    background: #cbd5e1; color: #94a3b8; box-shadow: none;
    cursor: not-allowed; animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

  .result-card { background: #fff; border-radius: 12px; padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    border: 1px solid #e2e8f0; min-height: 200px; }
  .result-card h2 { font-size: .8rem; color: #64748b; margin-bottom: 18px;
    font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }

  .result-empty { color: #94a3b8; text-align: center; padding: 32px 0; font-size: .9rem; }
  .spinner-box { text-align: center; padding: 32px 0; color: #64748b; font-size: .9rem; }
  .result-error { color: #dc2626; text-align: center; padding: 24px 0; font-size: .9rem;
    background: #fef2f2; border-radius: 8px; }

  /* Compare: GT vs Prediction */
  .compare-row { display: flex; gap: 10px; margin-bottom: 16px; }
  .gt-box, .pred-box { flex: 1; border-radius: 10px; padding: 14px; text-align: center; }
  .gt-box { background: #f8fafc; border: 2px solid #cbd5e1; }
  .gt-label { font-size: .7rem; color: #64748b; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }
  .gt-timestamp { font-size: 1.15rem; font-weight: 700; color: #475569; font-family: 'SF Mono', 'Menlo', monospace; }
  .gt-sentence { font-size: .75rem; color: #94a3b8; font-style: italic; margin-top: 6px; line-height: 1.4; }
  .pred-box { background: linear-gradient(135deg, #ecfdf5, #f0fdf4); border: 2px solid #10b981; }
  .pred-label { font-size: .7rem; color: #065f46; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }
  .pred-timestamp { font-size: 1.15rem; font-weight: 700; color: #065f46; font-family: 'SF Mono', 'Menlo', monospace; }
  .pred-method { font-size: .7rem; color: #10b981; margin-top: 6px; }

  .timeline { margin-bottom: 20px; }
  .timeline-label { font-size: .72rem; color: #64748b; font-weight: 600; margin-bottom: 8px; }
  .timeline .track {
    position: relative; height: 32px; background: #f1f5f9;
    border-radius: 16px; overflow: hidden;
  }
  .timeline .track .fill {
    position: absolute; top: 0; height: 100%; border-radius: 16px; transition: all .3s;
  }
  .timeline .track .gt-fill { background: #cbd5e1; z-index: 1; }
  .timeline .track .pred-fill { background: linear-gradient(90deg, #3b82f6, #6366f1); z-index: 2; opacity: .85; }
  .timeline .labels {
    display: flex; justify-content: space-between; font-size: .75rem;
    color: #94a3b8; margin-top: 6px; padding: 0 4px;
  }
  .legend { display: flex; gap: 16px; justify-content: center; margin-top: 8px; font-size: .72rem; color: #64748b; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; }
  .gt-dot { background: #cbd5e1; }
  .pred-dot { background: #3b82f6; }

  .meta-grid {
    display: grid; grid-template-columns: auto 1fr; gap: 6px 16px;
    font-size: .82rem; line-height: 1.7;
  }
  .meta-grid .key { color: #64748b; font-weight: 500; white-space: nowrap; }
  .meta-grid .val { color: #1e293b; word-break: break-all; }
  .meta-grid .val .tag {
    display: inline-block; background: #eff6ff; color: #3b82f6;
    padding: 1px 8px; border-radius: 4px; font-size: .75rem; font-weight: 600;
  }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>视频时序定位</h1>
    <p>弱监督单视频推理 &mdash; CPL / PPS / CPL-MoE</p>
  </div>

  <div class="card" style="padding:18px 24px; margin-bottom:20px">
    <label style="margin-bottom:10px">选择模型</label>
    <div class="model-selector" id="model-selector" style="margin-bottom:0">
      <div class="model-btn active" data-model="cpl">CPL</div>
      <div class="model-btn" data-model="pps">PPS</div>
      <div class="model-btn" data-model="cplmoe">CPL-MoE</div>
    </div>
  </div>

  <div class="main-row">
    <div class="left-col">

  <div class="card">
    <form id="form">
      <label>上传视频</label>
      <div class="upload-zone" id="upload-zone">
        <div class="icon">&#x1F3AC;</div>
        <div class="text">点击或拖拽视频文件到此处</div>
        <div class="file-name" id="file-name"></div>
        <input type="file" id="video" name="video" accept="video/*" required>
      </div>

      <label for="query">查询文本</label>
      <input type="text" id="query" name="query" class="query-input"
             placeholder="描述你想定位的画面，例如：一个人正在跑步" required>

      <button type="submit" id="submit-btn" class="submit-btn">查找片段</button>
    </form>
  </div>

    </div>
    <div class="right-col">

  <div class="result-card">
    <h2>定位结果</h2>
    <div id="result-area">
      <div class="result-empty">上传视频并输入查询文本，然后点击"查找片段"。</div>
    </div>
  </div>

    </div>
  </div>

</div>

<script>
var selectedModel = 'cpl';
var PPS_PORT = """ + str(PPS_PORT) + """;

// Ground truth for demo video v_DRWMUsADKFM
var DEMO_GT = [
  {query: "A camera pans around a room and leads into a room rubbing paper down and putting a box in the middle.", start: 0.92, end: 56.74},
  {query: "The woman wraps the box up in paper and pushing in the sides.", start: 54.91, end: 128.11},
  {query: "She tapes up the sides and uses a ribbon to tie the box up and ends by unwrapping it and showing what's inside.", start: 127.2, end: 181.19}
];

// Model selector
document.getElementById('model-selector').addEventListener('click', function(e) {
  if (e.target.classList.contains('model-btn')) {
    document.querySelectorAll('.model-btn').forEach(function(b) { b.classList.remove('active'); });
    e.target.classList.add('active');
    selectedModel = e.target.dataset.model;
  }
});

// Upload zone
var zone = document.getElementById('upload-zone');
var fileInput = document.getElementById('video');
var fileNameEl = document.getElementById('file-name');

zone.addEventListener('click', function() { fileInput.click(); });
zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', function() { zone.classList.remove('dragover'); });
zone.addEventListener('drop', function(e) {
  e.preventDefault(); zone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    updateFileName();
  }
});
fileInput.addEventListener('change', updateFileName);
function updateFileName() {
  fileNameEl.textContent = fileInput.files[0] ? fileInput.files[0].name : '';
}

// Form submit
var form = document.getElementById('form');
var resultArea = document.getElementById('result-area');
var btn = document.getElementById('submit-btn');

form.addEventListener('submit', function(e) {
  e.preventDefault();
  var video = fileInput.files[0];
  var query = document.getElementById('query').value;
  if (!video || !query) return;

  btn.disabled = true;
  btn.textContent = '推理中...';
  resultArea.innerHTML = '<div class="spinner-box">正在使用 ' + selectedModel.toUpperCase() + ' 推理&hellip;</div>';

  var data = new FormData();
  data.append('video', video);
  data.append('query', query);

  // PPS uses separate server on port 8200
  var fetchUrl;
  if (selectedModel === 'pps') {
    fetchUrl = window.location.protocol + '//' + window.location.hostname + ':' + PPS_PORT + '/predict';
    // PPS doesn't need model field (it's the only model there)
  } else {
    fetchUrl = '/predict';
    data.append('model', selectedModel);
  }

  fetch(fetchUrl, { method: 'POST', body: data })
    .then(function(resp) { return resp.json().then(function(json) { return {ok: resp.ok, json: json}; }); })
    .then(function(r) {
      if (r.ok && r.json.success) { showResult(r.json); }
      else { resultArea.innerHTML = '<div class="result-error">' + (r.json.detail || '未知错误') + '</div>'; }
    })
    .catch(function(err) {
      resultArea.innerHTML = '<div class="result-error">请求失败：' + err.message + '</div>';
    })
    .finally(function() {
      btn.disabled = false;
      btn.textContent = '查找片段';
    });
});

function showResult(json) {
  var dur = json.duration;
  var start = json.interval[0];
  var end = json.interval[1];
  var leftPct = (start / dur * 100).toFixed(1);
  var widthPct = ((end - start) / dur * 100).toFixed(1);

  // Match GT: demo video or server-provided
  var gtMatch = null;
  if (json.video_id === 'v_DRWMUsADKFM') {
    // Find exact match by query text
    var q = json.query.trim().toLowerCase();
    for (var i = 0; i < DEMO_GT.length; i++) {
      if (DEMO_GT[i].query.trim().toLowerCase() === q) {
        gtMatch = DEMO_GT[i];
        break;
      }
    }
    // Fallback: word overlap
    if (!gtMatch) {
      var qWords = q.split(' ');
      var bestOverlap = 0;
      for (var i = 0; i < DEMO_GT.length; i++) {
        var gWords = DEMO_GT[i].query.split(' ');
        var overlap = 0;
        for (var j = 0; j < qWords.length; j++) {
          for (var k = 0; k < gWords.length; k++) {
            if (qWords[j] === gWords[k]) overlap++;
          }
        }
        if (overlap > bestOverlap) { bestOverlap = overlap; gtMatch = DEMO_GT[i]; }
      }
    }
  } else if (json.gt) {
    gtMatch = {query: json.gt_sentence, start: json.gt[0], end: json.gt[1]};
  }

  var gtHtml = '', gtTrackHtml = '';
  if (gtMatch) {
    var gs = gtMatch.start, ge = gtMatch.end;
    var gl = (gs / dur * 100).toFixed(1), gw = ((ge - gs) / dur * 100).toFixed(1);
    gtHtml =
      '<div class="gt-box">' +
        '<div class="gt-label">真实标注 (Ground Truth)</div>' +
        '<div class="gt-timestamp">[' + gs.toFixed(2) + 's &mdash; ' + ge.toFixed(2) + 's]</div>' +
        (gtMatch.query ? '<div class="gt-sentence">&ldquo;' + gtMatch.query.substring(0, 80) + '&hellip;&rdquo;</div>' : '') +
      '</div>';
    gtTrackHtml = '<div class="fill gt-fill" style="left:' + gl + '%;width:' + gw + '%;"></div>';
  }

  resultArea.innerHTML =
    '<div class="result-success">' +
      '<div class="compare-row">' +
        gtHtml +
        '<div class="pred-box">' +
          '<div class="pred-label">模型预测 (' + (json.model || selectedModel).toUpperCase() + ')</div>' +
          '<div class="pred-timestamp">[' + start.toFixed(2) + 's &mdash; ' + end.toFixed(2) + 's]</div>' +
          '<div class="pred-method">策略：' + (json.selection === 'vote' ? '投票机制' : '损失最小') + '</div>' +
        '</div>' +
      '</div>' +

      '<div class="timeline">' +
        '<div class="timeline-label">时间轴</div>' +
        '<div class="track">' +
          gtTrackHtml +
          '<div class="fill pred-fill" style="left:' + leftPct + '%;width:' + widthPct + '%;"></div>' +
        '</div>' +
        '<div class="labels"><span>0s</span><span>' + dur.toFixed(0) + 's</span></div>' +
        '<div class="legend">' +
          (gtMatch ? '<span class="legend-item"><span class="legend-dot gt-dot"></span>真实标注</span>' : '') +
          '<span class="legend-item"><span class="legend-dot pred-dot"></span>模型预测</span>' +
        '</div>' +
      '</div>' +

      '<div class="meta-grid">' +
        '<span class="key">视频</span><span class="val">' + json.video_name + '</span>' +
        '<span class="key">时长</span><span class="val">' + dur.toFixed(2) + 's</span>' +
        '<span class="key">数据集</span><span class="val"><span class="tag">' + json.dataset + '</span></span>' +
        '<span class="key">模型</span><span class="val"><span class="tag">' + (json.model || selectedModel).toUpperCase() + '</span></span>' +
        '<span class="key">查询</span><span class="val">&ldquo;' + json.query.substring(0, 60) + '&hellip;&rdquo;</span>' +
      '</div>' +
    '</div>';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
