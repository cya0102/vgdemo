"""
CPL Single Video Inference Server (FastAPI)
Serves a web UI for temporal video grounding with CPL model.
"""
import os
import sys
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import h5py
import nltk
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
CPL_ROOT = REPO_ROOT / "cpl-main"
sys.path.insert(0, str(CPL_ROOT))

from models.cpl import CPL
from models.loss import cal_nll_loss

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ──────────────────────────────────────────────────────────────
MAX_NUM_FRAMES = 200
MAX_NUM_WORDS = 20

# ── Helpers ────────────────────────────────────────────────────────────────

def get_video_duration(filepath: str) -> float:
    """Read video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", filepath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def build_vocab_mapping(vocab: dict, vocab_size: int) -> dict:
    """Build word→id mapping, matching BaseDataset.keep_vocab."""
    keep_vocab = {}
    for w, _ in vocab["counter"].most_common(vocab_size):
        keep_vocab[w] = len(keep_vocab) + 1
    return keep_vocab


def process_query(query: str, vocab: dict, keep_vocab: dict):
    """Tokenize query and build word features, matching BaseDataset.__getitem__."""
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

    # Truncate
    words = words[:MAX_NUM_WORDS]
    weights = weights[:MAX_NUM_WORDS]

    if len(words) == 0:
        raise ValueError("No known words in query (all OOV)")

    words_id = [keep_vocab[w] for w in words]
    words_feat = [
        vocab["id2vec"][vocab["w2id"][words[0]]].astype(np.float32)  # placeholder for start token
    ]
    words_feat.extend(
        vocab["id2vec"][vocab["w2id"][w]].astype(np.float32) for w in words
    )
    return words_id, np.array(words_feat, dtype=np.float32), np.array(weights, dtype=np.float32)


def load_and_sample_features(hdf5_path: str, video_id: str, feature_key: Optional[str]):
    """Load frame features from HDF5 and sample to MAX_NUM_FRAMES."""
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
    """Vectorized IoU between two sets of intervals."""
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0),
             np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0),
             np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def select_best_proposal(output: dict, use_vote: bool):
    """Select best proposal from model output. Returns (start, end) in [0,1]."""
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
    ], dim=-1).cpu().numpy()  # (1, num_props, 2)

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


# ── Model loader ───────────────────────────────────────────────────────────

def load_model(config_path: str, vocab_path: str, checkpoint_path: str):
    """Load CPL model, vocab, and keep_vocab mapping."""
    with open(config_path) as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]

    # Resolve relative paths (config paths are relative to cpl-main/)
    vocab_path = str(CPL_ROOT / vocab_path)
    feature_path = str(CPL_ROOT / dataset_cfg["feature_path"]) if not os.path.isabs(dataset_cfg["feature_path"]) else dataset_cfg["feature_path"]
    checkpoint_path = str(CPL_ROOT / checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path

    # Load vocab
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    keep_vocab = build_vocab_mapping(vocab, dataset_cfg["vocab_size"])

    # Build model
    model = CPL(model_cfg["config"])
    model = model.to(DEVICE)
    model.eval()

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict["model_parameters"])

    return {
        "model": model,
        "vocab": vocab,
        "keep_vocab": keep_vocab,
        "feature_path": feature_path,
        "feature_key": "c3d_features" if dataset_cfg["dataset"] == "ActivityNet" else None,
        "dataset_name": dataset_cfg["dataset"],
    }


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(title="CPL Video Grounding", version="0.1.0")

# Globals set in startup
MODELS = {}
DATASETS = {}  # video_name_prefix → dataset key


@app.on_event("startup")
def startup():
    """Load both ActivityNet and Charades models."""
    print(f"Using device: {DEVICE}")

    # ActivityNet
    print("Loading ActivityNet model...")
    MODELS["activitynet"] = load_model(
        config_path=str(CPL_ROOT / "config/activitynet/main.json"),
        vocab_path="data/activitynet/glove.pkl",
        checkpoint_path="checkpoints/activitynet/model-best.pt",
    )
    print("  ActivityNet model loaded.")

    # Charades
    print("Loading Charades model...")
    MODELS["charades"] = load_model(
        config_path=str(CPL_ROOT / "config/charades/main.json"),
        vocab_path="data/charades/glove.pkl",
        checkpoint_path="checkpoints/charades/model-best.pt",
    )
    print("  Charades model loaded.")
    print("Server ready.")


def identify_dataset(filename: str) -> str:
    """Identify dataset from video filename."""
    name = Path(filename).stem
    if name.startswith("v_"):
        return "activitynet"
    return "charades"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.post("/predict")
async def predict(video: UploadFile = File(...), query: str = Form(...)):
    # 1. Identify dataset
    filename = video.filename or "unknown.mp4"
    dataset = identify_dataset(filename)
    video_id = Path(filename).stem

    model_info = MODELS[dataset]
    model = model_info["model"]
    vocab = model_info["vocab"]
    keep_vocab = model_info["keep_vocab"]
    feature_path = model_info["feature_path"]
    feature_key = model_info["feature_key"]

    # 2. Save uploaded video to temp file, get duration
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            shutil.copyfileobj(video.file, tmp)
            tmp_path = tmp.name

        try:
            duration = get_video_duration(tmp_path)
        except RuntimeError:
            raise HTTPException(400, "Failed to read video duration from file")

        # 3. Load features
        try:
            frames_feat = load_and_sample_features(feature_path, video_id, feature_key)
        except KeyError:
            raise HTTPException(400, f"Video '{video_id}' not found in feature file")

        # 4. Process query
        try:
            words_id, words_feat, weights = process_query(query, vocab, keep_vocab)
        except ValueError as e:
            raise HTTPException(400, str(e))

        # 5. Inference
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
            output = model(epoch=0, **batch)

        # 6. Select best proposal
        use_vote = (dataset == "activitynet")
        start_norm, end_norm = select_best_proposal(output, use_vote=use_vote)

        # 7. Convert to real timestamps
        start_time = start_norm * duration
        end_time = end_norm * duration

        return {
            "success": True,
            "video_name": filename,
            "video_id": video_id,
            "dataset": dataset,
            "interval": [round(start_time, 2), round(end_time, 2)],
            "duration": round(duration, 2),
            "selection": "vote" if use_vote else "loss",
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── HTML Page ──────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CPL Video Grounding</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #333; padding: 40px 20px; }
  .container { max-width: 640px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 24px; color: #1a1a1a; }
  .card { background: #fff; border-radius: 8px; padding: 24px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  label { display: block; font-weight: 600; margin-bottom: 6px; font-size: .875rem; }
  input[type="file"], input[type="text"] { width: 100%; padding: 10px 12px;
    border: 1px solid #ddd; border-radius: 6px; font-size: .9rem; margin-bottom: 16px; }
  button { width: 100%; padding: 12px; background: #2563eb; color: #fff;
    border: none; border-radius: 6px; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: background .15s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #93c5fd; cursor: not-allowed; }
  .result { margin-top: 20px; padding: 16px; border-radius: 6px; display: none; }
  .result.success { background: #ecfdf5; border: 1px solid #a7f3d0; display: block; }
  .result.error { background: #fef2f2; border: 1px solid #fecaca; display: block; }
  .result .interval { font-size: 1.5rem; font-weight: 700; color: #065f46; }
  .result .meta { font-size: .8rem; color: #6b7280; margin-top: 8px; }
  .result.error .msg { color: #991b1b; }
  .spinner { display: none; text-align: center; padding: 16px; color: #6b7280; }
</style>
</head>
<body>
<div class="container">
  <h1>CPL Temporal Video Grounding</h1>
  <div class="card">
    <form id="form">
      <label for="video">Video file</label>
      <input type="file" id="video" name="video" accept="video/*" required>

      <label for="query">Text query</label>
      <input type="text" id="query" name="query"
             placeholder="e.g. a person is running"
             required>

      <button type="submit" id="submit-btn">Find segment</button>
    </form>

    <div id="spinner" class="spinner">Running inference...</div>
    <div id="result" class="result"></div>
  </div>
</div>

<script>
const form = document.getElementById('form');
const result = document.getElementById('result');
const spinner = document.getElementById('spinner');
const btn = document.getElementById('submit-btn');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const video = document.getElementById('video').files[0];
  const query = document.getElementById('query').value;
  if (!video || !query) return;

  btn.disabled = true;
  spinner.style.display = 'block';
  result.style.display = 'none';

  const data = new FormData();
  data.append('video', video);
  data.append('query', query);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: data });
    const json = await resp.json();
    if (resp.ok && json.success) {
      result.className = 'result success';
      result.innerHTML = `
        <div class="interval">[${json.interval[0]}, ${json.interval[1]}]</div>
        <div class="meta">
          Video: ${json.video_name} &middot;
          Duration: ${json.duration}s &middot;
          Dataset: ${json.dataset} &middot;
          Selection: ${json.selection}
        </div>`;
    } else {
      result.className = 'result error';
      result.innerHTML = `<div class="msg">${json.detail || 'Unknown error'}</div>`;
    }
  } catch (err) {
    result.className = 'result error';
    result.innerHTML = `<div class="msg">Request failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
