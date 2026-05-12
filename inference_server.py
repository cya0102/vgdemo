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
import time
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
    # softmax-normalize weights (matching build_collate_data)
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = np.exp(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    return words_id, np.array(words_feat, dtype=np.float32), weights_arr


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

    # Inject runtime config fields (normally set by Runner._build_model)
    model_cfg["config"]["vocab_size"] = len(keep_vocab) + 1
    model_cfg["config"]["max_epoch"] = config["train"]["max_num_epochs"]

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


# ── Cache dirs ─────────────────────────────────────────────────────────────

CACHE_VIDEOS = REPO_ROOT / "cache" / "videos"
CACHE_RESULTS = REPO_ROOT / "cache" / "results"


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(title="CPL Video Grounding", version="0.1.0")

# Globals set in startup
MODELS = {}


@app.on_event("startup")
def startup():
    """Load both ActivityNet and Charades models."""
    print(f"Using device: {DEVICE}")

    CACHE_VIDEOS.mkdir(parents=True, exist_ok=True)
    CACHE_RESULTS.mkdir(parents=True, exist_ok=True)

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

    # 2. Save uploaded video to cache
    ts = str(int(time.time() * 1000))
    safe_name = f"{ts}_{filename}"
    cached_video_path = CACHE_VIDEOS / safe_name
    with open(cached_video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # 3. Get duration via ffprobe
    try:
        duration = get_video_duration(str(cached_video_path))
    except RuntimeError:
        raise HTTPException(400, "Failed to read video duration from file")

    # 4. Load features
    try:
        frames_feat = load_and_sample_features(feature_path, video_id, feature_key)
    except KeyError:
        raise HTTPException(400, f"Video '{video_id}' not found in feature file")

    # 5. Process query
    try:
        words_id, words_feat, weights = process_query(query, vocab, keep_vocab)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 6. Inference
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

    # 7. Select best proposal
    use_vote = (dataset == "activitynet")
    start_norm, end_norm = select_best_proposal(output, use_vote=use_vote)

    # 8. Convert to real timestamps
    start_time = start_norm * duration
    end_time = end_norm * duration

    result = {
        "success": True,
        "video_name": filename,
        "video_id": video_id,
        "dataset": dataset,
        "query": query,
        "interval": [round(start_time, 2), round(end_time, 2)],
        "duration": round(duration, 2),
        "selection": "vote" if use_vote else "loss",
        "cached_video": str(cached_video_path),
    }

    # 9. Save result JSON
    json_name = f"{ts}_{video_id}.json"
    with open(CACHE_RESULTS / json_name, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ── HTML Page ──────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CPL Video Grounding</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f0f4ff 0%, #f8fafc 50%, #f0fdf4 100%);
    color: #1e293b; min-height: 100vh; padding: 40px 20px;
  }
  .container { max-width: 680px; margin: 0 auto; }

  /* Header */
  .header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(37,99,235,.25);
  }
  .header h1 { font-size: 1.4rem; color: #fff; font-weight: 700; letter-spacing: -.01em; }
  .header p { color: rgba(255,255,255,.75); font-size: .85rem; margin-top: 4px; }

  /* Cards */
  .card {
    background: #fff; border-radius: 12px; padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    margin-bottom: 20px; border: 1px solid #e2e8f0;
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #cbd5e1; border-radius: 10px; padding: 28px;
    text-align: center; cursor: pointer; transition: all .2s;
    background: #f8fafc; margin-bottom: 18px;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: #3b82f6; background: #eff6ff;
  }
  .upload-zone .icon { font-size: 2rem; margin-bottom: 8px; }
  .upload-zone .text { color: #64748b; font-size: .875rem; }
  .upload-zone .file-name { color: #1e40af; font-weight: 600; font-size: .85rem; margin-top: 6px; }
  .upload-zone input[type="file"] { display: none; }

  /* Query input */
  label { display: block; font-weight: 600; margin-bottom: 6px; font-size: .8rem;
    color: #64748b; text-transform: uppercase; letter-spacing: .05em; }
  .query-input {
    width: 100%; padding: 12px 16px; border: 1.5px solid #e2e8f0;
    border-radius: 10px; font-size: .95rem; margin-bottom: 18px; outline: none;
    transition: border-color .2s, box-shadow .2s;
  }
  .query-input:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,.15); }

  /* Button */
  button {
    width: 100%; padding: 14px; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: all .2s;
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: #fff; box-shadow: 0 2px 8px rgba(37,99,235,.3);
  }
  button:hover { box-shadow: 0 4px 16px rgba(37,99,235,.4); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled {
    background: #cbd5e1; color: #94a3b8; box-shadow: none;
    cursor: not-allowed; animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }

  /* Result card */
  .result-card { background: #fff; border-radius: 12px; padding: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    border: 1px solid #e2e8f0; }
  .result-card h2 { font-size: .8rem; color: #64748b; margin-bottom: 18px;
    font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }

  /* States */
  .result-empty { color: #94a3b8; text-align: center; padding: 32px 0; font-size: .9rem; }
  .spinner-box { text-align: center; padding: 32px 0; color: #64748b; font-size: .9rem; }
  .result-error { color: #dc2626; text-align: center; padding: 24px 0; font-size: .9rem;
    background: #fef2f2; border-radius: 8px; }

  /* Success */
  .result-success .timestamp-box {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 2px solid #10b981; border-radius: 12px; padding: 24px;
    text-align: center; margin-bottom: 20px;
  }
  .result-success .timestamp {
    font-size: 1.75rem; font-weight: 700; color: #065f46;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
  }
  .result-success .timestamp .sep { color: #10b981; }

  /* Timeline */
  .timeline { margin-bottom: 20px; }
  .timeline .track {
    position: relative; height: 32px; background: #f1f5f9;
    border-radius: 16px; overflow: hidden;
  }
  .timeline .track .fill {
    position: absolute; top: 0; height: 100%;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    border-radius: 16px; transition: all .3s;
  }
  .timeline .track .handle {
    position: absolute; top: -3px; width: 10px; height: 38px;
    background: #fff; border: 2.5px solid #3b82f6; border-radius: 5px;
    z-index: 2;
  }
  .timeline .labels {
    display: flex; justify-content: space-between; font-size: .75rem;
    color: #94a3b8; margin-top: 6px; padding: 0 4px;
  }
  .timeline .markers {
    position: relative; height: 18px; margin-top: 2px;
  }
  .timeline .markers .marker {
    position: absolute; font-size: .7rem; color: #3b82f6; font-weight: 600;
    transform: translateX(-50%); white-space: nowrap;
  }

  /* Meta grid */
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

  <!-- Header -->
  <div class="header">
    <h1>Temporal Video Grounding</h1>
    <p>CPL: Contrastive Proposal Learning &mdash; Single Video Inference</p>
  </div>

  <!-- Input card -->
  <div class="card">
    <form id="form">
      <label>Video file</label>
      <div class="upload-zone" id="upload-zone">
        <div class="icon">&#x1F3AC;</div>
        <div class="text">Click or drag video file here</div>
        <div class="file-name" id="file-name"></div>
        <input type="file" id="video" name="video" accept="video/*" required>
      </div>

      <label for="query">Text query</label>
      <input type="text" id="query" name="query" class="query-input"
             placeholder="Describe the moment you want to find, e.g. a person is running" required>

      <button type="submit" id="submit-btn">Find Segment</button>
    </form>
  </div>

  <!-- Result card -->
  <div class="result-card">
    <h2>Result</h2>
    <div id="result-area">
      <div class="result-empty">Upload a video and enter a text query, then click Find Segment.</div>
    </div>
  </div>

</div>

<script>
var uploadedFileName = '';

// Upload zone interactions
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
  uploadedFileName = fileInput.files[0] ? fileInput.files[0].name : '';
  fileNameEl.textContent = uploadedFileName || '';
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
  btn.textContent = 'Running...';
  resultArea.innerHTML = '<div class="spinner-box">Running inference &hellip;</div>';

  var data = new FormData();
  data.append('video', video);
  data.append('query', query);

  fetch('/predict', { method: 'POST', body: data })
    .then(function(resp) { return resp.json().then(function(json) { return {ok: resp.ok, json: json}; }); })
    .then(function(r) {
      if (r.ok && r.json.success) { showResult(r.json); }
      else { resultArea.innerHTML = '<div class="result-error">' + (r.json.detail || 'Unknown error') + '</div>'; }
    })
    .catch(function(err) {
      resultArea.innerHTML = '<div class="result-error">Request failed: ' + err.message + '</div>';
    })
    .finally(function() {
      btn.disabled = false;
      btn.textContent = 'Find Segment';
    });
});

function showResult(json) {
  var dur = json.duration;
  var start = json.interval[0];
  var end = json.interval[1];
  var leftPct = (start / dur * 100).toFixed(1);
  var widthPct = ((end - start) / dur * 100).toFixed(1);

  resultArea.innerHTML =
    '<div class="result-success">' +
      '<div class="timestamp-box">' +
        '<div class="timestamp">' +
          start.toFixed(2) + 's  <span class="sep">&mdash;</span>  ' + end.toFixed(2) + 's' +
        '</div>' +
      '</div>' +

      '<div class="timeline">' +
        '<div class="track">' +
          '<div class="fill" style="left:' + leftPct + '%;width:' + widthPct + '%;"></div>' +
          '<div class="handle" style="left:' + leftPct + '%;"></div>' +
          '<div class="handle" style="left:' + (parseFloat(leftPct) + parseFloat(widthPct)) + '%;"></div>' +
        '</div>' +
        '<div class="labels"><span>0s</span><span>' + dur.toFixed(0) + 's</span></div>' +
        '<div class="markers">' +
          '<div class="marker" style="left:' + leftPct + '%;">' + start.toFixed(1) + 's</div>' +
          '<div class="marker" style="left:' + (parseFloat(leftPct) + parseFloat(widthPct)) + '%;">' + end.toFixed(1) + 's</div>' +
        '</div>' +
      '</div>' +

      '<div class="meta-grid">' +
        '<span class="key">Video</span><span class="val">' + json.video_name + '</span>' +
        '<span class="key">Duration</span><span class="val">' + dur.toFixed(2) + 's</span>' +
        '<span class="key">Dataset</span><span class="val"><span class="tag">' + json.dataset + '</span></span>' +
        '<span class="key">Query</span><span class="val">&ldquo;' + json.query + '&rdquo;</span>' +
        '<span class="key">Method</span><span class="val"><span class="tag">' + json.selection + '</span></span>' +
      '</div>' +
    '</div>';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
