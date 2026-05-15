"""
Extract I3D features for a single Charades-STA video.
Based on VSLNet-master/prepare/extract_charades.py.

Usage:
    python extract_charades.py \
        --load_model ./weights/rgb_charades.pt \
        --video_path /path/to/video.mp4 \
        --save_path /path/to/output.npy \
        --gpu_idx 0
"""
import os
import sys
import json
import cv2
import torch
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extractor import InceptionI3d


class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))
        return imgs[:, i:i + th, j:j + tw, :]


def extract_frames(video_path: str, image_dir: str, video_id: str, fps: int):
    """Extract frames from video using ffmpeg."""
    os.makedirs(image_dir, exist_ok=True)
    cmd = (f"ffmpeg -hide_banner -loglevel panic -i {video_path} "
           f"-filter:v fps=fps={fps} {image_dir}/{video_id}-%6d.jpg")
    subprocess.call(cmd, shell=True)


def load_frames(image_dir: str, video_id: str, num_frames: int):
    """Load and preprocess extracted frames."""
    frames = []
    for i in range(1, num_frames + 1):
        img_path = os.path.join(image_dir, f"{video_id}-{str(i).zfill(6)}.jpg")
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]  # BGR -> RGB
        h, w, _ = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def extract_i3d_features(frames: np.ndarray, model, strides: int):
    """Sliding-window I3D feature extraction."""
    crop = CenterCrop(224)
    imgs = crop(frames)
    img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
    print(f"  Frames: {frames.shape} -> {imgs.shape} -> {tuple(img_tensor.size())}")

    _, _, t, _, _ = img_tensor.shape
    features = []
    for start in range(0, t, strides):
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        window = img_tensor[:, :, start:end].cuda()
        with torch.no_grad():
            feat = model.extract_features(window).cpu().numpy()
        features.append(feat)
    return np.concatenate(features, axis=0)


# ── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Extract I3D features for one Charades video")
parser.add_argument("--gpu_idx", type=str, default="0")
parser.add_argument("--load_model", type=str, required=True, help="Pretrained I3D (rgb_charades.pt)")
parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 video")
parser.add_argument("--save_path", type=str, default=None, help="Output .npy path (default: same dir as video)")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--strides", type=int, default=24)
parser.add_argument("--keep_frames", action="store_true", help="Keep extracted frames after extraction")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

video_path = args.video_path
if not os.path.exists(video_path):
    raise ValueError(f"Video not found: {video_path}")

video_id = Path(video_path).stem
save_path = args.save_path or os.path.join(os.path.dirname(video_path), f"{video_id}.npy")

# ── Build model ────────────────────────────────────────────────────────────

print("Loading I3D model (Charades finetuned)...")
i3d_model = InceptionI3d(400, in_channels=3)
i3d_model.replace_logits(157)
state_dict = torch.load(args.load_model, map_location="cpu")
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
i3d_model.load_state_dict(state_dict)
i3d_model.cuda()
i3d_model.train(False)

# ── Extract ────────────────────────────────────────────────────────────────

print(f"Video: {video_path}")
print(f"Video ID: {video_id}")

with tempfile.TemporaryDirectory() as image_dir:
    print(f"Extracting frames (fps={args.fps})...")
    extract_frames(video_path, image_dir, video_id, args.fps)

    num_frames = len(os.listdir(image_dir))
    if num_frames == 0:
        raise RuntimeError("No frames extracted — check ffmpeg and video file")

    print(f"Extracted {num_frames} frames, loading...")
    frames = load_frames(image_dir, video_id, num_frames)

    print(f"Running I3D inference (stride={args.strides})...")
    features = extract_i3d_features(frames, i3d_model, args.strides)

    if args.keep_frames:
        keep_dir = os.path.join(os.path.dirname(save_path), f"{video_id}_frames")
        os.makedirs(keep_dir, exist_ok=True)
        for f in os.listdir(image_dir):
            os.rename(os.path.join(image_dir, f), os.path.join(keep_dir, f))
        print(f"Frames kept at: {keep_dir}")

os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
np.save(save_path, features)
print(f"Saved: {save_path}  shape={features.shape}")
