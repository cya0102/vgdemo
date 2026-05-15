"""
Extract I3D features for a single ActivityNet video.
Based on VSLNet-master/prepare/extract_activitynet.py.

Usage:
    python extract_activitynet.py \
        --load_model ./weights/rgb_imagenet.pt \
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


def extract_frames(video_path: str, image_dir: str, video_id: str, fps):
    """Extract frames from video using ffmpeg."""
    os.makedirs(image_dir, exist_ok=True)
    if fps and fps > 0:
        cmd = (f"ffmpeg -hide_banner -loglevel panic -i {video_path} "
               f"-filter:v fps=fps={fps} {image_dir}/{video_id}-%6d.jpg")
    else:
        cmd = (f"ffmpeg -hide_banner -loglevel panic -i {video_path} "
               f"{image_dir}/{video_id}-%6d.jpg")
    subprocess.call(cmd, shell=True)


def load_frames(image_dir: str, video_id: str, start_frame: int, num_to_load: int):
    """Load and preprocess a range of frames (supports batching for long videos)."""
    frames = []
    for x in range(start_frame, start_frame + num_to_load):
        img_path = os.path.join(image_dir, f"{video_id}-{str(x).zfill(6)}.jpg")
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]  # BGR -> RGB
        h, w, _ = img.shape
        scale = 1 + (224.0 - min(w, h)) / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        img = (img / 255.0) * 2 - 1
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

parser = argparse.ArgumentParser(description="Extract I3D features for one ActivityNet video")
parser.add_argument("--gpu_idx", type=str, default="0")
parser.add_argument("--load_model", type=str, required=True, help="Pretrained I3D (rgb_imagenet.pt)")
parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 video")
parser.add_argument("--save_path", type=str, default=None, help="Output .npy path (default: same dir as video)")
parser.add_argument("--fps", type=int, default=None, help="FPS for extraction (None = original)")
parser.add_argument("--strides", type=int, default=16)
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

print("Loading I3D model (ImageNet pretrained)...")
i3d_model = InceptionI3d(400, in_channels=3)
state_dict = torch.load(args.load_model, map_location="cpu")
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
i3d_model.load_state_dict(state_dict)
i3d_model.cuda()
i3d_model.train(False)

# ── Extract ────────────────────────────────────────────────────────────────

print(f"Video: {video_path}")
print(f"Video ID: {video_id}")

MAX_BATCH_FRAMES = 10000

with tempfile.TemporaryDirectory() as image_dir:
    print(f"Extracting frames (fps={args.fps or 'original'})...")
    extract_frames(video_path, image_dir, video_id, args.fps)

    num_frames = len(os.listdir(image_dir))
    if num_frames == 0:
        raise RuntimeError("No frames extracted — check ffmpeg and video file")

    print(f"Extracted {num_frames} frames, running I3D inference (stride={args.strides})...")

    if num_frames < MAX_BATCH_FRAMES:
        frames = load_frames(image_dir, video_id, 1, num_frames)
        features = extract_i3d_features(frames, i3d_model, args.strides)
    else:
        all_features = []
        for start_idx in range(1, num_frames, MAX_BATCH_FRAMES):
            end_idx = min(start_idx + MAX_BATCH_FRAMES, num_frames + 1)
            cur_num = end_idx - start_idx
            if cur_num < args.strides:
                cur_num = args.strides
                start_idx = end_idx - cur_num
            frames = load_frames(image_dir, video_id, start_idx, cur_num)
            feats = extract_i3d_features(frames, i3d_model, args.strides)
            all_features.append(feats)
        features = np.concatenate(all_features, axis=0)

    if args.keep_frames:
        keep_dir = os.path.join(os.path.dirname(save_path), f"{video_id}_frames")
        os.makedirs(keep_dir, exist_ok=True)
        for f in os.listdir(image_dir):
            os.rename(os.path.join(image_dir, f), os.path.join(keep_dir, f))
        print(f"Frames kept at: {keep_dir}")

os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
np.save(save_path, features)
print(f"Saved: {save_path}  shape={features.shape}")
