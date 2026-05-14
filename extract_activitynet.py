"""
Extract I3D features for ActivityNet Captions videos.
Based on VSLNet-master/prepare/extract_activitynet.py.

Usage:
    python extract_activitynet.py \
        --load_model /path/to/rgb_imagenet.pt \
        --video_dir /path/to/ActivityNet_videos \
        --save_dir /path/to/features \
        --gpu_idx 0
"""
import os
import sys
import json
import cv2
import torch
import argparse
import subprocess
import numpy as np
from pathlib import Path

# Add VSLNet feature extractor to path
VSLNET_DIR = Path(__file__).resolve().parent / "VSLNet-master" / "prepare"
sys.path.insert(0, str(VSLNET_DIR))
from feature_extractor import InceptionI3d


# ── CenterCrop (inlined from VSLNet videotransforms.py) ────────────────────

class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))
        return imgs[:, i:i + th, j:j + tw, :]


# ── Helpers ────────────────────────────────────────────────────────────────

def load_frames(image_dir, video_id, start_frame, num_frames_to_load):
    """Load and preprocess a range of frames."""
    frames = []
    for x in range(start_frame, start_frame + num_frames_to_load):
        img_path = os.path.join(image_dir, f"{video_id}-{str(x).zfill(6)}.jpg")
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]  # BGR → RGB
        h, w, _ = img.shape
        # resize so min dimension >= 224
        scale = 1 + (224.0 - min(w, h)) / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        img = (img / 255.0) * 2 - 1  # normalize to [-1, 1]
        frames.append(img)
    return frames


def extract_window_features(img_tensor, model, strides):
    """Sliding window I3D feature extraction."""
    b, c, t, h, w = img_tensor.shape
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

parser = argparse.ArgumentParser(description="Extract I3D features for ActivityNet videos")
parser.add_argument("--gpu_idx", type=str, default="0")
parser.add_argument("--load_model", type=str, required=True, help="Pretrained I3D model (rgb_imagenet.pt)")
parser.add_argument("--video_dir", type=str, required=True, help="Directory containing ActivityNet .mp4 videos")
parser.add_argument("--save_dir", type=str, required=True, help="Output directory for .npy features")
parser.add_argument("--images_dir", type=str, default=None, help="Temp directory for extracted frames (default: save_dir/images)")
parser.add_argument("--fps", type=int, default=None, help="FPS for frame extraction (None = original FPS)")
parser.add_argument("--strides", type=int, default=16, help="Sliding window stride")
parser.add_argument("--remove_images", action="store_true", help="Delete frames after extraction")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

if not os.path.exists(args.video_dir):
    raise ValueError(f"Video directory '{args.video_dir}' does not exist")

images_dir = args.images_dir or os.path.join(args.save_dir, "images_activitynet")
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# ── Build model ────────────────────────────────────────────────────────────

print("Loading I3D model (ImageNet pretrained)...")
i3d_model = InceptionI3d(400, in_channels=3)
state_dict = torch.load(args.load_model, map_location="cpu")
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
i3d_model.load_state_dict(state_dict)
i3d_model.cuda()
i3d_model.train(False)

crop = CenterCrop(224)

# ── Collect video IDs ──────────────────────────────────────────────────────

repo_root = Path(__file__).resolve().parent
activitynet_dir = repo_root / "cpl-main" / "data" / "activitynet"
video_ids = set()
for json_name in ["train_data.json", "test_data.json", "val_data.json"]:
    json_path = activitynet_dir / json_name
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            for item in data:
                video_ids.add(item[0])
video_ids = sorted(video_ids)
print(f"Found {len(video_ids)} unique video IDs")

# ── Extract ────────────────────────────────────────────────────────────────

feature_shapes = {}
for idx, video_id in enumerate(video_ids):
    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
    image_dir = os.path.join(images_dir, video_id)
    save_path = os.path.join(args.save_dir, f"{video_id}.npy")

    print(f"[{idx+1}/{len(video_ids)}] {video_id}", flush=True)

    if os.path.exists(save_path):
        feature = np.load(save_path)
        feature_shapes[video_id] = feature.shape[0]
        print(f"  Already exists, shape={feature.shape}, skipping.\n")
        continue

    # Check video exists
    if not os.path.exists(video_path):
        print(f"  WARNING: video not found at {video_path}, skipping.\n")
        continue

    # Extract frames with ffmpeg
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        if args.fps and args.fps > 0:
            cmd = (f"ffmpeg -hide_banner -loglevel panic -i {video_path} "
                   f"-filter:v fps=fps={args.fps} {image_dir}/{video_id}-%6d.jpg")
        else:
            cmd = (f"ffmpeg -hide_banner -loglevel panic -i {video_path} "
                   f"{image_dir}/{video_id}-%6d.jpg")
        subprocess.call(cmd, shell=True)

    num_frames = len(os.listdir(image_dir))
    if num_frames == 0:
        print(f"  WARNING: no frames extracted, skipping.\n")
        continue

    # ActivityNet videos can be long; process in batches of 10000 frames
    MAX_BATCH_FRAMES = 10000
    if num_frames < MAX_BATCH_FRAMES:
        frames = load_frames(image_dir, video_id, 1, num_frames)
        frames = np.asarray(frames, dtype=np.float32)
        imgs = crop(frames)
        img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
        print(f"  Frames: {frames.shape} -> {imgs.shape} -> {tuple(img_tensor.size())}")
        features = extract_window_features(img_tensor, i3d_model, args.strides)
    else:
        all_features = []
        for start_idx in range(1, num_frames, MAX_BATCH_FRAMES):
            end_idx = min(start_idx + MAX_BATCH_FRAMES, num_frames + 1)
            cur_num = end_idx - start_idx
            if cur_num < args.strides:
                cur_num = args.strides
                start_idx = end_idx - cur_num
            frames = load_frames(image_dir, video_id, start_idx, cur_num)
            frames = np.asarray(frames, dtype=np.float32)
            imgs = crop(frames)
            img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
            print(f"  Batch [{start_idx}:{end_idx}]: {frames.shape} -> {imgs.shape}")
            feats = extract_window_features(img_tensor, i3d_model, args.strides)
            all_features.append(feats)
        features = np.concatenate(all_features, axis=0)

    np.save(save_path, features)
    feature_shapes[video_id] = features.shape[0]
    print(f"  Saved: {save_path} shape={features.shape}\n")

    if args.remove_images:
        subprocess.call(f"rm -rf {image_dir}", shell=True)

# Save shape index
shapes_path = os.path.join(args.save_dir, "feature_shapes.json")
with open(shapes_path, "w") as f:
    json.dump(feature_shapes, f)
print(f"Done. {len(feature_shapes)} videos. Shapes saved to {shapes_path}")
