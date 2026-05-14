"""
Compare extracted I3D features (.npy) against existing HDF5 features.
Usage:
    python compare_features.py --extracted /path/to/video.npy --video_id v_xxx --dataset activitynet
    python compare_features.py --extracted /path/to/video.npy --video_id 3MSZA --dataset charades
"""
import argparse
import h5py
import numpy as np


def load_hdf5_feature(hdf5_path: str, video_id: str, dataset: str):
    """Load pre-extracted feature from HDF5 for a given video."""
    with h5py.File(hdf5_path, "r") as f:
        if dataset == "activitynet":
            return np.asarray(f[video_id]["c3d_features"]).astype(np.float32)
        else:
            return np.asarray(f[video_id]).astype(np.float32)


def sample_frames(features: np.ndarray, target_frames: int = 200):
    """Uniform sampling to target frames (matching BaseDataset._sample_frame_features)."""
    num = len(features)
    keep_idx = np.arange(0, target_frames + 1) / target_frames * num
    keep_idx = np.round(keep_idx).astype(np.int64)
    keep_idx[keep_idx >= num] = num - 1
    sampled = []
    for j in range(target_frames):
        s, e = keep_idx[j], keep_idx[j + 1]
        if s == e:
            sampled.append(features[s])
        else:
            sampled.append(features[s:e].mean(axis=0))
    return np.stack(sampled, 0)


def main():
    parser = argparse.ArgumentParser(description="Compare extracted .npy features with HDF5 features")
    parser.add_argument("--extracted", type=str, required=True, help="Path to extracted .npy file")
    parser.add_argument("--video_id", type=str, required=True, help="Video ID in HDF5")
    parser.add_argument("--dataset", type=str, required=True, choices=["activitynet", "charades"])
    args = parser.parse_args()

    # Paths to HDF5 files
    HDF5_PATHS = {
        "activitynet": "/data/chenyuan/videogrounding/cpl-main/data/activitynet/sub_activitynet_v1-3.c3d.hdf5",
        "charades": "/data/chenyuan/videogrounding/cpl-main/data/charades/i3d_features.hdf5",
    }
    hdf5_path = HDF5_PATHS[args.dataset]

    # Load extracted feature
    extracted = np.load(args.extracted).astype(np.float32)
    print(f"=== Feature Comparison: {args.dataset.upper()} ===")
    print(f"Video ID: {args.video_id}")
    print(f"Extracted: {args.extracted}  shape={extracted.shape}  dtype={extracted.dtype}")

    # Load HDF5 feature
    hdf5_feat = load_hdf5_feature(hdf5_path, args.video_id, args.dataset)
    print(f"HDF5 raw:  shape={hdf5_feat.shape}  dtype={hdf5_feat.dtype}")

    # --- Dimension check ---
    print(f"\n--- Dimension Check ---")
    ext_dim = extracted.shape[-1]
    hdf5_dim = hdf5_feat.shape[-1]
    print(f"Extracted dim: {ext_dim}")
    print(f"HDF5 dim:      {hdf5_dim}")

    if ext_dim != hdf5_dim:
        print(f"\n*** DIMENSIONS DIFFER ({ext_dim} vs {hdf5_dim}) — features are NOT interchangeable ***")
        print("This means the model config must match the feature type used.")
        return

    # --- Same-dimension comparison (Charades only, both I3D 1024-dim) ---
    print(f"\n--- Temporal Length ---")
    print(f"Extracted: {extracted.shape[0]} steps")
    print(f"HDF5 raw:  {hdf5_feat.shape[0]} frames")

    # Sample both to 200 frames for comparison
    ext_sampled = sample_frames(extracted, 200)
    hdf5_sampled = sample_frames(hdf5_feat, 200)
    print(f"After sampling to 200: ext={ext_sampled.shape}  hdf5={hdf5_sampled.shape}")

    # --- Similarity metrics ---
    print(f"\n--- Similarity Metrics ---")

    # Cosine similarity per frame, then average
    dot = np.sum(ext_sampled * hdf5_sampled, axis=-1)
    norm_ext = np.linalg.norm(ext_sampled, axis=-1)
    norm_hdf5 = np.linalg.norm(hdf5_sampled, axis=-1)
    cos_sim = dot / (norm_ext * norm_hdf5 + 1e-10)
    print(f"Mean cosine similarity: {cos_sim.mean():.6f}  (range: [{cos_sim.min():.4f}, {cos_sim.max():.4f}])")

    # L2 distance
    l2 = np.linalg.norm(ext_sampled - hdf5_sampled, axis=-1)
    print(f"Mean L2 distance:       {l2.mean():.4f}  (range: [{l2.min():.2f}, {l2.max():.2f}])")

    # Overall correlation
    flat_ext = ext_sampled.flatten()
    flat_hdf5 = hdf5_sampled.flatten()
    corr = np.corrcoef(flat_ext, flat_hdf5)[0, 1]
    print(f"Pearson correlation:    {corr:.6f}")

    # --- Interpretation ---
    print(f"\n--- Interpretation ---")
    if cos_sim.mean() > 0.99:
        print("Features are NEARLY IDENTICAL (cosine > 0.99). Same extraction pipeline likely used.")
    elif cos_sim.mean() > 0.8:
        print("Features are SIMILAR (cosine > 0.8). Same model architecture with different weights/preprocessing.")
    elif cos_sim.mean() > 0.5:
        print("Features are SOMEWHAT SIMILAR (cosine > 0.5). Different feature extractors on same video.")
    else:
        print("Features are VERY DIFFERENT (cosine < 0.5). Likely different model architectures or inputs.")

    # Check ranges
    print(f"\n--- Value Statistics ---")
    print(f"Extracted:  mean={extracted.mean():.4f}  std={extracted.std():.4f}  min={extracted.min():.4f}  max={extracted.max():.4f}")
    print(f"HDF5:       mean={hdf5_feat.mean():.4f}  std={hdf5_feat.std():.4f}  min={hdf5_feat.min():.4f}  max={hdf5_feat.max():.4f}")


if __name__ == "__main__":
    main()
