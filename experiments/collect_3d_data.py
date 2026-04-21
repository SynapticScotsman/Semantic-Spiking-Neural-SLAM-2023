"""
collect_3d_data.py
==================
Runs the SLAMRoom 3D environment, collects trajectory and perception data,
and saves everything as .npy files that can be fed directly into
run_slam_features.py or the SLAM dashboard.

Usage:
    python experiments/collect_3d_data.py --n-steps 2000 --policy explore --save-dir data/3d
    python experiments/collect_3d_data.py --n-steps 5000 --feat-method sift --save-dir data/3d
"""

import numpy as np
import argparse
import os
import sys
import json
import time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sspslam.environments import SLAMRoom, collect_episode

parser = argparse.ArgumentParser(
    description="Collect 3D environment data for SSP-SLAM.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--n-objects", default=4, type=int, help="Number of objects (max 6)")
parser.add_argument("--n-steps", default=2000, type=int, help="Steps to simulate")
parser.add_argument("--room-size", default=12.0, type=float, help="Room side length")
parser.add_argument("--view-radius", default=3.0, type=float, help="Object detection radius")
parser.add_argument("--policy", default="explore", choices=["random", "explore"],
                    help="Agent movement policy")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--obs-width", default=160, type=int)
parser.add_argument("--obs-height", default=120, type=int)

parser.add_argument("--feat-method", default="hog", choices=["sift", "hog", "orb", "clip", "none"],
                    help="Feature extraction method to apply to RGB frames")
parser.add_argument("--save-dir", default="data/3d", help="Output directory")
parser.add_argument("--plot", action="store_true", help="Show trajectory plot")

args = parser.parse_args()

# ─── Create environment ───
print(f"Creating SLAMRoom with {args.n_objects} objects, room_size={args.room_size}")
env = SLAMRoom(
    n_objects=args.n_objects,
    room_size=args.room_size,
    view_radius=args.view_radius,
    obs_width=args.obs_width,
    obs_height=args.obs_height,
    max_episode_steps=args.n_steps + 100,
    render_mode="rgb_array",
)

# ─── Collect trajectory ───
print(f"Collecting {args.n_steps} steps with policy='{args.policy}'...")
start = time.time()
data = collect_episode(env, n_steps=args.n_steps, policy=args.policy, seed=args.seed)
elapsed = time.time() - start
print(f"Collection done in {elapsed:.1f}s")

path = data["path"]               # (T, 2)
velocities = data["velocities"]   # (T, 2)
rgb_frames = data["rgb_frames"]   # (T, H, W, 3)
vec_to_landmarks = data["vec_to_landmarks"]  # (T, n_objects, 2)
landmark_positions = data["landmark_positions"]  # (n_objects, 2)

print(f"  Path shape: {path.shape}")
print(f"  Velocities shape: {velocities.shape}")
print(f"  RGB frames shape: {rgb_frames.shape}")
print(f"  Vec to landmarks shape: {vec_to_landmarks.shape}")
print(f"  Landmark positions: {landmark_positions}")

# ─── Extract visual features ───
feature_vectors = None
if args.feat_method != "none":
    try:
        if args.feat_method == "clip":
            from sspslam.perception.clip_encoder import CLIPFeatureExtractor
            # Get feature dim from CLIP model
            clip_extractor = CLIPFeatureExtractor()
            feat_dim = clip_extractor.feat_dim
            print(f"\nExtracting CLIP features (dim={feat_dim}) from {len(rgb_frames)} frames...")
            from PIL import Image
            
            # Per-object features: use CLIP text encoding of object label as a stable
            # per-object identity, blended with the frame's image features.
            # This gives each object a distinct appearance SP while remaining grounded
            # in visual context. Pure frame features would make all objects identical.
            feature_vectors = np.zeros((len(rgb_frames), args.n_objects, feat_dim), dtype=np.float32)
            frame_features = clip_extractor.encode_image([Image.fromarray(f) for f in rgb_frames])
            label_features = clip_extractor.encode_text(data["landmark_labels"])  # (n_objects, feat_dim)

            for j in range(args.n_objects):
                # 50/50 blend of frame appearance and object-specific label embedding
                feature_vectors[:, j, :] = 0.5 * frame_features + 0.5 * label_features[j]
                
        else:
            from sspslam.perception.image_feature_encoder import extract_sift, extract_hog, extract_orb

            EXTRACT_FN = {
                "sift": extract_sift,
                "hog": extract_hog,
                "orb": extract_orb,
            }
            extract = EXTRACT_FN[args.feat_method]

            # Get feature dim from a test frame
            test_feat = extract(rgb_frames[0])
            feat_dim = test_feat.shape[0]
            print(f"\nExtracting {args.feat_method.upper()} features (dim={feat_dim}) from {len(rgb_frames)} frames...")

            # Per-object features: frame-level descriptor plus a deterministic per-object
            # perturbation so that each object maps to a distinct SP. Without bounding
            # boxes we cannot crop per-object regions, so the perturbation is the
            # lightweight stand-in for object identity.
            feature_vectors = np.zeros((len(rgb_frames), args.n_objects, feat_dim), dtype=np.float32)
            frame_features = np.array([extract(frame) for frame in rgb_frames])  # (T, feat_dim)
            # Normalize each frame feature to unit length before blending, so the
            # per-object identity vector contributes a meaningful fraction regardless
            # of the raw descriptor magnitude (which varies by method).
            frame_norms = np.linalg.norm(frame_features, axis=1, keepdims=True)
            frame_features_norm = frame_features / np.maximum(frame_norms, 1e-8)

            for j in range(args.n_objects):
                rng = np.random.default_rng(seed=j)
                obj_identity = rng.standard_normal(feat_dim).astype(np.float32)
                obj_identity /= np.linalg.norm(obj_identity)
                # 50/50 blend of frame appearance and stable per-object identity vector
                feature_vectors[:, j, :] = 0.5 * frame_features_norm + 0.5 * obj_identity

        print(f"  Feature vectors shape: {feature_vectors.shape}")

    except ImportError as e:
        print(f"[WARN] Could not extract features ({e}). Saving without feature vectors.")
        feature_vectors = None

# ─── Save data ───
os.makedirs(args.save_dir, exist_ok=True)

np.save(os.path.join(args.save_dir, "path.npy"), path)
np.save(os.path.join(args.save_dir, "velocities.npy"), velocities)
np.save(os.path.join(args.save_dir, "vec_to_landmarks.npy"), vec_to_landmarks)
np.save(os.path.join(args.save_dir, "landmark_positions.npy"), landmark_positions)

if feature_vectors is not None:
    np.save(os.path.join(args.save_dir, "feature_vectors.npy"), feature_vectors)

# Save a subset of frames (every 100th) for visualization
subset_stride = max(1, len(rgb_frames) // 10)
np.save(os.path.join(args.save_dir, "rgb_frames_subset.npy"), rgb_frames[::subset_stride])

# Metadata
metadata = {
    "n_objects": args.n_objects,
    "n_steps": args.n_steps,
    "room_size": args.room_size,
    "view_radius_world": args.view_radius,
    "view_radius_normalized": args.view_radius / args.room_size * 2,
    "policy": args.policy,
    "seed": args.seed,
    "obs_resolution": [args.obs_height, args.obs_width],
    "feat_method": args.feat_method,
    "feat_dim": int(feature_vectors.shape[2]) if feature_vectors is not None else None,
    "landmark_labels": data["landmark_labels"],
}
with open(os.path.join(args.save_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nData saved to {args.save_dir}/")
for f_name in os.listdir(args.save_dir):
    fpath = os.path.join(args.save_dir, f_name)
    size = os.path.getsize(fpath) / 1024
    print(f"  {f_name}: {size:.0f} KB")

env.close()

# ─── Optional plot ───
if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Trajectory
    axes[0].plot(path[:, 0], path[:, 1], 'k-', alpha=0.5, linewidth=0.5)
    axes[0].plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    axes[0].plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='End')
    for i, (pos, label) in enumerate(zip(landmark_positions, data["landmark_labels"])):
        axes[0].scatter(pos[0], pos[1], s=100, marker='*', zorder=5)
        axes[0].annotate(label, pos, fontsize=7, ha='center', va='bottom')
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=8)
    axes[0].set_title("Agent Trajectory")

    # Sample frames
    axes[1].imshow(rgb_frames[0])
    axes[1].set_title("First frame")
    axes[1].axis('off')

    axes[2].imshow(rgb_frames[len(rgb_frames)//2])
    axes[2].set_title("Mid frame")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "collection_summary.png"), dpi=150)
    plt.show()
    print(f"Plot saved to {args.save_dir}/collection_summary.png")
