"""
collect_habitat_data.py
=======================
Collect trajectory and RGB frames from a Habitat-sim scene, convert frames
to synthetic DVS events, and save everything in the same format used by
``collect_3d_data.py`` so the data can be fed directly into
``run_slam_features.py`` or ``run_habitat_event_slam.py``.

Usage
-----
    python experiments/collect_habitat_data.py \\
        --scene path/to/scene.glb \\
        --n-steps 2000 \\
        --feat-method hog \\
        --save-dir data/habitat

Requirements
------------
    conda install habitat-sim -c conda-forge -c aihabitat
    pip install scikit-image          # for HOG
    pip install opencv-python         # for SIFT / ORB
    pip install torch transformers    # for CLIP
"""

import numpy as np
import argparse
import os
import sys
import json
import time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(
    description="Collect Habitat-sim data for SSP-SLAM.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--scene", required=True,
                    help="Path to .glb / .ply / .obj scene file")
parser.add_argument("--n-steps", default=2000, type=int)
parser.add_argument("--n-objects", default=4, type=int, help="Synthetic landmarks (max 6)")
parser.add_argument("--room-size", default=None, type=float,
                    help="Override room size for normalisation (metres). "
                         "Auto-detected from navmesh if omitted.")
parser.add_argument("--view-radius", default=3.0, type=float,
                    help="Landmark detection radius (metres)")
parser.add_argument("--obs-width", default=256, type=int)
parser.add_argument("--obs-height", default=192, type=int)
parser.add_argument("--sensor-height", default=1.5, type=float)
parser.add_argument("--policy", default="explore", choices=["random", "explore", "forward"])
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--feat-method", default="hog",
                    choices=["sift", "hog", "orb", "clip", "none"])
parser.add_argument("--save-dir", default="data/habitat")
parser.add_argument("--event-sim", action="store_true",
                    help="Also run event camera simulation and save event frames")
parser.add_argument("--event-backend", default="numpy", choices=["numpy", "v2e"])
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# ─── Create environment ───────────────────────────────────────────────────────
try:
    from sspslam.environments.habitat_env import HabitatSLAMEnv, collect_episode
except ImportError as e:
    print(f"[ERROR] Could not import HabitatSLAMEnv: {e}")
    print("Install habitat-sim: conda install habitat-sim -c conda-forge -c aihabitat")
    sys.exit(1)

print(f"Creating HabitatSLAMEnv from scene: {args.scene}")
env = HabitatSLAMEnv(
    scene_path=args.scene,
    sensor_height=args.sensor_height,
    obs_width=args.obs_width,
    obs_height=args.obs_height,
    n_objects=args.n_objects,
    room_size=args.room_size,
    view_radius=args.view_radius,
    enable_depth=False,
    seed=args.seed,
)
print(f"  Room size (metres): {env._room_size:.2f}")
print(f"  Landmark labels: {env.object_labels}")

# ─── Collect trajectory ───────────────────────────────────────────────────────
print(f"Collecting {args.n_steps} steps with policy='{args.policy}'...")
start = time.time()
data = collect_episode(env, n_steps=args.n_steps, policy=args.policy, seed=args.seed)
elapsed = time.time() - start
print(f"Collection done in {elapsed:.1f}s")

path = data["path"]
velocities = data["velocities"]
rgb_frames = data["rgb_frames"]
vec_to_landmarks = data["vec_to_landmarks"]
landmark_positions = data["landmark_positions"]

print(f"  Path shape:           {path.shape}")
print(f"  Velocities shape:     {velocities.shape}")
print(f"  RGB frames shape:     {rgb_frames.shape}")
print(f"  Vec to landmarks:     {vec_to_landmarks.shape}")

# ─── Event camera simulation ──────────────────────────────────────────────────
event_frames = None
event_times = None
raw_events = None

if args.event_sim:
    from sspslam.perception.event_simulator import make_event_simulator
    print(f"\nRunning event simulation (backend='{args.event_backend}')...")
    sim = make_event_simulator(backend=args.event_backend, dt=1 / 30)
    event_frames, event_times, raw_events = sim.simulate(rgb_frames, dt_window=0.05)
    print(f"  Event frames shape: {event_frames.shape}")
    print(f"  Raw events:         {len(raw_events)}")

# ─── Extract visual features ──────────────────────────────────────────────────
feature_vectors = None
feat_dim = None

if args.feat_method != "none":
    try:
        if args.feat_method == "clip":
            from sspslam.perception.clip_encoder import CLIPFeatureExtractor
            from PIL import Image
            clip_extractor = CLIPFeatureExtractor()
            feat_dim = clip_extractor.feat_dim
            print(f"\nExtracting CLIP features (dim={feat_dim})...")
            feature_vectors = np.zeros(
                (len(rgb_frames), args.n_objects, feat_dim), dtype=np.float32
            )
            frame_features = clip_extractor.encode_image(
                [Image.fromarray(f) for f in rgb_frames]
            )
            label_features = clip_extractor.encode_text(data["landmark_labels"])
            for j in range(args.n_objects):
                feature_vectors[:, j, :] = 0.5 * frame_features + 0.5 * label_features[j]
        else:
            from sspslam.perception.image_feature_encoder import (
                extract_sift, extract_hog, extract_orb
            )
            EXTRACT_FN = {"sift": extract_sift, "hog": extract_hog, "orb": extract_orb}
            extract = EXTRACT_FN[args.feat_method]

            test_feat = extract(rgb_frames[0])
            feat_dim = test_feat.shape[0]
            print(f"\nExtracting {args.feat_method.upper()} features (dim={feat_dim})...")

            feature_vectors = np.zeros(
                (len(rgb_frames), args.n_objects, feat_dim), dtype=np.float32
            )
            frame_features = np.array([extract(f) for f in rgb_frames])
            frame_norms = np.linalg.norm(frame_features, axis=1, keepdims=True)
            frame_features_norm = frame_features / np.maximum(frame_norms, 1e-8)

            for j in range(args.n_objects):
                rng_j = np.random.default_rng(seed=j)
                obj_identity = rng_j.standard_normal(feat_dim).astype(np.float32)
                obj_identity /= np.linalg.norm(obj_identity)
                feature_vectors[:, j, :] = 0.5 * frame_features_norm + 0.5 * obj_identity

        print(f"  Feature vectors shape: {feature_vectors.shape}")

    except ImportError as e:
        print(f"[WARN] Could not extract features ({e}). Saving without feature vectors.")
        feature_vectors = None

# ─── Save data ────────────────────────────────────────────────────────────────
os.makedirs(args.save_dir, exist_ok=True)

np.save(os.path.join(args.save_dir, "path.npy"), path)
np.save(os.path.join(args.save_dir, "velocities.npy"), velocities)
np.save(os.path.join(args.save_dir, "vec_to_landmarks.npy"), vec_to_landmarks)
np.save(os.path.join(args.save_dir, "landmark_positions.npy"), landmark_positions)
np.save(os.path.join(args.save_dir, "rgb_frames_subset.npy"), rgb_frames[::100])

if feature_vectors is not None:
    np.save(os.path.join(args.save_dir, "feature_vectors.npy"), feature_vectors)

if event_frames is not None:
    np.save(os.path.join(args.save_dir, "event_frames.npy"), event_frames)
    np.save(os.path.join(args.save_dir, "event_times.npy"), event_times)
    np.save(os.path.join(args.save_dir, "raw_events.npy"), raw_events)

metadata = {
    "scene": args.scene,
    "n_objects": args.n_objects,
    "n_steps": args.n_steps,
    "room_size_metres": env._room_size,
    "view_radius": args.view_radius,
    "obs_resolution": [args.obs_height, args.obs_width],
    "policy": args.policy,
    "seed": args.seed,
    "feat_method": args.feat_method,
    "feat_dim": int(feat_dim) if feat_dim is not None else None,
    "event_sim": args.event_sim,
    "event_backend": args.event_backend if args.event_sim else None,
    "landmark_labels": data["landmark_labels"],
}
with open(os.path.join(args.save_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nData saved to {args.save_dir}/")
for f_name in sorted(os.listdir(args.save_dir)):
    fpath = os.path.join(args.save_dir, f_name)
    size = os.path.getsize(fpath) / 1024
    print(f"  {f_name}: {size:.0f} KB")

env.close()

# ─── Optional plot ────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib.pyplot as plt

    n_panel = 4 if event_frames is not None else 3
    fig, axes = plt.subplots(1, n_panel, figsize=(5 * n_panel, 4))

    axes[0].plot(path[:, 0], path[:, 1], "k-", alpha=0.5, linewidth=0.5)
    axes[0].plot(path[0, 0], path[0, 1], "go", ms=8, label="Start")
    axes[0].plot(path[-1, 0], path[-1, 1], "ro", ms=8, label="End")
    for pos, label in zip(landmark_positions, data["landmark_labels"]):
        axes[0].scatter(pos[0], pos[1], s=80, marker="*", zorder=5)
        axes[0].annotate(label, pos, fontsize=7, ha="center", va="bottom")
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Agent Trajectory")

    axes[1].imshow(rgb_frames[0])
    axes[1].set_title("First RGB frame")
    axes[1].axis("off")

    axes[2].imshow(rgb_frames[len(rgb_frames) // 2])
    axes[2].set_title("Mid RGB frame")
    axes[2].axis("off")

    if event_frames is not None:
        axes[3].imshow(event_frames[len(event_frames) // 2], cmap="gray")
        axes[3].set_title("Mid event frame")
        axes[3].axis("off")

    plt.tight_layout()
    out_png = os.path.join(args.save_dir, "collection_summary.png")
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Plot saved to {out_png}")
