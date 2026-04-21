"""
run_habitat_event_slam.py
=========================
Full end-to-end pipeline:

  Habitat-sim scene
      └─► RGB frames
              └─► ContrastThresholdEventSimulator  →  event frames
                      └─► EventVisionAdapter        →  feature vectors
                              └─► SSP-SLAM network  →  map + path

Can also skip straight to SLAM if you have already collected data with
``collect_habitat_data.py`` (pass ``--data-dir``).

Usage (collect fresh data then run SLAM):
    python experiments/run_habitat_event_slam.py \\
        --scene path/to/scene.glb \\
        --n-steps 2000 \\
        --save-dir data/habitat

Usage (SLAM on already-collected data):
    python experiments/run_habitat_event_slam.py \\
        --data-dir data/habitat \\
        --save-dir data/habitat

Requirements
------------
    conda install habitat-sim -c conda-forge -c aihabitat  (for --scene)
    pip install nengo nengo_spa numpy scipy
"""

import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(
    description="Habitat + event-camera SSP-SLAM pipeline.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# --- data source (pick one) ---
parser.add_argument("--scene", default=None,
                    help="Habitat scene file (.glb). If set, data is collected live.")
parser.add_argument("--data-dir", default=None,
                    help="Directory with pre-collected .npy files from "
                         "collect_habitat_data.py (or collect_3d_data.py).")

# --- collection params (used when --scene is set) ---
parser.add_argument("--n-steps", default=2000, type=int)
parser.add_argument("--n-objects", default=4, type=int)
parser.add_argument("--obs-width", default=256, type=int)
parser.add_argument("--obs-height", default=192, type=int)
parser.add_argument("--view-radius", default=3.0, type=float)
parser.add_argument("--policy", default="explore", choices=["random", "explore", "forward"])

# --- event simulation ---
parser.add_argument("--event-backend", default="numpy", choices=["numpy", "v2e"],
                    help="Event simulator backend.")
parser.add_argument("--C-pos", default=0.2, type=float, help="Positive contrast threshold")
parser.add_argument("--C-neg", default=0.2, type=float, help="Negative contrast threshold")
parser.add_argument("--dt-window", default=0.05, type=float,
                    help="Event accumulation window (seconds) for output frames.")

# --- SLAM params ---
parser.add_argument("--ssp-dim", default=97, type=int)
parser.add_argument("--n-neurons", default=1000, type=int)
parser.add_argument("--T", default=10.0, type=float,
                    help="Nengo simulation duration in seconds. "
                         "Should be >= n_steps / 30 to cover the whole trajectory.")
parser.add_argument("--seed", default=0, type=int)

# --- output ---
parser.add_argument("--save-dir", default="data/habitat")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

if args.scene is None and args.data_dir is None:
    parser.error("Provide either --scene (collect live) or --data-dir (use saved data).")

os.makedirs(args.save_dir, exist_ok=True)

# ─── Step 1: Data source ──────────────────────────────────────────────────────

if args.scene is not None:
    print("=== Step 1: Collecting Habitat data ===")
    try:
        from sspslam.environments.habitat_env import HabitatSLAMEnv, collect_episode
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    env = HabitatSLAMEnv(
        scene_path=args.scene,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        n_objects=args.n_objects,
        view_radius=args.view_radius,
        seed=args.seed,
    )
    t0 = time.time()
    data = collect_episode(env, n_steps=args.n_steps, policy=args.policy, seed=args.seed)
    print(f"  Collected {args.n_steps} steps in {time.time()-t0:.1f}s")
    env.close()

    rgb_frames = data["rgb_frames"]
    path = data["path"]
    velocities = data["velocities"]
    vec_to_landmarks = data["vec_to_landmarks"]
    landmark_positions = data["landmark_positions"]
    landmark_labels = data["landmark_labels"]

else:
    print("=== Step 1: Loading pre-collected data ===")
    d = args.data_dir
    path = np.load(os.path.join(d, "path.npy"))
    velocities = np.load(os.path.join(d, "velocities.npy"))
    vec_to_landmarks = np.load(os.path.join(d, "vec_to_landmarks.npy"))
    landmark_positions = np.load(os.path.join(d, "landmark_positions.npy"))

    import json
    meta_path = os.path.join(d, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        landmark_labels = meta.get("landmark_labels", [f"obj_{i}" for i in range(landmark_positions.shape[0])])
    else:
        landmark_labels = [f"obj_{i}" for i in range(landmark_positions.shape[0])]

    # Try to load rgb frames
    rgb_subset_path = os.path.join(d, "rgb_frames_subset.npy")
    if os.path.exists(rgb_subset_path):
        rgb_frames = np.load(rgb_subset_path)
        print(f"  Loaded RGB subset: {rgb_frames.shape}")
    else:
        rgb_frames = None
        print("  No RGB frames found — event simulation will be skipped.")

n_steps = len(path)
n_objects = landmark_positions.shape[0]
print(f"  Steps: {n_steps}, Objects: {n_objects}")
print(f"  Path range: x=[{path[:,0].min():.2f}, {path[:,0].max():.2f}], "
      f"y=[{path[:,1].min():.2f}, {path[:,1].max():.2f}]")

# ─── Step 2: Event simulation ─────────────────────────────────────────────────

feature_vectors = None

if rgb_frames is not None:
    print("\n=== Step 2: Event camera simulation ===")
    from sspslam.perception.event_simulator import make_event_simulator

    ev_sim = make_event_simulator(
        backend=args.event_backend,
        C_pos=args.C_pos,
        C_neg=args.C_neg,
        dt=1 / 30,
    )
    t0 = time.time()
    event_frames, event_times, raw_events = ev_sim.simulate(
        rgb_frames, dt_window=args.dt_window
    )
    print(f"  Event frames: {event_frames.shape}, raw events: {len(raw_events)}, "
          f"({time.time()-t0:.1f}s)")

    # Save events for reuse
    np.save(os.path.join(args.save_dir, "event_frames.npy"), event_frames)
    np.save(os.path.join(args.save_dir, "event_times.npy"), event_times)
    np.save(os.path.join(args.save_dir, "raw_events.npy"), raw_events)

    # ─── Step 3: Feature extraction from event frames ─────────────────────────
    print("\n=== Step 3: Feature extraction from event frames ===")
    from sspslam.perception.image_feature_encoder import extract_hog

    # Compute HOG on each event frame, then align with trajectory timesteps
    all_feats = np.array([extract_hog(f) for f in event_frames])
    feat_dim = all_feats.shape[1]
    print(f"  HOG feature dim: {feat_dim}")

    # Align event frames to trajectory steps (nearest-neighbour by time)
    traj_times = np.arange(n_steps) / 30.0
    frame_indices = np.searchsorted(event_times, traj_times).clip(0, len(all_feats) - 1)
    frame_feats_aligned = all_feats[frame_indices]          # (n_steps, feat_dim)

    frame_norms = np.linalg.norm(frame_feats_aligned, axis=1, keepdims=True)
    frame_feats_norm = frame_feats_aligned / np.maximum(frame_norms, 1e-8)

    feature_vectors = np.zeros((n_steps, n_objects, feat_dim), dtype=np.float32)
    for j in range(n_objects):
        rng_j = np.random.default_rng(seed=j)
        obj_id = rng_j.standard_normal(feat_dim).astype(np.float32)
        obj_id /= np.linalg.norm(obj_id)
        feature_vectors[:, j, :] = 0.5 * frame_feats_norm + 0.5 * obj_id

    np.save(os.path.join(args.save_dir, "feature_vectors.npy"), feature_vectors)
    print(f"  Feature vectors: {feature_vectors.shape}")

else:
    # Fallback: load pre-existing feature vectors
    fv_path = os.path.join(args.data_dir or args.save_dir, "feature_vectors.npy")
    if os.path.exists(fv_path):
        feature_vectors = np.load(fv_path)
        feat_dim = feature_vectors.shape[2]
        print(f"\n=== Steps 2-3 skipped: loaded features {feature_vectors.shape} ===")
    else:
        print("[ERROR] No RGB frames and no feature_vectors.npy found. Cannot run SLAM.")
        sys.exit(1)

# ─── Step 4: SSP-SLAM ─────────────────────────────────────────────────────────
print("\n=== Step 4: SSP-SLAM ===")

import nengo
import nengo_spa as spa
from sspslam.networks import SLAMNetwork
from sspslam.sspspace import HexagonalSSPSpace
from sspslam.perception.image_feature_encoder import ImageFeatureEncoder

ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=args.ssp_dim, seed=args.seed)
ssp_dim = ssp_space.ssp_dim
print(f"  SSP dim: {ssp_dim}")

# Encode landmark identity SPs from feature vectors
encoder = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=args.seed)
landmark_sps = np.array([
    encoder.encode(feature_vectors[:, j, :].mean(axis=0)).squeeze()
    for j in range(n_objects)
])  # (n_objects, ssp_dim)

# Encode true landmark locations as SSPs
landmark_ssps_true = np.array([
    ssp_space.encode(landmark_positions[j]).squeeze()
    for j in range(n_objects)
])  # (n_objects, ssp_dim)

# Build SLAM input functions
dt_nengo = 1e-3
T_sim = args.T
n_sim_steps = int(T_sim / dt_nengo)

traj_indices = (np.arange(n_sim_steps) / n_sim_steps * n_steps).astype(int).clip(0, n_steps - 1)

def velocity_fn(t):
    idx = min(int(t / dt_nengo), n_steps - 1)
    return velocities[idx]

def landmark_id_fn(t):
    idx = traj_indices[min(int(t / dt_nengo), n_sim_steps - 1)]
    # Use the closest visible object at this timestep
    vecs = vec_to_landmarks[idx]  # (n_objects, 2)
    dists = np.linalg.norm(vecs, axis=1)
    j = int(np.argmin(dists))
    return landmark_sps[j]

def landmark_vec_fn(t):
    idx = traj_indices[min(int(t / dt_nengo), n_sim_steps - 1)]
    vecs = vec_to_landmarks[idx]
    dists = np.linalg.norm(vecs, axis=1)
    j = int(np.argmin(dists))
    return vecs[j]

view_rad = float(np.linalg.norm(landmark_positions.max(axis=0) -
                                landmark_positions.min(axis=0))) / 2

print(f"  Simulating {T_sim}s ({n_sim_steps} steps) ...")
t0 = time.time()

with nengo.Network(seed=args.seed) as model:
    slam_net = SLAMNetwork(
        path_origin=path[0],
        scale=view_rad,
        ssp_space=ssp_space,
        n_landmarks=n_objects,
        n_neurons=args.n_neurons,
        dims=ssp_dim,
        voja_learning_rate=5e-4,
        pes_learning_rate=1e-3,
        seed=args.seed,
    )

    vel_node = nengo.Node(velocity_fn)
    id_node = nengo.Node(landmark_id_fn)
    vec_node = nengo.Node(landmark_vec_fn)

    nengo.Connection(vel_node, slam_net.velocity_input, synapse=None)
    nengo.Connection(id_node, slam_net.landmark_id_input, synapse=None)
    nengo.Connection(vec_node, slam_net.landmark_vec_input, synapse=None)

    p_pos = nengo.Probe(slam_net.position, synapse=0.05)

with nengo.Simulator(model, dt=dt_nengo, progress_bar=True) as sim:
    sim.run(T_sim)

elapsed = time.time() - t0
print(f"  Simulation done in {elapsed:.1f}s")

slam_out = sim.data[p_pos]
ts = sim.trange()

# Decode estimated path
slam_path = np.array([
    ssp_space.decode(slam_out[i]).squeeze()
    for i in range(0, len(slam_out), max(1, len(slam_out) // n_steps))
])[:n_steps]

# Evaluate
path_clipped = path[:len(slam_path)]
errors = np.linalg.norm(slam_path - path_clipped, axis=1)
print(f"  Mean position error: {errors.mean():.4f}  (normalised coords)")

# ─── Save results ─────────────────────────────────────────────────────────────
out_path = os.path.join(args.save_dir,
    f"slam_habitat_sspdim_{ssp_dim}_featdim_{feat_dim}_T_{int(T_sim)}_seed_{args.seed}.npz"
)
np.savez(
    out_path,
    ts=ts,
    path=path,
    slam_path=slam_path,
    slam_error=errors,
    landmark_positions=landmark_positions,
    landmark_labels=np.array(landmark_labels),
    slam_out=slam_out,
)
print(f"  Results saved to {out_path}")

# ─── Optional plot ────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(path[:, 0], path[:, 1], "k-", lw=0.8, alpha=0.6, label="True path")
    ax.plot(slam_path[:, 0], slam_path[:, 1], "b--", lw=0.8, alpha=0.6, label="SLAM estimate")
    for pos, lbl in zip(landmark_positions, landmark_labels):
        ax.scatter(pos[0], pos[1], s=80, marker="*", zorder=5)
        ax.annotate(lbl, pos, fontsize=7, ha="center", va="bottom")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Path: true vs SLAM estimate")

    axes[1].plot(errors)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Position error (normalised)")
    axes[1].set_title(f"SLAM error  (mean={errors.mean():.3f})")

    plt.tight_layout()
    out_png = os.path.join(args.save_dir, "habitat_slam_result.png")
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Plot saved to {out_png}")
