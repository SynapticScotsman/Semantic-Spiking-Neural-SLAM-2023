"""
run_event_slam.py
=================
SSP-SLAM driven by a DVS (event-camera) perception pipeline.

Two modes
---------
1. **Real 3D data** (default, no extra installs needed):
   Loads ``data/3d/rgb_frames_subset.npy``, simulates DVS events with
   ``ContrastThresholdEventSimulator``, extracts ORB features via
   ``EventVisionAdapter``, and runs SSP-SLAM.

2. **UZH-RPG text format** (``--events-txt``):
   Loads a real event-camera dataset (``events.txt``, ``groundtruth.txt``)
   in the format used by the UZH-RPG dataset.

Usage
-----
    # Mode 1 — built-in 3D data (no extra packages)
    python experiments/run_event_slam.py

    # Override which 3D data dir to use
    python experiments/run_event_slam.py --data-dir data/3d --T 10 --plot

    # Mode 2 — real UZH-RPG event file
    python experiments/run_event_slam.py \\
        --events-txt path/to/events.txt \\
        --gt-txt     path/to/groundtruth.txt \\
        --T 5 --plot
"""

import numpy as np
import nengo
import sys, os, time, argparse

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sspslam
from sspslam.networks import SLAMNetwork, get_slam_input_functions_from_features
from sspslam.perception import ImageFeatureEncoder
from sspslam.perception.event_simulator import ContrastThresholdEventSimulator
from sspslam.perception.event_dataset_parser import (
    load_events_txt, load_groundtruth_txt, accumulate_events_to_frames
)
from sspslam.perception.event_vision_adapter import EventVisionAdapter

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="SSP-SLAM with event-camera input.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--data-dir", default=None,
                    help="Path to data/3d/ folder with rgb_frames_subset.npy etc. "
                         "Defaults to <project_root>/data/3d.")
parser.add_argument("--events-txt", default=None,
                    help="UZH-RPG events.txt file (real event camera data).")
parser.add_argument("--gt-txt", default=None,
                    help="UZH-RPG groundtruth.txt file (needed with --events-txt).")
parser.add_argument("--T", default=10.0, type=float,
                    help="Nengo simulation duration (seconds).")
parser.add_argument("--ssp-dim", default=97, type=int)
parser.add_argument("--pi-neurons", default=300, type=int)
parser.add_argument("--mem-neurons", default=500, type=int)
parser.add_argument("--max-landmarks", default=8, type=int)
parser.add_argument("--dt-window", default=0.05, type=float,
                    help="Event accumulation window (seconds).")
parser.add_argument("--C-pos", default=0.15, type=float)
parser.add_argument("--C-neg", default=0.15, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--save-dir", default=None,
                    help="Directory to save results. Defaults to --data-dir.")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if args.data_dir is None:
    args.data_dir = os.path.join(PROJECT_ROOT, "data", "3d")
if args.save_dir is None:
    args.save_dir = args.data_dir

dt_nengo = 0.001

# ---------------------------------------------------------------------------
# Step 1: Get event frames + ground-truth path
# ---------------------------------------------------------------------------

if args.events_txt is not None:
    # ── Real UZH-RPG dataset ──────────────────────────────────────────────
    print(f"Loading events from: {args.events_txt}")
    t_ev, x_ev, y_ev, p_ev = load_events_txt(args.events_txt)

    if args.gt_txt:
        gt_t, gt_pos = load_groundtruth_txt(args.gt_txt)
        # Extract x,z ground plane
        path_raw = gt_pos[:, [0, 2]]
    else:
        # Fake linear path over the event duration
        T_data = t_ev[-1] - t_ev[0]
        n_gt = 200
        path_raw = np.column_stack([np.linspace(0, 1, n_gt), np.zeros(n_gt)])
        gt_t = np.linspace(t_ev[0], t_ev[-1], n_gt)
        print("[WARN] No groundtruth.txt — using synthetic linear path.")

    # Determine sensor size from data
    H, W = int(y_ev.max()) + 1, int(x_ev.max()) + 1
    print(f"Sensor size: {H}×{W}")

    frames, frame_times = accumulate_events_to_frames(
        t_ev, x_ev, y_ev, p_ev, args.dt_window, W, H
    )

    # Interpolate path to frame timestamps
    path_frames = np.stack([
        np.interp(frame_times, gt_t, path_raw[:, d])
        for d in range(path_raw.shape[1])
    ], axis=1)

else:
    # ── Simulate events from the pre-collected RGB frames ─────────────────
    rgb_path = os.path.join(args.data_dir, "rgb_frames_subset.npy")
    vel_path = os.path.join(args.data_dir, "velocities.npy")
    pos_path = os.path.join(args.data_dir, "path.npy")

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(
            f"No rgb_frames_subset.npy in {args.data_dir}.\n"
            "Collect data with: python experiments/collect_3d_data.py"
        )

    print(f"Loading RGB frames from {rgb_path} ...")
    rgb_frames = np.load(rgb_path)
    path_raw   = np.load(pos_path) if os.path.exists(pos_path) else None
    vel_raw    = np.load(vel_path) if os.path.exists(vel_path) else None
    print(f"  RGB frames: {rgb_frames.shape}")

    print(f"Simulating DVS events (C_pos={args.C_pos}, dt_window={args.dt_window}s)...")
    ev_sim = ContrastThresholdEventSimulator(
        C_pos=args.C_pos, C_neg=args.C_neg,
        dt=1/30, noise_rate=0.05, seed=args.seed
    )
    frames, frame_times, raw_events = ev_sim.simulate(rgb_frames, dt_window=args.dt_window)
    print(f"  Event frames: {frames.shape}   raw events: {len(raw_events):,}")

    # Build path aligned to event frame timestamps
    n_frames = len(frames)
    if path_raw is not None:
        t_src = np.linspace(0, 1, len(path_raw))
        t_dst = np.linspace(0, 1, n_frames)
        path_frames = np.stack([
            np.interp(t_dst, t_src, path_raw[:, d])
            for d in range(path_raw.shape[1])
        ], axis=1)
    else:
        path_frames = np.column_stack([
            np.linspace(-0.8, 0.8, n_frames),
            np.linspace(-0.8, 0.8, n_frames)
        ])
        print("[WARN] No path.npy found — using synthetic path.")

H, W = frames.shape[1], frames.shape[2]
n_frames = len(frames)
print(f"\nEvent frames: {frames.shape}  |  Path: {path_frames.shape}")

# ---------------------------------------------------------------------------
# Step 2: Extract visual features from event frames via ORB optical flow
# ---------------------------------------------------------------------------

print("\nRunning EventVisionAdapter (ORB optical flow across event frames)...")
try:
    adapter = EventVisionAdapter(
        max_landmarks=args.max_landmarks,
        view_radius=2.0,
        backend='orb',
    )
    vel_data, vec_data, feat_data = adapter.process_dataset(
        frames, agent_velocities=None
    )
    print(f"  Feature tracks: {feat_data.shape}  (frames × landmarks × feat_dim)")
    n_landmarks = feat_data.shape[1]
    feat_dim    = feat_data.shape[2]
    view_rad    = 2.0
    using_orb   = True

except ImportError:
    print("[WARN] opencv-python not installed — using HOG features on event frames instead.")
    from sspslam.perception.image_feature_encoder import extract_hog

    raw_feats = np.array([extract_hog(f) for f in frames])   # (n_frames, feat_dim)
    feat_dim  = raw_feats.shape[1]
    n_landmarks = min(args.max_landmarks, 4)

    # Normalise and give each "landmark" a distinct identity
    norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
    raw_norm = raw_feats / np.maximum(norms, 1e-8)
    feat_data = np.zeros((n_frames, n_landmarks, feat_dim), dtype=np.float32)
    for j in range(n_landmarks):
        rng_j = np.random.default_rng(seed=j)
        ident = rng_j.standard_normal(feat_dim).astype(np.float32)
        ident /= np.linalg.norm(ident)
        feat_data[:, j, :] = 0.5 * raw_norm + 0.5 * ident

    # Fake relative vectors (circular arrangement)
    angles = np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False)
    lm_offsets = 0.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    vec_data = lm_offsets[np.newaxis, :, :] * np.ones((n_frames, 1, 1))
    vel_data = np.diff(path_frames, axis=0, prepend=path_frames[[0]])
    view_rad = 0.8
    using_orb = False

print(f"  n_landmarks={n_landmarks}  feat_dim={feat_dim}  view_rad={view_rad}")

# ---------------------------------------------------------------------------
# Step 3: Stretch arrays to Nengo timestep
# ---------------------------------------------------------------------------

def stretch(arr, n_out):
    n_in = arr.shape[0]
    t0 = np.linspace(0, 1, n_in)
    t1 = np.linspace(0, 1, n_out)
    if arr.ndim == 2:
        return np.stack([np.interp(t1, t0, arr[:, d]) for d in range(arr.shape[1])], axis=1)
    elif arr.ndim == 3:
        out = np.zeros((n_out, arr.shape[1], arr.shape[2]))
        for i in range(arr.shape[1]):
            for j in range(arr.shape[2]):
                out[:, i, j] = np.interp(t1, t0, arr[:, i, j])
        return out
    return arr

T_sim  = args.T
n_sim  = int(T_sim / dt_nengo)

path_sim = stretch(path_frames, n_sim)
vel_sim  = stretch(vel_data,    n_sim)
vec_sim  = stretch(vec_data,    n_sim)
feat_sim = stretch(feat_data,   n_sim)

# Normalise velocity magnitude so the PI network is in a reasonable range
vel_scale = np.abs(vel_sim).max()
vel_sim_scaled = vel_sim / (vel_scale + 1e-8)

print(f"\nSimulation: T={T_sim}s  steps={n_sim:,}")

# ---------------------------------------------------------------------------
# Step 4: SSP-SLAM
# ---------------------------------------------------------------------------

domain_dim = 2
bounds = np.tile([-1, 1], (domain_dim, 1)).astype(float)

ssp_space = sspslam.HexagonalSSPSpace(
    domain_dim, ssp_dim=args.ssp_dim,
    domain_bounds=bounds, seed=args.seed
)
ssp_dim = ssp_space.ssp_dim
print(f"SSP space: ssp_dim={ssp_dim}")

encoder = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=args.seed)

real_ssp = ssp_space.encode(path_sim)

(velocity_func, vel_scaling_factor,
 is_landmark_in_view, landmark_id_func,
 landmark_sp_func, landmark_vec_func,
 landmark_vecssp_func) = get_slam_input_functions_from_features(
    ssp_space, encoder,
    vel_sim_scaled, vec_sim, feat_sim,
    view_rad, dt=dt_nengo
)

lm_sps    = encoder.encode(feat_sim.mean(axis=0))
lm_space  = sspslam.SPSpace(n_landmarks, ssp_dim, seed=args.seed, vectors=lm_sps)

model = nengo.Network(seed=args.seed)
with model:
    vel_node  = nengo.Node(velocity_func)
    init_node = nengo.Node(
        lambda t: real_ssp[int(t / dt_nengo)] if t < 0.05 else np.zeros(ssp_dim)
    )
    vec_node  = nengo.Node(landmark_vecssp_func)
    id_node   = nengo.Node(landmark_sp_func)
    view_node = nengo.Node(is_landmark_in_view)

    slam = SLAMNetwork(
        ssp_space, lm_space, view_rad, n_landmarks,
        args.pi_neurons, args.mem_neurons, 70,
        tau_pi=0.05,
        update_thres=0.2,
        vel_scaling_factor=vel_scaling_factor,
        shift_rate=0.1,
        voja_learning_rate=1e-4,
        pes_learning_rate=5e-3,
        intercept=0.1,
        clean_up_method='grid',
        voja=True,
        seed=args.seed,
    )

    nengo.Connection(vel_node,  slam.velocity_input,      synapse=None)
    nengo.Connection(init_node, slam.pathintegrator.input, synapse=None)
    nengo.Connection(vec_node,  slam.landmark_vec_ssp,     synapse=None)
    nengo.Connection(id_node,   slam.landmark_id_input,    synapse=None)
    nengo.Connection(view_node, slam.no_landmark_in_view,  synapse=None)

    pi_probe  = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    mem_probe = nengo.Probe(slam.assomemory.conn_out, 'weights', sample_every=T_sim)

print("Building simulator...")
t0 = time.time()
with nengo.Simulator(model, dt=dt_nengo, progress_bar=True) as sim:
    sim.run(T_sim)
elapsed = time.time() - t0
print(f"Simulation complete in {elapsed:.1f}s")

slam_output = sim.data[pi_probe]
mem_weights = sim.data[mem_probe][-1].T
ts          = sim.trange()

slam_path = ssp_space.decode(slam_output, 'from-set', 'grid', 100)
slam_sims = np.sum(slam_output * real_ssp, axis=1) / np.maximum(
    1e-6, np.linalg.norm(slam_output, axis=1)
)
slam_error = np.linalg.norm(slam_path - path_sim[:len(slam_path)], axis=1)

print(f"\nMean cosine similarity: {slam_sims.mean():.3f}")
print(f"Mean position error:    {slam_error.mean():.4f} (normalised coords)")

# ---------------------------------------------------------------------------
# Decode landmark locations
# ---------------------------------------------------------------------------

acts = nengo.builder.ensemble.get_activities(
    sim.data[slam.assomemory.memory],
    slam.assomemory.memory,
    lm_space.vectors,
)
landmark_ssps_est = np.dot(acts, mem_weights)
landmark_loc_est  = ssp_space.decode(landmark_ssps_est, 'from-set', 'grid', 100)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs(args.save_dir, exist_ok=True)
mode = "events" if args.events_txt else "rgb_sim"
fname = (
    f"slam_event_{mode}_sspdim_{ssp_dim}_featdim_{feat_dim}"
    f"_T_{int(T_sim)}_seed_{args.seed}.npz"
)
np.savez(
    os.path.join(args.save_dir, fname),
    ts=ts,
    path=path_sim,
    slam_path=slam_path,
    slam_sims=slam_sims,
    slam_error=slam_error,
    slam_sim_out=slam_output,
    obj_locs=landmark_loc_est,
    landmark_ssps_est=landmark_ssps_est,
    landmark_loc_est=landmark_loc_est,
    elapsed_time=elapsed,
)
print(f"Saved: {os.path.join(args.save_dir, fname)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

if args.plot:
    import matplotlib.pyplot as plt

    # Build query grid for cognitive map
    res = 50
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts  = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_ssps = ssp_space.encode(grid_pts)

    fig, axes = plt.subplots(2, max(n_landmarks, 3), figsize=(4 * max(n_landmarks, 3), 8))

    # Row 1: slam performance
    axes[0, 0].plot(ts, slam_sims, lw=0.8, color='steelblue')
    axes[0, 0].set_title('SSP cosine similarity'); axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_xlabel('Time (s)')

    n = min(len(slam_path), len(path_sim))
    axes[0, 1].plot(path_sim[:n, 0], path_sim[:n, 1], 'k-', lw=1.5, alpha=0.6, label='True')
    axes[0, 1].plot(slam_path[:, 0], slam_path[:, 1], 'b--', lw=1.2, alpha=0.7, label='SLAM')
    axes[0, 1].set_aspect('equal'); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title('Path: True vs SLAM estimate')

    axes[0, 2].plot(ts[:len(slam_error)], slam_error, lw=0.8, color='darkorange')
    axes[0, 2].axhline(slam_error.mean(), color='red', ls='--', lw=1.5,
                       label=f'Mean={slam_error.mean():.3f}')
    axes[0, 2].legend(fontsize=8); axes[0, 2].set_title('Position error over time')
    for j in range(3, axes.shape[1]):
        axes[0, j].axis('off')

    # Row 2: cognitive map heatmaps
    for i in range(n_landmarks):
        ax = axes[1, i] if i < axes.shape[1] else None
        if ax is None: break
        sims = (grid_ssps @ landmark_ssps_est[i]).reshape(res, res)
        im = ax.imshow(sims, origin='lower', extent=[-1, 1, -1, 1],
                       cmap='hot', interpolation='bilinear', vmin=0)
        ax.scatter(*landmark_loc_est[i], s=120, marker='x', color='lime', zorder=5,
                   linewidths=2, label='SLAM est.')
        ax.set_title(f'Landmark {i}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.legend(fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n_landmarks, axes.shape[1]):
        axes[1, j].axis('off')

    plt.suptitle('Event-camera SSP-SLAM Results', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'event_slam_{mode}.png'), dpi=150)
    plt.show()
    print(f"Plot saved to {args.save_dir}/event_slam_{mode}.png")
