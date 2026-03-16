"""
run_slam_features.py
====================
SSP-SLAM driven by continuous image feature vectors rather than discrete
landmark IDs.

This script demonstrates how to replace the classical ``SPSpace`` (which
assigns an arbitrary random unit-vector to each landmark ID) with an
``ImageFeatureEncoder`` that maps a landmark's *appearance* feature vector
(e.g. from SIFT, HOG, ORB, or any CNN backbone) to a Semantic Pointer.

The script can run in two modes
--------------------------------
1. **Synthetic features** (default)
   Generates a random 2-D path, places landmarks, and synthesises a
   feature descriptor for each landmark as a fixed random vector (with
   optional Gaussian noise to simulate noisy observations).

2. **Feature file** (``--feature-data``)
   Loads a pre-computed ``.npy`` file of shape
   ``(T, N_landmarks, feat_dim)`` — e.g. per-frame SIFT descriptors
   extracted offline with OpenCV — and uses those directly.

Usage
-----
    # Synthetic features, CPU backend
    python experiments/run_slam_features.py --backend cpu --T 60 --plot

    # Pre-computed feature file
    python experiments/run_slam_features.py \\
        --feature-data my_features.npy \\
        --path-data example_paths/oneRoom_path2.npy \\
        --backend ocl --plot

See ``python experiments/run_slam_features.py --help`` for all options.
"""

import numpy as np
import nengo
import sys, os, time, argparse

sys.path.insert(1, os.path.dirname(os.getcwd()))
import sspslam
from sspslam.networks import SLAMNetwork, get_slam_input_functions_from_features
from sspslam.perception import ImageFeatureEncoder

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="SSP-SLAM with image feature vectors as landmark identifiers.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--backend", default="cpu", type=str,
                    help="Nengo backend: cpu | ocl | loihi-sim | loihi")
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dimensionality of the space the agent moves in.")
parser.add_argument("--path-data", default=None, type=str,
                    help=".npy file with an (T × domain_dim) trajectory array. "
                         "If None a random path is generated.")
parser.add_argument("--data-dt", default=0.001, type=float,
                    help="Timestep of --path-data (s). Interpolated to 0.001 s if different.")
parser.add_argument("--feature-data", default=None, type=str,
                    help=".npy file with a (T × N_landmarks × feat_dim) array of "
                         "pre-computed feature vectors. If None, synthetic features are used.")
parser.add_argument("--feat-dim", default=128, type=int,
                    help="Dimensionality of synthetic feature vectors (ignored if "
                         "--feature-data is supplied).")
parser.add_argument("--feat-noise", default=0.05, type=float,
                    help="Std-dev of Gaussian noise added to synthetic features "
                         "to simulate noisy observations.")

parser.add_argument("--limit", default=0.1, type=float,
                    help="Max frequency of random path (WhiteSignal high parameter).")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=100.0, type=float,
                    help="Simulation duration (seconds).")
parser.add_argument("--n-landmarks", default=30, type=int,
                    help="Number of landmarks (only used with synthetic path/features).")
parser.add_argument("--view-rad", default=0.2, type=float,
                    help="Agent view radius.")
parser.add_argument("--update-thres", default=0.2, type=float,
                    help="Cosine-similarity threshold for loop closure.")
parser.add_argument("--shift-rate", default=0.2, type=float,
                    help="Loop-closure correction rate.")

parser.add_argument("--pi-n-neurons", default=500, type=int,
                    help="Neurons per VCO in the path-integrator network.")
parser.add_argument("--mem-n-neurons", default=500, type=int,
                    help="Neurons in the associative-memory population.")
parser.add_argument("--circonv-n-neurons", default=70, type=int,
                    help="Neurons per dim in circular-convolution sub-networks.")
parser.add_argument("--gc-n-neurons", default=0, type=int,
                    help="Grid-cell cleanup neurons (0 = node-based cleanup).")

parser.add_argument("--ssp-dim", default=97, type=int,
                    help="SSP / SP dimensionality.")
parser.add_argument("--n-scales", default=0, type=int,
                    help="Hex-SSP scales (0 → use --ssp-dim).")
parser.add_argument("--n-rotates", default=3, type=int,
                    help="Hex-SSP rotations (only used when --n-scales > 0).")
parser.add_argument("--use-rand", action="store_true",
                    help="Use RandomSSPSpace instead of HexagonalSSPSpace.")
parser.add_argument("--length-scale", default=0.2, type=float,
                    help="SSP length scale.")

parser.add_argument("--no-voja", action="store_true",
                    help="Disable Voja learning rule on memory encoders.")
parser.add_argument("--no-cleanup", action="store_true",
                    help="Disable PI-output cleanup step.")

parser.add_argument("--save", action="store_true",
                    help="Save simulation results to --save-dir.")
parser.add_argument("--save-dir", default="data",
                    help="Directory for saved results.")
parser.add_argument("--plot", action="store_true",
                    help="Plot results after simulation.")
parser.add_argument("--save-plot", action="store_true",
                    help="Save figure (requires --plot).")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------

if args.backend == "ocl":
    import nengo_ocl
elif "loihi" in args.backend:
    import nengo_loihi
    from nengo_loihi.neurons import LoihiLIF
    nengo_loihi.set_defaults()

# ---------------------------------------------------------------------------
# Path / trajectory
# ---------------------------------------------------------------------------

dt = 0.001
radius = 1.0

def stretch_trajectory(traj, original_dt=0.02, new_dt=0.001):
    n_steps = traj.shape[0]
    total_time = n_steps * original_dt
    n_timesteps = int(total_time / new_dt)
    orig_t = np.linspace(0, total_time, n_steps)
    new_t = np.linspace(0, total_time, n_timesteps)
    out = np.stack(
        [np.interp(new_t, orig_t, traj[:, i]) for i in range(traj.shape[1])],
        axis=1,
    )
    return out


if args.path_data is None:
    T = args.T
    domain_dim = args.domain_dim
    path = np.hstack([
        nengo.processes.WhiteSignal(T, high=args.limit, seed=args.seed + i).run(T, dt=dt)
        for i in range(domain_dim)
    ])
else:
    path = np.load(os.path.join(os.getcwd(), args.path_data))[:99999, :]
    if args.data_dt != dt:
        path = stretch_trajectory(path, original_dt=args.data_dt, new_dt=dt)
    T = path.shape[0] * dt
    domain_dim = path.shape[1]

# Normalise path to [-0.9, 0.9]
shift = lambda x, lo, hi: (hi - lo) * (x - x.min()) / (x.max() - x.min()) + lo
for i in range(domain_dim):
    path[:, i] = shift(path[:, i], -0.9 * radius, 0.9 * radius)

pathlen = path.shape[0]
timesteps = np.arange(0, T, dt)
vels = (1.0 / dt) * np.diff(path, axis=0, prepend=path[[0], :])

# ---------------------------------------------------------------------------
# Landmark positions and feature vectors
# ---------------------------------------------------------------------------

if args.feature_data is not None:
    feature_vectors_data = np.load(
        os.path.join(os.getcwd(), args.feature_data)
    )
    # Derive n_landmarks and feat_dim from the loaded array
    n_landmarks = feature_vectors_data.shape[1]
    feat_dim = feature_vectors_data.shape[2]
    # Still need landmark positions — assume uniform random placement
    rng = np.random.RandomState(args.seed)
    obj_locs = 0.9 * radius * 2 * (
        sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5
    )
else:
    n_landmarks = args.n_landmarks
    feat_dim = args.feat_dim
    obj_locs = 0.9 * radius * 2 * (
        sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5
    )
    # Each landmark has a fixed "true" feature descriptor (a random unit vector)
    rng = np.random.RandomState(args.seed + 42)
    true_features = rng.randn(n_landmarks, feat_dim)
    true_features /= np.linalg.norm(true_features, axis=1, keepdims=True)

    # Replicate across time and add observation noise
    feature_vectors_data = np.tile(
        true_features[np.newaxis, :, :], (pathlen, 1, 1)
    )
    if args.feat_noise > 0:
        noise = rng.randn(*feature_vectors_data.shape) * args.feat_noise
        feature_vectors_data = feature_vectors_data + noise

vec_to_landmarks = obj_locs[np.newaxis, :, :] - path[:, np.newaxis, :]

# ---------------------------------------------------------------------------
# SSP space
# ---------------------------------------------------------------------------

bounds = radius * np.tile([-1, 1], (domain_dim, 1))
if args.use_rand:
    ssp_space = sspslam.RandomSSPSpace(
        domain_dim, ssp_dim=args.ssp_dim,
        domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed,
    )
else:
    if args.n_scales > 0:
        ssp_space = sspslam.HexagonalSSPSpace(
            domain_dim, n_scales=args.n_scales, n_rotates=args.n_rotates,
            domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed,
        )
    else:
        ssp_space = sspslam.HexagonalSSPSpace(
            domain_dim, ssp_dim=args.ssp_dim,
            domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed,
        )
d = ssp_space.ssp_dim

real_ssp = ssp_space.encode(path)

# ---------------------------------------------------------------------------
# Image feature encoder  ← THE NEW PIECE
# ---------------------------------------------------------------------------

feature_encoder = ImageFeatureEncoder(
    feat_dim=feat_dim,
    ssp_dim=d,
    seed=args.seed,
)

print(f"ImageFeatureEncoder: feat_dim={feat_dim} → ssp_dim={d}")

# Build SLAM input functions using feature vectors instead of a discrete SPSpace
(velocity_func, vel_scaling_factor,
 is_landmark_in_view, landmark_id_func,
 landmark_sp_func, landmark_vec_func,
 landmark_vecssp_func) = get_slam_input_functions_from_features(
    ssp_space, feature_encoder,
    vels, vec_to_landmarks, feature_vectors_data,
    args.view_rad, dt=dt,
)

# Build a dummy lm_space so SLAMNetwork can compute the intercept threshold.
# We pass the pre-encoded SP matrix directly, so the SPSpace.vectors are
# the feature-derived SPs rather than random ones.
landmark_sps_precomputed = feature_encoder.encode(
    feature_vectors_data.mean(axis=0)
)
lm_space = sspslam.SPSpace(n_landmarks, d, seed=args.seed,
                           vectors=landmark_sps_precomputed)

# ---------------------------------------------------------------------------
# Build Nengo model
# ---------------------------------------------------------------------------

clean_up_method = None if args.no_cleanup else "grid"
tau = 0.05

model = nengo.Network(seed=args.seed)
if "loihi" in args.backend:
    model.config[nengo.Ensemble].neuron_type = LoihiLIF()

with model:
    vel_input = nengo.Node(velocity_func, label="vel_input")
    init_state = nengo.Node(
        lambda t: real_ssp[int((t - dt) / dt)] if t < 0.05 else np.zeros(d),
        label="init_state",
    )
    landmark_vec_node = nengo.Node(landmark_vecssp_func, label="lm_vecssp_input")
    landmark_id_node = nengo.Node(landmark_sp_func, label="lm_sp_input")
    is_landmark_node = nengo.Node(is_landmark_in_view, label="lm_in_view_input")

    slam = SLAMNetwork(
        ssp_space, lm_space, args.view_rad, n_landmarks,
        args.pi_n_neurons, args.mem_n_neurons, args.circonv_n_neurons,
        tau_pi=tau,
        update_thres=args.update_thres,
        vel_scaling_factor=vel_scaling_factor,
        shift_rate=args.shift_rate,
        voja_learning_rate=1e-4,
        pes_learning_rate=5e-3,
        intercept=0.1,
        clean_up_method=clean_up_method,
        gc_n_neurons=args.gc_n_neurons,
        encoders=None,
        voja=not args.no_voja,
        seed=args.seed,
    )

    nengo.Connection(vel_input, slam.velocity_input, synapse=None)
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    nengo.Connection(landmark_vec_node, slam.landmark_vec_ssp, synapse=None)
    nengo.Connection(landmark_id_node, slam.landmark_id_input, synapse=None)
    nengo.Connection(is_landmark_node, slam.no_landmark_in_view, synapse=None)

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    if args.save:
        mem_weights = nengo.Probe(
            slam.assomemory.conn_out, "weights", sample_every=T
        )

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

nengo.rc["progress"]["progress_bar"] = "nengo.utils.progress.TerminalProgressBar"

if args.backend == "cpu":
    sim = nengo.Simulator(model)
elif args.backend == "ocl":
    sim = nengo_ocl.Simulator(model)
elif args.backend == "loihi-sim":
    sim = nengo_loihi.Simulator(model, remove_passthrough=False, target="sim")
elif args.backend == "loihi":
    sim = nengo_loihi.Simulator(model, remove_passthrough=False, target="loihi",
                                precompute=False)

start = time.time()
with sim:
    sim.run(T)
elapsed = time.time() - start
print(f"Simulation finished in {elapsed:.1f} s")

# ---------------------------------------------------------------------------
# Decode and evaluate
# ---------------------------------------------------------------------------

slam_out = sim.data[slam_output_p]
sim_path_est = ssp_space.decode(slam_out, "from-set", "grid",
                                100 if domain_dim < 3 else 30)
slam_sims = (
    np.sum(slam_out * real_ssp, axis=1)
    / np.maximum(1e-6, np.linalg.norm(slam_out, axis=1))
)
slam_error = np.sqrt(np.sum((path - sim_path_est) ** 2, axis=1))
ts = sim.trange()

print(f"Mean cosine similarity: {slam_sims.mean():.3f}")
print(f"Mean position error:    {slam_error.mean():.4f} m")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

if args.save:
    os.makedirs(args.save_dir, exist_ok=True)
    decoders = sim.data[mem_weights][-1].T
    acts = nengo.builder.ensemble.get_activities(
        sim.data[slam.assomemory.memory],
        slam.assomemory.memory,
        lm_space.vectors,
    )
    landmark_ssps_est = np.dot(acts, decoders)
    landmark_loc_est = ssp_space.decode(
        landmark_ssps_est, "from-set", "grid", 100 if domain_dim < 3 else 30
    )
    fname = (
        f"slam_features_sspdim_{d}_featdim_{feat_dim}"
        f"_pinneurons_{args.pi_n_neurons}"
        f"_T_{int(T)}_seed_{args.seed}.npz"
    )
    np.savez(
        os.path.join(os.getcwd(), args.save_dir, fname),
        timesteps=timesteps, ts=ts,
        path=path, real_ssp=real_ssp,
        obj_locs=obj_locs, view_rad=args.view_rad,
        slam_sim_out=slam_out, slam_sims=slam_sims,
        slam_path=sim_path_est, slam_error=slam_error,
        landmark_ssps_est=landmark_ssps_est,
        landmark_loc_est=landmark_loc_est,
        elapsed_time=elapsed,
    )
    print(f"Saved to {os.path.join(args.save_dir, fname)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

if args.plot:
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cosine

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    axes[0].plot(ts, slam_sims)
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("SSP-SLAM with image feature vectors")

    axes[1].plot(ts, slam_error)
    axes[1].set_ylabel("Position error (m)")
    axes[1].set_xlabel("Time (s)")

    if domain_dim == 2:
        axes[2].plot(path[:, 0], path[:, 1], color="gray", label="True path")
        axes[2].plot(sim_path_est[:, 0], sim_path_est[:, 1], "--",
                     color="k", label="SLAM estimate")
        axes[2].scatter(obj_locs[:, 0], obj_locs[:, 1],
                        marker="*", s=60, color="tab:orange", label="Landmarks")
        axes[2].legend(fontsize=8)
        axes[2].set_xlabel("x (m)")
        axes[2].set_ylabel("y (m)")

    fig.tight_layout()
    plt.show()

    if args.save_plot:
        fig_path = os.path.join("figures",
                                f"slam_features_{args.backend}.png")
        os.makedirs("figures", exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
