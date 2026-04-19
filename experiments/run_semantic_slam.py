"""
run_semantic_slam.py
====================
SSP-SLAM demonstration using ``SemanticLandmarkEncoder`` to produce landmark
identity Semantic Pointers that encode *both* visual appearance and uncertain
semantic class labels.

Landmark identities are no longer arbitrary random vectors.  Instead each
landmark's identity SP is the circular convolution of its appearance SP
(derived from a random synthetic feature descriptor) and its semantic SP
(a probability-weighted sum of class-label SPs from an SPSpace vocabulary).

After simulation the associative memory is queried in two ways:

  **Query by class** (e.g. "Where are the CHAIRs?")
      key = normalize( Σ_i  bind(app_i, CHAIR_sp) )
      → the decoded map lights up near all CHAIR-probability landmarks,
        with brightness proportional to each landmark's CHAIR weight.

  **Query by appearance** (e.g. "Where is the thing that looks like this?")
      key = normalize( bind(app_novel, sem_expected) )
        where app_novel comes from a slightly-noisy version of one landmark's
        feature descriptor and sem_expected is that landmark's semantic SP.
      → the decoded map lights up near the matching landmark.

Usage
-----
    # CPU backend (default)
    python experiments/run_semantic_slam.py

    # OpenCL backend (faster, requires nengo_ocl)
    python experiments/run_semantic_slam.py --backend ocl

    # Shorter run for a quick check
    python experiments/run_semantic_slam.py --T 15 --pi-n-neurons 150

    python experiments/run_semantic_slam.py --help
"""

import argparse
import os
import sys
import time

import numpy as np
import nengo
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; change to "TkAgg" for pop-up windows
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sspslam
from sspslam import SPSpace, HexagonalSSPSpace
from sspslam.networks import SLAMNetwork
from sspslam.perception import ImageFeatureEncoder, SemanticLandmarkEncoder

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="SSP-SLAM with semantic landmark encoding.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--backend",         default="cpu",  type=str,
                    help="Nengo backend: cpu | ocl")
parser.add_argument("--T",               default=30.0,   type=float,
                    help="Simulation duration (seconds).")
parser.add_argument("--seed",            default=0,      type=int)
parser.add_argument("--ssp-dim",         default=97,     type=int)
parser.add_argument("--feat-dim",        default=128,    type=int,
                    help="Dimensionality of synthetic feature vectors.")
parser.add_argument("--feat-noise",      default=0.05,   type=float,
                    help="Std-dev of Gaussian noise on per-step feature observations.")
parser.add_argument("--view-rad",        default=0.25,   type=float)
parser.add_argument("--pi-n-neurons",    default=250,    type=int)
parser.add_argument("--mem-n-neurons",   default=500,    type=int)
parser.add_argument("--circonv-n-neurons", default=70,   type=int)
parser.add_argument("--save-fig",        default="slam_semantic_results.png", type=str,
                    help="Path to save the results figure (empty string = don't save).")
parser.add_argument("--show-fig",        action="store_true",
                    help="Display the figure interactively (requires a display).")
args = parser.parse_args()

if args.backend == "ocl":
    import nengo_ocl

# ---------------------------------------------------------------------------
# Landmark semantic definitions
# ---------------------------------------------------------------------------

LABEL_NAMES = ["CHAIR", "TABLE", "DOOR"]

# Each landmark's class-probability distribution
LANDMARK_LABEL_PROBS = [
    {"CHAIR": 0.9, "TABLE": 0.1},          # lm 0 — almost certainly a CHAIR
    {"TABLE": 0.8, "CHAIR": 0.2},          # lm 1 — mostly TABLE
    {"CHAIR": 0.5, "DOOR":  0.5},          # lm 2 — equal CHAIR / DOOR
    {"DOOR":  0.85, "TABLE": 0.15},        # lm 3 — mostly DOOR
]
N_LANDMARKS = len(LANDMARK_LABEL_PROBS)

# Expected CHAIR probabilities (for verification)
CHAIR_PROBS = np.array([lp.get("CHAIR", 0.0) for lp in LANDMARK_LABEL_PROBS])

# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

dt     = 0.001
radius = 1.0
T      = args.T
seed   = args.seed

path = np.hstack([
    nengo.processes.WhiteSignal(T, high=0.05, seed=seed + 0).run(T, dt=dt),
    nengo.processes.WhiteSignal(T, high=0.05, seed=seed + 1).run(T, dt=dt),
])
shift = lambda x, lo, hi: (hi - lo) * (x - x.min()) / (x.max() - x.min()) + lo
path[:, 0] = shift(path[:, 0], -0.85 * radius, 0.85 * radius)
path[:, 1] = shift(path[:, 1], -0.85 * radius, 0.85 * radius)

pathlen = path.shape[0]
vels    = (1.0 / dt) * np.diff(path, axis=0, prepend=path[[0], :])

# ---------------------------------------------------------------------------
# Landmark positions
# ---------------------------------------------------------------------------

item_locations = np.array([
    [-0.7, -0.5],
    [ 0.6, -0.4],
    [-0.1,  0.5],
    [ 0.4,  0.6],
])
vec_to_landmarks = item_locations[np.newaxis, :, :] - path[:, np.newaxis, :]

# ---------------------------------------------------------------------------
# SSP space
# ---------------------------------------------------------------------------

domain_dim = 2
bounds     = radius * np.tile([-1, 1], (domain_dim, 1))
ssp_space  = HexagonalSSPSpace(
    domain_dim, ssp_dim=args.ssp_dim,
    domain_bounds=1.2 * bounds, length_scale=0.3, seed=seed,
)
d = ssp_space.ssp_dim

real_ssp     = ssp_space.encode(path)
real_inv_ssp = ssp_space.invert(real_ssp)

# ---------------------------------------------------------------------------
# Semantic landmark encoder
# ---------------------------------------------------------------------------

# SPSpace vocabulary: one random unitary vector per class label.
# domain_size must be >= number of labels.
semantic_sp_space = SPSpace(domain_size=len(LABEL_NAMES), dim=d, seed=seed)

image_encoder = ImageFeatureEncoder(
    feat_dim=args.feat_dim, ssp_dim=d, seed=seed,
)
sem_enc = SemanticLandmarkEncoder(
    image_encoder=image_encoder,
    sp_space=semantic_sp_space,
    label_names=LABEL_NAMES,
)

# Generate a fixed "true" appearance feature for each landmark
rng           = np.random.RandomState(seed + 42)
true_features = rng.randn(N_LANDMARKS, args.feat_dim)
true_features /= np.linalg.norm(true_features, axis=1, keepdims=True)

# Pre-encode: appearance SPs and compound landmark identity SPs
appearance_sps = np.vstack([
    sem_enc.get_appearance_sp(true_features[i]) for i in range(N_LANDMARKS)
])
landmark_sps = np.vstack([
    sem_enc.encode(true_features[i], LANDMARK_LABEL_PROBS[i])
    for i in range(N_LANDMARKS)
])

print(f"SSP dim: {d}   feat dim: {args.feat_dim}   n_landmarks: {N_LANDMARKS}")
print(f"Labels : {LABEL_NAMES}")
for i in range(N_LANDMARKS):
    print(f"  lm {i}: {LANDMARK_LABEL_PROBS[i]}")

# ---------------------------------------------------------------------------
# SPSpace wrapping landmark SPs (needed by SLAMNetwork for intercept calc)
# ---------------------------------------------------------------------------

lm_space = SPSpace(N_LANDMARKS, d, seed=seed, vectors=landmark_sps)

# ---------------------------------------------------------------------------
# SLAM input functions
# ---------------------------------------------------------------------------

vel_scaling_factor = 1.0 / np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled        = vels * vel_scaling_factor
velocity_func      = lambda t: vels_scaled[int(np.minimum(np.floor(t / dt), pathlen - 2))]


def landmark_id_func(t):
    vecs  = vec_to_landmarks[int((t - dt) / dt), :, :]
    dists = np.linalg.norm(vecs, axis=1)
    if np.all(dists > args.view_rad):
        return -1
    return int(np.argmin(dists))


def landmark_sp_func(t):
    idx = landmark_id_func(t)
    if idx < 0:
        return np.zeros(d)
    # Add small observation noise to feature and re-encode on each call
    noisy_feat = true_features[idx] + args.feat_noise * rng.randn(args.feat_dim)
    return sem_enc.encode(noisy_feat, LANDMARK_LABEL_PROBS[idx])


def landmark_vecssp_func(t):
    idx = landmark_id_func(t)
    if idx < 0:
        return np.zeros(d)
    return ssp_space.encode(
        vec_to_landmarks[int((t - dt) / dt), idx, :]
    ).flatten()


def is_landmark_in_view(t):
    return 0 if landmark_id_func(t) >= 0 else 10


# ---------------------------------------------------------------------------
# Nengo model
# ---------------------------------------------------------------------------

intercept = min(
    (landmark_sps @ landmark_sps.T - np.eye(N_LANDMARKS)).flatten().max(),
    0.1,
)

model = nengo.Network(seed=seed)
with model:
    vel_input    = nengo.Node(velocity_func, label="vel_input")
    init_state   = nengo.Node(
        lambda t: real_ssp[int((t - dt) / dt)] if t < 0.05 else np.zeros(d),
        label="init_state",
    )
    lm_vec_node  = nengo.Node(landmark_vecssp_func, label="lm_vec")
    lm_id_node   = nengo.Node(landmark_sp_func,     label="lm_id")
    is_lm_node   = nengo.Node(is_landmark_in_view,  label="is_lm")

    slam = SLAMNetwork(
        ssp_space, lm_space, args.view_rad, N_LANDMARKS,
        args.pi_n_neurons, args.mem_n_neurons, args.circonv_n_neurons,
        tau_pi=0.05,
        update_thres=0.2,
        vel_scaling_factor=vel_scaling_factor,
        shift_rate=0.1,
        voja_learning_rate=5e-4,
        pes_learning_rate=5e-3,
        clean_up_method="grid",
        gc_n_neurons=0,
        encoders=None,
        voja=True,
        seed=seed,
    )

    nengo.Connection(lm_vec_node,  slam.landmark_vec_ssp,  synapse=None)
    nengo.Connection(lm_id_node,   slam.landmark_id_input, synapse=None)
    nengo.Connection(is_lm_node,   slam.no_landmark_in_view, synapse=None)
    nengo.Connection(vel_input,    slam.velocity_input,    synapse=None)
    nengo.Connection(init_state,   slam.pathintegrator.input, synapse=None)

    # Reference path integrator (no SLAM correction)
    pi_ref = sspslam.networks.PathIntegration(
        ssp_space, args.pi_n_neurons, 0.05,
        scaling_factor=vel_scaling_factor, stable=True, solver_weights=False,
    )
    nengo.Connection(vel_input,  pi_ref.velocity_input, synapse=None)
    nengo.Connection(init_state, pi_ref.input,          synapse=None)

    ssp_p       = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    pi_ref_p    = nengo.Probe(pi_ref.output, synapse=0.05)
    mem_weights = nengo.Probe(slam.assomemory.conn_out, "weights", sample_every=T)
    mem_enc_p   = nengo.Probe(slam.assomemory.conn_in.learning_rule,
                              "scaled_encoders", sample_every=T)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

nengo.rc["progress"]["progress_bar"] = "nengo.utils.progress.TerminalProgressBar"

if args.backend == "ocl":
    sim = nengo_ocl.Simulator(model)
else:
    sim = nengo.Simulator(model)

print(f"\nRunning {T}s simulation on backend '{args.backend}'…")
t0 = time.time()
with sim:
    sim.run(T)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f} s")

# ---------------------------------------------------------------------------
# Post-simulation analysis helpers
# ---------------------------------------------------------------------------

decoders   = sim.data[mem_weights][-1].T
encoders_w = sim.data[mem_enc_p][-1]


def get_mem_out(key_sp):
    """Query the associative memory with a single SP key vector."""
    key_sp = np.atleast_2d(key_sp)
    x = np.dot(key_sp, encoders_w.T)
    with sim:
        acts = slam.assomemory.memory.neuron_type.rates(
            x,
            sim.data[slam.assomemory.memory].gain,
            sim.data[slam.assomemory.memory].bias,
        )
    return np.dot(acts, decoders).squeeze()


# ---------------------------------------------------------------------------
# Build query keys
# ---------------------------------------------------------------------------

# --- Class queries ---
q_chair = sem_enc.class_query_sp("CHAIR", appearance_sps)
q_table = sem_enc.class_query_sp("TABLE", appearance_sps)
q_door  = sem_enc.class_query_sp("DOOR",  appearance_sps)

# --- Appearance query: landmark 0 with slight feature noise ---
novel_feat = true_features[0] + 0.08 * rng.randn(args.feat_dim)
sem_lm0    = sem_enc.get_semantic_sp(LANDMARK_LABEL_PROBS[0])
q_app_lm0  = sem_enc.appearance_query_sp(novel_feat, sem_lm0)

# Map query results through the associative memory
lm_ssp_hat_chair = get_mem_out(q_chair)
lm_ssp_hat_table = get_mem_out(q_table)
lm_ssp_hat_door  = get_mem_out(q_door)
lm_ssp_hat_app   = get_mem_out(q_app_lm0)

# ---------------------------------------------------------------------------
# Cosine error traces
# ---------------------------------------------------------------------------

slam_out    = sim.data[ssp_p]
pi_ref_out  = sim.data[pi_ref_p]

slam_sims = (
    np.sum(slam_out * real_ssp, axis=1)
    / np.maximum(1e-6, np.linalg.norm(slam_out, axis=1))
)
pi_sims   = (
    np.sum(pi_ref_out * real_ssp, axis=1)
    / np.maximum(1e-6, np.linalg.norm(pi_ref_out, axis=1))
)

print(f"\nMean cosine similarity (SLAM): {slam_sims.mean():.3f}")
print(f"Mean cosine similarity (PI  ): {pi_sims.mean():.3f}")

# ---------------------------------------------------------------------------
# Verify class-query dot-product scaling (console report)
# ---------------------------------------------------------------------------

print("\n--- Class-query dot-product scaling ---")
print(f"  {'lm':>4}  {'p_CHAIR':>8}  {'dot(lm_sp, q_chair)':>22}")
for i in range(N_LANDMARKS):
    d_val = float(np.dot(landmark_sps[i], q_chair * np.linalg.norm(q_chair)))
    # Compare raw dot before normalisation with p_chair
    d_raw = float(landmark_sps[i] @ sem_enc.class_query_sp(
        "CHAIR",
        np.tile(appearance_sps[i], (1, 1)),  # single-landmark version
    ))
    print(f"  {i:4d}  {CHAIR_PROBS[i]:8.2f}  {d_raw:22.4f}  "
          f"(expected ≈ {CHAIR_PROBS[i]:.2f})")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 7))
fig.suptitle("SSP-SLAM with Semantic Landmark Encoding", fontsize=13)

gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

ax_err  = fig.add_subplot(gs[0, 0])
ax_env  = fig.add_subplot(gs[1, 0])
ax_ch   = fig.add_subplot(gs[0, 1])
ax_tb   = fig.add_subplot(gs[0, 2])
ax_dr   = fig.add_subplot(gs[0, 3])
ax_app  = fig.add_subplot(gs[1, 1])

# -- Cosine error --
ts = sim.trange()
ax_err.plot(ts[::10], 1 - slam_sims[::10], label="SLAM", color="tab:blue")
ax_err.plot(ts[::10], 1 - pi_sims[::10],   label="PI",   color="tab:orange", alpha=0.7)
ax_err.set_xlabel("Time (s)")
ax_err.set_ylabel("Cosine error")
ax_err.set_title("Position tracking error")
ax_err.legend(fontsize=8)
ax_err.set_ylim(bottom=0)

# -- Environment --
ax_env.plot(path[:, 0], path[:, 1], color="k", lw=0.8, alpha=0.5, label="True path")
colours = ["tab:blue", "tab:orange", "tab:purple", "tab:brown"]
markers_lm = ["^", "s", "o", "D"]
for i in range(N_LANDMARKS):
    ax_env.plot(*item_locations[i], markers_lm[i], color=colours[i],
                markersize=10, label=f"lm {i}")
    circ = plt.Circle(item_locations[i], args.view_rad,
                      color=colours[i], alpha=0.15, linewidth=0)
    ax_env.add_patch(circ)
ax_env.set_xlim(-1.2, 1.2); ax_env.set_ylim(-1.2, 1.2)
ax_env.set_aspect("equal"); ax_env.set_title("Environment")
ax_env.legend(fontsize=7, loc="upper left")

# Chair-prob label below markers
for i in range(N_LANDMARKS):
    ax_env.text(item_locations[i, 0], item_locations[i, 1] - 0.17,
                f"P(CHAIR)={CHAIR_PROBS[i]:.1f}", ha="center", fontsize=6)

# -- Class query maps --
def sim_plot(ax, ssp_vec, title, item_idx=None):
    ssp_space.similarity_plot(ssp_vec, plot_type="contourf", cmap="Blues",
                              vmin=0.0, ax=ax)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    if item_idx is not None:
        ax.plot(*item_locations[item_idx], markers_lm[item_idx],
                color=colours[item_idx], markersize=9, markeredgecolor="w",
                markeredgewidth=0.5)
    else:
        for i in range(N_LANDMARKS):
            ax.plot(*item_locations[i], markers_lm[i], color=colours[i],
                    markersize=9, markeredgecolor="w", markeredgewidth=0.5)

sim_plot(ax_ch,  lm_ssp_hat_chair, 'Query: "CHAIR"\n(P=0.9,–,0.5,–)')
sim_plot(ax_tb,  lm_ssp_hat_table, 'Query: "TABLE"\n(–,P=0.8,–,P=0.15)')
sim_plot(ax_dr,  lm_ssp_hat_door,  'Query: "DOOR"\n(–,–,P=0.5,P=0.85)')
sim_plot(ax_app, lm_ssp_hat_app,   "Query: appearance ≈ lm 0\n(noisy observation)",
         item_idx=0)

# Tidy unused axes
for ax in [fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]:
    ax.set_visible(False)

if args.save_fig:
    fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.save_fig}")

if args.show_fig:
    matplotlib.use("TkAgg")
    plt.show()
else:
    plt.close(fig)

print("\nDone.")
