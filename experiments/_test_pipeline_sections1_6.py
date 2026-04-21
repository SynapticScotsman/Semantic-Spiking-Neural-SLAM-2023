"""Quick smoke-test for demo_full_pipeline.ipynb Sections 1-6 (no SLAM/nengo needed)."""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "3d")

# ── Section 1: Load data ────────────────────────────────────────────────────
print("=== Section 1: Load trajectory data ===")
path               = np.load(os.path.join(DATA_DIR, "path.npy"))
velocities         = np.load(os.path.join(DATA_DIR, "velocities.npy"))
vec_to_landmarks   = np.load(os.path.join(DATA_DIR, "vec_to_landmarks.npy"))
landmark_positions = np.load(os.path.join(DATA_DIR, "landmark_positions.npy"))
feature_vectors    = np.load(os.path.join(DATA_DIR, "feature_vectors.npy"))
rgb_frames         = np.load(os.path.join(DATA_DIR, "rgb_frames_subset.npy"))
with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    meta = json.load(f)

n_steps, n_objects, feat_dim = feature_vectors.shape
labels = meta["landmark_labels"]
print(f"  path:             {path.shape}")
print(f"  velocities:       {velocities.shape}")
print(f"  vec_to_landmarks: {vec_to_landmarks.shape}")
print(f"  feature_vectors:  {feature_vectors.shape}")
print(f"  rgb_frames:       {rgb_frames.shape}")
print(f"  labels:           {labels}")
assert path.shape[1] == 2
assert velocities.shape == path.shape
assert vec_to_landmarks.shape == (n_steps, n_objects, 2)
assert feature_vectors.shape[0] == n_steps
print("  PASS\n")

# ── Section 2: Event simulation ─────────────────────────────────────────────
print("=== Section 2: Event camera simulation ===")
from sspslam.perception.event_simulator import ContrastThresholdEventSimulator
ev_sim = ContrastThresholdEventSimulator(C_pos=0.15, C_neg=0.15, dt=1/30, noise_rate=0.05, seed=0)
# Need ≥2 frames to detect changes; if only 1 saved, synthesise a second by adding noise
_sim_frames = rgb_frames
if len(_sim_frames) < 2:
    rng = np.random.default_rng(42)
    noise = rng.integers(-20, 20, _sim_frames.shape, dtype=np.int16)
    extra = np.clip(_sim_frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _sim_frames = np.concatenate([_sim_frames, extra], axis=0)
event_frames, event_times, raw_events = ev_sim.simulate(_sim_frames, dt_window=0.1)
print(f"  RGB frames:   {rgb_frames.shape}")
print(f"  Event frames: {event_frames.shape}")
print(f"  Raw events:   {len(raw_events):,}")
print(f"  Pos/Neg:      {(raw_events[:,3]>0).sum():,} / {(raw_events[:,3]<0).sum():,}")
assert event_frames.ndim == 3
assert event_frames.shape[1] == rgb_frames.shape[1]
assert event_frames.shape[2] == rgb_frames.shape[2]
assert len(raw_events) > 0, "No events fired — thresholds may be too high"
print("  PASS\n")

# ── Section 3: Feature extraction ──────────────────────────────────────────
print("=== Section 3: Feature extraction ===")
from sspslam.perception.image_feature_encoder import extract_hog
rgb_hog = np.array([extract_hog(f) for f in rgb_frames])
ev_hog  = np.array([extract_hog(f) for f in event_frames])
print(f"  RGB HOG:    {rgb_hog.shape}  norm_mean={np.linalg.norm(rgb_hog,axis=1).mean():.2f}")
print(f"  Event HOG:  {ev_hog.shape}  norm_mean={np.linalg.norm(ev_hog,axis=1).mean():.2f}")

mean_feats = feature_vectors.mean(axis=0)
norms = np.linalg.norm(mean_feats, axis=1, keepdims=True)
mean_feats_norm = mean_feats / np.maximum(norms, 1e-8)
sim_matrix = mean_feats_norm @ mean_feats_norm.T
max_off_diag = sim_matrix[~np.eye(n_objects, dtype=bool)].max()
print(f"  Pairwise feature similarity:\n{np.round(sim_matrix, 3)}")
print(f"  Max off-diagonal: {max_off_diag:.3f}  (< 0.9 = distinguishable)")
assert max_off_diag < 0.9, f"Objects not distinguishable: max_off_diag={max_off_diag:.3f}"
print("  PASS\n")

# ── Section 4: SSP + SP encoding ───────────────────────────────────────────
print("=== Section 4: SSP space + ImageFeatureEncoder ===")
from sspslam import HexagonalSSPSpace
from sspslam.perception import ImageFeatureEncoder

ssp_space = HexagonalSSPSpace(
    domain_dim=2, ssp_dim=97,
    domain_bounds=np.array([[-1, 1], [-1, 1]]), seed=0
)
ssp_dim = ssp_space.ssp_dim
encoder = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=0)

object_sps = np.vstack([
    encoder.encode(feature_vectors[:, j, :].mean(axis=0)[None]).squeeze()
    for j in range(n_objects)
])
sp_sim = object_sps @ object_sps.T
print(f"  SSP dim: {ssp_dim}")
print(f"  Object SP similarity:\n{np.round(sp_sim, 3)}")

path_ssps = ssp_space.encode(path)
assert path_ssps.shape == (len(path), ssp_dim)

# Build query grid
res = 20
xs = np.linspace(-1, 1, res)
ys = np.linspace(-1, 1, res)
xx, yy = np.meshgrid(xs, ys)
grid_ssps = ssp_space.encode(np.stack([xx.ravel(), yy.ravel()], axis=1))
assert grid_ssps.shape == (res * res, ssp_dim)
print(f"  Grid SSPs: {grid_ssps.shape}")

# Quick cognitive-map sanity check: encode true positions, score similarity
lm_ssps_true = np.vstack([ssp_space.encode(landmark_positions[i][None]).squeeze() for i in range(n_objects)])
for i in range(n_objects):
    sims = grid_ssps @ lm_ssps_true[i]
    peak_idx = sims.argmax()
    peak_pt  = np.stack([xx.ravel(), yy.ravel()], axis=1)[peak_idx]
    err = np.linalg.norm(peak_pt - landmark_positions[i])
    print(f"  SSP decode check [{labels[i]}]: true={landmark_positions[i].round(3)}, "
          f"grid_peak={peak_pt.round(3)}, err={err:.3f}")
    assert err < 0.5, f"SSP decode error too large: {err:.3f}"
print("  PASS\n")

# ── Section 5: Stretch + input function build ───────────────────────────────
print("=== Section 5: Build SLAM input functions (no Nengo sim) ===")
from sspslam.networks import get_slam_input_functions_from_features
import sspslam

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

T_sim  = 2.0
dt     = 0.001
n_sim  = int(T_sim / dt)

path_sim = stretch(path, n_sim)
vel_sim  = stretch(velocities, n_sim)
vec_sim  = stretch(vec_to_landmarks, n_sim)
feat_sim = stretch(feature_vectors, n_sim)

vel_scale      = np.abs(vel_sim).max()
vel_sim_scaled = vel_sim / (vel_scale + 1e-8)

view_rad = meta.get("view_radius_normalized", 0.5)

fns = get_slam_input_functions_from_features(
    ssp_space, encoder,
    vel_sim_scaled, vec_sim, feat_sim,
    view_rad, dt=dt
)
(velocity_func, vel_scaling_factor,
 is_landmark_in_view, landmark_id_func,
 landmark_sp_func, landmark_vec_func,
 landmark_vecssp_func) = fns

# Spot-check functions at a few timesteps
for t_check in [0.0, 0.5, 1.0, 1.5]:
    v = velocity_func(t_check)
    in_view = is_landmark_in_view(t_check)
    sp = landmark_sp_func(t_check)
    assert v.shape == (2,), f"velocity shape wrong: {v.shape}"
    assert sp.shape == (ssp_dim,), f"SP shape wrong: {sp.shape}"

lm_sps   = encoder.encode(feat_sim.mean(axis=0))
lm_space = sspslam.SPSpace(n_objects, ssp_dim, seed=0, vectors=lm_sps)
print(f"  Input functions built.  vel_scaling_factor={vel_scaling_factor:.4f}")
print(f"  lm_space.vectors: {lm_space.vectors.shape}")
print("  PASS\n")

print("=" * 50)
print("ALL SECTIONS 1-5 PASSED")
print("=" * 50)
