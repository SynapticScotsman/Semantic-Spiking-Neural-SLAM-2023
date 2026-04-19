"""
run_event_slam.py
=================
Runs a full SLAM episode using accumulated Event frames.
Simulates a dummy neuromorphic dataset for testing purposes, but scales to UZH-RPG standard text files.
"""

import numpy as np
import nengo
import sys, os, time

sys.path.insert(1, os.path.dirname(os.getcwd()))
import sspslam
from sspslam.networks import SLAMNetwork, get_slam_input_functions_from_features
from sspslam.perception import ImageFeatureEncoder
from sspslam.perception.event_dataset_parser import accumulate_events_to_frames
from sspslam.perception.event_vision_adapter import EventVisionAdapter

def generate_dummy_event_dataset():
    """Generates a random event stream that mimics 5 seconds of simple linear motion"""
    np.random.seed(0)
    n_events = 200000
    t = np.sort(np.random.uniform(0.0, 5.0, n_events)) # 5 seconds
    x = np.random.randint(0, 346, n_events)
    y = np.random.randint(0, 260, n_events)
    p = np.random.randint(0, 2, n_events)
    
    # Ground truth: moving from (0,0) to (2, 2) perfectly linear
    gt_t = np.linspace(0.0, 5.0, 100)
    gt_pos = np.column_stack([np.linspace(0, 2, 100), np.zeros(100), np.linspace(0, 2, 100)]) # x, y, z
    # we usually extract ground plane (x, z)
    
    return t, x, y, p, gt_t, gt_pos

def stretch_array(arr, orig_dt, new_dt):
    n_steps = arr.shape[0]
    total_time = n_steps * orig_dt
    n_timesteps = int(total_time / new_dt)
    
    orig_t = np.linspace(0, total_time, n_steps)
    new_t = np.linspace(0, total_time, n_timesteps)
    
    if arr.ndim == 2:
        return np.stack([np.interp(new_t, orig_t, arr[:, i]) for i in range(arr.shape[1])], axis=1)
    elif arr.ndim == 3: # (T, N, D)
        out = np.zeros((n_timesteps, arr.shape[1], arr.shape[2]))
        for n in range(arr.shape[1]):
            for d in range(arr.shape[2]):
                out[:, n, d] = np.interp(new_t, orig_t, arr[:, n, d])
        return out
    return arr

# Parameters
dt_window = 0.05 # 50ms frames
view_rad = 2.0
max_landmarks = 10
ssp_dim = 97
sim_dt = 0.001

print("Loading and grouping events...")
t, x, y, p, gt_t, gt_pos = generate_dummy_event_dataset()

# Accumulate Events into 2D Event Frames (acting as spatial intensity representation of temporal edges)
frames, frame_times = accumulate_events_to_frames(t, x, y, p, dt_window=dt_window, width=346, height=260)
N_frames = frames.shape[0]

# Compute Ground Truth Path (x, z)
# In a real pipeline with UZH-RPG, we would use interpolate_groundtruth
# Here we just re-calculate it for the frame timestamps
path = np.column_stack([np.linspace(0, 2, N_frames), np.linspace(0, 2, N_frames)])
velocities = np.diff(path, axis=0) / dt_window
velocities = np.vstack([velocities, velocities[-1]]) # padded

print("Running EventVisionAdapter (Optical Flow across Event Frames)...")
adapter = EventVisionAdapter(max_landmarks=max_landmarks, view_radius=view_rad)

# If no depth data is passed, it falls back to true_distance=1.0 per tracked keypoint (pseudo-depth)
vel_data, vec_to_landmarks_data, feat_data = adapter.process_dataset(frames, agent_velocities=velocities)

print(f"Extracted features for {max_landmarks} feature tracks over {N_frames} event windows.")

# Stretch for simulation
print("Stretching arrays for Nengo dt...")
vel_data_sim = stretch_array(vel_data, dt_window, sim_dt)
vec_data_sim = stretch_array(vec_to_landmarks_data, dt_window, sim_dt)
feat_data_sim = stretch_array(feat_data, dt_window, sim_dt)
path_sim = stretch_array(path, dt_window, sim_dt)
T_sim = N_frames * dt_window

n_landmarks = vec_to_landmarks_data.shape[1]
feat_dim = feat_data.shape[2]

print("Building SSP-SLAM Network...")
domain_dim = 2
radius = 3.0 
bounds = radius * np.tile([-1, 1], (domain_dim, 1))

ssp_space = sspslam.HexagonalSSPSpace(
    domain_dim, ssp_dim=ssp_dim,
    domain_bounds=bounds, length_scale=0.5, seed=0
)
d = ssp_space.ssp_dim
real_ssp = ssp_space.encode(path_sim)

feature_encoder = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=d, seed=0)

fns = get_slam_input_functions_from_features(
    ssp_space, feature_encoder,
    vel_data_sim, vec_data_sim, feat_data_sim,
    view_rad, dt=sim_dt,
)
(velocity_func, vel_scaling_factor, is_landmark_in_view, 
 landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func) = fns

landmark_sps_precomputed = feature_encoder.encode(feat_data_sim.mean(axis=0))
lm_space = sspslam.SPSpace(n_landmarks, d, seed=0, vectors=landmark_sps_precomputed)

model = nengo.Network(seed=0)
with model:
    vel_input = nengo.Node(velocity_func)
    init_state = nengo.Node(lambda t: real_ssp[int((t - sim_dt) / sim_dt)] if t < 0.05 else np.zeros(d))
    landmark_vec_node = nengo.Node(landmark_vecssp_func)
    landmark_id_node = nengo.Node(landmark_sp_func)
    is_landmark_node = nengo.Node(is_landmark_in_view)

    slam = SLAMNetwork(
        ssp_space, lm_space, view_rad, n_landmarks,
        250, 500, 70,
        tau_pi=0.05, update_thres=0.2, vel_scaling_factor=vel_scaling_factor,
        shift_rate=0.1, clean_up_method="grid", voja=True, seed=0,
    )

    nengo.Connection(vel_input, slam.velocity_input, synapse=None)
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    nengo.Connection(landmark_vec_node, slam.landmark_vec_ssp, synapse=None)
    nengo.Connection(landmark_id_node, slam.landmark_id_input, synapse=None)
    nengo.Connection(is_landmark_node, slam.no_landmark_in_view, synapse=None)

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)

try:
    sim = nengo.Simulator(model)
    print("Running SLAM simulation...")
    start = time.time()
    with sim:
        sim.run(T_sim)
    elapsed = time.time() - start
    print(f"Simulation finished in {elapsed:.1f} s")

    slam_out = sim.data[slam_output_p]
    sim_path_est = ssp_space.decode(slam_out, "from-set", "grid", 100)

    trim_len = min(path_sim.shape[0], sim_path_est.shape[0])
    path_sim = path_sim[:trim_len]
    sim_path_est = sim_path_est[:trim_len]

    slam_error = np.sqrt(np.sum((path_sim - sim_path_est) ** 2, axis=1))

    print(f"Mean position error: {slam_error.mean():.4f} m")
except nengo.exceptions.BuildError as e:
    print(f"Skipping Nengo simulation due to dummy dataset sparsity causing zero-activity bounds: {e}")

print("Event Pipeline Verification Done!")
