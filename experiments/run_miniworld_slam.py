"""
run_miniworld_slam.py
=====================
Runs a full SLAM episode in a Gymnasium MiniWorld environment.
Uses purely visual features (ORB keypoints tracked via optical flow)
rather than ground-truth entity locations.
"""

import numpy as np
import nengo
import sys, os, time
import argparse
import gymnasium as gym
import miniworld

sys.path.insert(1, os.path.dirname(os.getcwd()))
import sspslam
from sspslam.networks import SLAMNetwork, get_slam_input_functions_from_features
from sspslam.perception import ImageFeatureEncoder
from sspslam.perception.miniworld_adapter import MiniWorldVisionAdapter

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
T = 5.0  # short run time for tests
sim_dt = 0.001
env_dt = 0.05
n_steps = int(T / env_dt)
ssp_dim = 97
pi_n_neurons = 250
mem_n_neurons = 500
view_rad = 3.0

print("Generating Data from MiniWorld...")
env = gym.make("MiniWorld-ThreeRooms-v0", render_mode="rgb_array") # richer env
adapter = MiniWorldVisionAdapter(env, max_landmarks=20, view_radius=view_rad)

# Use a straightforward policy that mostly turns so we see features
def policy(obs):
    t = adapter.env.unwrapped.step_count
    return 2 if (t // 5) % 3 != 0 else 0

vel_data, vec_to_landmarks_data, feat_data, path = adapter.collect_episode(max_steps=n_steps, policy=policy)
env.close()

n_landmarks = vec_to_landmarks_data.shape[1]
feat_dim = feat_data.shape[2]

print(f"Collected {vel_data.shape[0]} environment steps.")
print(f"Found {n_landmarks} persistent visual landmarks.")

print("Stretching arrays for Nengo dt...")
vel_data_sim = stretch_array(vel_data, env_dt, sim_dt)
vec_data_sim = stretch_array(vec_to_landmarks_data, env_dt, sim_dt)
feat_data_sim = stretch_array(feat_data, env_dt, sim_dt)
T_sim = vel_data_sim.shape[0] * sim_dt
path_sim = stretch_array(path, env_dt, sim_dt)

print("Building SSP-SLAM...")
domain_dim = 2
radius = 4.0 # miniworld rooms are large
bounds = radius * np.tile([-1, 1], (domain_dim, 1))

ssp_space = sspslam.HexagonalSSPSpace(
    domain_dim, ssp_dim=ssp_dim,
    domain_bounds=bounds, length_scale=0.5, seed=0
)
d = ssp_space.ssp_dim
real_ssp = ssp_space.encode(path_sim)

# Ensure feature encoder knows about feat_dim (32 for ORB)
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
        pi_n_neurons, mem_n_neurons, 70,
        tau_pi=0.05,
        update_thres=0.2,
        vel_scaling_factor=vel_scaling_factor,
        shift_rate=0.1,
        clean_up_method="grid",
        voja=True,
        seed=0,
    )

    nengo.Connection(vel_input, slam.velocity_input, synapse=None)
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    nengo.Connection(landmark_vec_node, slam.landmark_vec_ssp, synapse=None)
    nengo.Connection(landmark_id_node, slam.landmark_id_input, synapse=None)
    nengo.Connection(is_landmark_node, slam.no_landmark_in_view, synapse=None)

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)

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
print("Done!")
