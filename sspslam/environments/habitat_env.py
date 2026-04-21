"""
habitat_env.py
==============
Wraps Habitat-sim into the SSP-SLAM pipeline with the same interface as
``miniworld_env.py``, so ``collect_habitat_data.py`` mirrors
``collect_3d_data.py``.

Requires:
    conda install habitat-sim -c conda-forge -c aihabitat
    or: pip install habitat-sim (Linux wheels only)

Habitat-sim documentation: https://aihabitat.org/docs/habitat-sim/

Landmark objects are approximated using Habitat's object-instance
bounding-box centres; if no semantic annotations are available a set of
random navigable points are used as synthetic "landmark positions" and
identities, which still exercises the full SLAM pipeline.
"""

import numpy as np

try:
    import habitat_sim
    import habitat_sim.utils.common as utils
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic landmark fallback (no real scene needed, used for testing)
# ──────────────────────────────────────────────────────────────────────────────

SYNTHETIC_LABELS = [
    "red_box", "blue_box", "green_ball",
    "yellow_ball", "purple_key", "orange_box",
]


class HabitatSLAMEnv:
    """Habitat-sim environment with SSP-SLAM compatible observation API.

    Parameters
    ----------
    scene_path : str
        Path to a ``.glb`` / ``.ply`` / ``.obj`` scene file.
        If ``None``, raises an error (scene is required).
    sensor_height : float
        Camera height above the floor (metres).
    obs_width, obs_height : int
        RGB sensor resolution.
    n_objects : int
        Number of landmark objects to track (up to 6 if no semantic mesh).
    room_size : float
        Approximate room extent used for coordinate normalisation.
        If ``None``, estimated from the navmesh bounding box.
    view_radius : float
        Objects within this distance (in metres) are "in view".
    enable_depth : bool
        If True, a depth sensor is also rendered.
    seed : int
        RNG seed for navmesh sampling.
    """

    def __init__(
        self,
        scene_path,
        sensor_height=1.5,
        obs_width=256,
        obs_height=192,
        n_objects=4,
        room_size=None,
        view_radius=3.0,
        enable_depth=False,
        seed=0,
    ):
        if not HAS_HABITAT:
            raise ImportError(
                "habitat-sim is required for HabitatSLAMEnv.\n"
                "Install with: conda install habitat-sim -c conda-forge -c aihabitat\n"
                "or visit: https://github.com/facebookresearch/habitat-sim"
            )

        self.scene_path = scene_path
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.n_objects = min(n_objects, len(SYNTHETIC_LABELS))
        self.view_radius = view_radius
        self.enable_depth = enable_depth
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._sim = self._make_sim(sensor_height, enable_depth)
        self._agent = self._sim.initialize_agent(0)

        # Estimate room size from navmesh if not given
        if room_size is None:
            bounds = self._sim.pathfinder.get_bounds()
            extents = bounds[1] - bounds[0]
            self._room_size = float(max(extents[0], extents[2]))
        else:
            self._room_size = float(room_size)

        # Sample landmark positions from navigable points
        self.object_positions = self._sample_landmark_positions()
        self.object_labels = SYNTHETIC_LABELS[: self.n_objects]

        self._prev_pos = None

    # ──────────────────────────────────────────────────────────────────────────
    # Habitat setup helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_sim(self, sensor_height, enable_depth):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path
        sim_cfg.enable_physics = False

        sensors = []

        color_spec = habitat_sim.CameraSensorSpec()
        color_spec.uuid = "color_sensor"
        color_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_spec.resolution = [self.obs_height, self.obs_width]
        color_spec.position = [0.0, sensor_height, 0.0]
        sensors.append(color_spec)

        if enable_depth:
            depth_spec = habitat_sim.CameraSensorSpec()
            depth_spec.uuid = "depth_sensor"
            depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_spec.resolution = [self.obs_height, self.obs_width]
            depth_spec.position = [0.0, sensor_height, 0.0]
            sensors.append(depth_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=0.25),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=10.0),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=10.0),
            ),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    def _sample_landmark_positions(self):
        """Sample n_objects navigable points as synthetic landmark positions."""
        positions = np.zeros((self.n_objects, 2))
        for i in range(self.n_objects):
            pt = self._sim.pathfinder.get_random_navigable_point()
            positions[i] = np.array([pt[0], pt[2]])
        return positions

    # ──────────────────────────────────────────────────────────────────────────
    # Agent state helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_agent_pos_2d(self):
        """Return agent (x, z) in world metres."""
        state = self._agent.get_state()
        p = state.position
        return np.array([p[0], p[2]], dtype=np.float32)

    def _get_agent_heading(self):
        """Return agent heading in radians (from quaternion)."""
        state = self._agent.get_state()
        q = state.rotation
        # Extract y-axis rotation from quaternion (yaw)
        return 2.0 * np.arctan2(q.y, q.w)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — mirrors miniworld_env.py
    # ──────────────────────────────────────────────────────────────────────────

    ACTIONS = ["move_forward", "turn_left", "turn_right"]

    def reset(self, seed=None):
        """Place agent at a random navigable point and return observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start_pos = self._sim.pathfinder.get_random_navigable_point()
        heading = self._rng.uniform(0, 2 * np.pi)

        agent_state = habitat_sim.AgentState()
        agent_state.position = start_pos
        cos_h = float(np.cos(heading / 2))
        sin_h = float(np.sin(heading / 2))
        agent_state.rotation = np.quaternion(cos_h, 0, sin_h, 0)
        self._agent.set_state(agent_state)

        obs = self._sim.get_sensor_observations()
        self._prev_pos = self._get_agent_pos_2d().copy()

        return self._build_obs_dict(obs, np.zeros(2))

    def step(self, action_idx):
        """Take an action (0=forward, 1=left, 2=right) and return obs dict."""
        action_name = self.ACTIONS[int(action_idx)]
        obs = self._sim.step(action_name)

        pos = self._get_agent_pos_2d()
        vel = pos - self._prev_pos
        self._prev_pos = pos.copy()

        return self._build_obs_dict(obs, vel)

    def _build_obs_dict(self, obs, vel):
        rgba = obs["color_sensor"]
        rgb = rgba[:, :, :3].copy()  # drop alpha channel

        pos = self._get_agent_pos_2d()
        heading = self._get_agent_heading()

        objects_in_view = []
        for i in range(self.n_objects):
            obj_pos = self.object_positions[i]
            vec = obj_pos - pos
            dist = np.linalg.norm(vec)
            if dist <= self.view_radius:
                objects_in_view.append({
                    "id": i,
                    "label": self.object_labels[i],
                    "vec_to_object": vec,
                    "distance": dist,
                    "world_position": obj_pos.copy(),
                })

        result = {
            "rgb": rgb,
            "position": pos,
            "velocity": vel,
            "heading": heading,
            "objects_in_view": objects_in_view,
            "terminated": False,
            "truncated": False,
        }

        if self.enable_depth:
            result["depth"] = obs["depth_sensor"]

        return result

    def close(self):
        self._sim.close()


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_to_slam_coords(positions, room_size, slam_radius=1.0):
    """Map world coordinates → [-slam_radius, slam_radius]."""
    return (positions / room_size) * 2 * slam_radius - slam_radius


# ──────────────────────────────────────────────────────────────────────────────
# Episode collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_episode(env, n_steps=2000, policy="explore", seed=0):
    """Run a Habitat episode and collect trajectory + perception data.

    Parameters
    ----------
    env : HabitatSLAMEnv
    n_steps : int
    policy : str
        ``"random"`` or ``"explore"`` (wall-avoidance heuristic).
    seed : int

    Returns
    -------
    dict with keys matching ``miniworld_env.collect_episode()``:
        path               : (T, 2) normalised positions in [-1, 1]
        velocities         : (T, 2) normalised velocities
        rgb_frames         : (T, H, W, 3) uint8
        vec_to_landmarks   : (T, n_objects, 2) relative landmark vectors
        landmark_positions : (n_objects, 2) normalised
        landmark_labels    : list[str]
    """
    rng = np.random.default_rng(seed)
    obs_dict = env.reset(seed=seed)

    room_size = env._room_size

    all_positions = []
    all_velocities = []
    all_frames = []
    all_vecs = []

    for step_i in range(n_steps):
        if policy == "random":
            action = int(rng.integers(0, 3))
        elif policy == "explore":
            pos = obs_dict["position"]
            # Simple wall-avoidance: turn when close to navmesh boundary
            bounds = env._sim.pathfinder.get_bounds()
            margin = 1.0
            near_wall = (
                pos[0] < bounds[0][0] + margin or pos[0] > bounds[1][0] - margin or
                pos[1] < bounds[0][2] + margin or pos[1] > bounds[1][2] - margin
            )
            if near_wall:
                action = int(rng.integers(1, 3))  # turn left or right
            else:
                action = int(rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))
        else:
            action = 0  # always forward

        obs_dict = env.step(action)

        pos_norm = normalize_to_slam_coords(obs_dict["position"], room_size)
        vel_norm = obs_dict["velocity"] / room_size * 2

        all_positions.append(pos_norm)
        all_velocities.append(vel_norm)
        all_frames.append(obs_dict["rgb"])

        obj_pos_norm = normalize_to_slam_coords(env.object_positions, room_size)
        vecs = obj_pos_norm - pos_norm
        all_vecs.append(vecs)

    lm_pos_norm = normalize_to_slam_coords(env.object_positions, room_size)

    return {
        "path": np.array(all_positions, dtype=np.float32),
        "velocities": np.array(all_velocities, dtype=np.float32),
        "rgb_frames": np.stack(all_frames, axis=0),
        "vec_to_landmarks": np.array(all_vecs, dtype=np.float32),
        "landmark_positions": lm_pos_norm.astype(np.float32),
        "landmark_labels": env.object_labels,
    }
