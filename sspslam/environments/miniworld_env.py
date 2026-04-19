"""
miniworld_env.py
================
Wraps MiniWorld into the SSP-SLAM pipeline.

Provides a custom room environment with distinct 3D objects and an
interface that outputs per-step: RGB frame, agent position, velocity,
and detected objects with their relative vectors.
"""

import math
import numpy as np
import gymnasium as gym
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, Ball, Key
from miniworld.params import DEFAULT_PARAMS


# Object definitions: (EntityClass, color_name, rgb_tuple, label)
OBJECT_DEFS = [
    (Box,  "red",    (1.0, 0.0, 0.0), "red_box"),
    (Box,  "blue",   (0.0, 0.0, 1.0), "blue_box"),
    (Ball, "green",  (0.0, 0.8, 0.0), "green_ball"),
    (Ball, "yellow", (1.0, 1.0, 0.0), "yellow_ball"),
    (Key,  "purple", (0.6, 0.0, 0.8), "purple_key"),
    (Box,  "orange", (1.0, 0.5, 0.0), "orange_box"),
]


class SLAMRoom(MiniWorldEnv):
    """A single room with distinct colored objects for SSP-SLAM testing.

    Parameters
    ----------
    n_objects : int
        Number of objects to place (max 6).
    room_size : float
        Side length of the square room in world units.
    view_radius : float
        Detection radius — objects within this distance are "in view".
    obs_width, obs_height : int
        Resolution of the RGB observation.
    max_episode_steps : int
        Maximum steps per episode.
    """

    def __init__(self, n_objects=4, room_size=12, view_radius=3.0,
                 obs_width=160, obs_height=120, max_episode_steps=2000,
                 **kwargs):
        self.n_objects = min(n_objects, len(OBJECT_DEFS))
        self._room_size = room_size
        self.view_radius = view_radius
        self._obj_defs = OBJECT_DEFS[:self.n_objects]

        # Positions will be assigned in _gen_world
        self.object_positions = None  # (n_objects, 2) in world coords
        self.object_labels = [d[3] for d in self._obj_defs]
        self._entities_list = []
        self._prev_pos = None

        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", 0.5)
        params.set("turn_step", 15)

        super().__init__(
            obs_width=obs_width,
            obs_height=obs_height,
            max_episode_steps=max_episode_steps,
            params=params,
            **kwargs,
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0, max_x=self._room_size,
            min_z=0, max_z=self._room_size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            ceil_tex="ceiling_tiles",
        )

        # Place objects at fixed, well-separated positions
        margin = 2.0
        positions_2d = [
            (margin, margin),
            (self._room_size - margin, margin),
            (margin, self._room_size - margin),
            (self._room_size - margin, self._room_size - margin),
            (self._room_size / 2, margin),
            (self._room_size / 2, self._room_size - margin),
        ]

        self.object_positions = np.zeros((self.n_objects, 2))
        self._entities_list = []

        for i, (EntityClass, color_name, rgb, label) in enumerate(self._obj_defs):
            pos_2d = positions_2d[i]
            ent = self.place_entity(
                EntityClass(color=color_name, size=0.6 if EntityClass != Key else 0.4),
                pos=np.array([pos_2d[0], 0, pos_2d[1]]),
                dir=0,
            )
            self._entities_list.append(ent)
            self.object_positions[i] = np.array(pos_2d)

        # Place agent near center
        self.place_agent(
            min_x=self._room_size * 0.4,
            max_x=self._room_size * 0.6,
            min_z=self._room_size * 0.4,
            max_z=self._room_size * 0.6,
        )
        self._prev_pos = self._get_agent_pos_2d().copy()

    def _get_agent_pos_2d(self):
        """Return agent (x, z) position as a 2D array."""
        p = self.agent.pos
        return np.array([p[0], p[2]])

    def _get_agent_dir_rad(self):
        """Return agent heading in radians."""
        return self.agent.dir

    def step_and_observe(self, action):
        """Take an action and return enriched observation dict.

        Parameters
        ----------
        action : int
            0=turn_left, 1=turn_right, 2=move_forward

        Returns
        -------
        dict with keys:
            rgb : (H, W, 3) uint8 array
            position : (2,) float — agent [x, z] in world coords
            velocity : (2,) float — change in position since last step
            heading : float — agent heading in radians
            objects_in_view : list of dicts, each with:
                id, label, color, vec_to_object, distance
            terminated : bool
            truncated : bool
        """
        obs, reward, terminated, truncated, info = self.step(action)

        pos = self._get_agent_pos_2d()
        vel = pos - self._prev_pos
        self._prev_pos = pos.copy()

        # Detect objects in view
        objects_in_view = []
        for i, ent in enumerate(self._entities_list):
            obj_pos = self.object_positions[i]
            vec = obj_pos - pos
            dist = np.linalg.norm(vec)

            if dist <= self.view_radius:
                objects_in_view.append({
                    "id": i,
                    "label": self.object_labels[i],
                    "color": self._obj_defs[i][2],
                    "vec_to_object": vec,
                    "distance": dist,
                    "world_position": obj_pos.copy(),
                })

        return {
            "rgb": obs,
            "position": pos,
            "velocity": vel,
            "heading": self._get_agent_dir_rad(),
            "objects_in_view": objects_in_view,
            "terminated": terminated,
            "truncated": truncated,
        }

    def reset_and_observe(self, **kwargs):
        """Reset environment and return initial observation dict."""
        obs, info = self.reset(**kwargs)
        self._prev_pos = self._get_agent_pos_2d().copy()

        return {
            "rgb": obs,
            "position": self._prev_pos.copy(),
            "velocity": np.zeros(2),
            "heading": self._get_agent_dir_rad(),
            "objects_in_view": [],
            "terminated": False,
            "truncated": False,
        }


def normalize_to_slam_coords(positions, room_size, slam_radius=1.0):
    """Map world coordinates [0, room_size] → [-slam_radius, slam_radius].

    This bridges MiniWorld's coordinate system to SSP-SLAM's [-1, 1] domain.
    """
    return (positions / room_size) * 2 * slam_radius - slam_radius


def collect_episode(env, n_steps=2000, policy="random", seed=0):
    """Run an episode and collect trajectory + perception data.

    Parameters
    ----------
    env : SLAMRoom
    n_steps : int
    policy : str
        "random" for random actions, "explore" for a simple exploration heuristic.

    Returns
    -------
    dict with:
        path : (T, 2) normalised agent positions
        velocities : (T, 2)
        rgb_frames : (T, H, W, 3) uint8
        vec_to_landmarks : (T, n_objects, 2) relative vectors
        landmark_positions : (n_objects, 2) normalised
        landmark_labels : list of str
    """
    rng = np.random.RandomState(seed)
    obs_dict = env.reset_and_observe(seed=seed)

    room_size = env._room_size
    n_objects = env.n_objects

    all_positions = []
    all_velocities = []
    all_frames = []
    all_vec_to_landmarks = []  # shape will be (T, n_objects, 2)

    for step_i in range(n_steps):
        # Simple exploration policy
        if policy == "random":
            action = rng.choice([0, 1, 2], p=[0.15, 0.15, 0.7])
        elif policy == "explore":
            # Mostly forward, turn when near walls
            pos = obs_dict["position"]
            margin = 1.5
            near_wall = (pos[0] < margin or pos[0] > room_size - margin or
                         pos[1] < margin or pos[1] > room_size - margin)
            if near_wall:
                action = rng.choice([0, 1])  # turn
            else:
                action = rng.choice([0, 1, 2], p=[0.1, 0.1, 0.8])
        else:
            action = 2  # always forward

        obs_dict = env.step_and_observe(action)

        pos_norm = normalize_to_slam_coords(obs_dict["position"], room_size)
        vel_norm = obs_dict["velocity"] / room_size * 2  # scale velocity too

        all_positions.append(pos_norm)
        all_velocities.append(vel_norm)
        all_frames.append(obs_dict["rgb"])

        # Calculate relative vectors to ALL landmarks (not just visible ones)
        obj_pos_norm = normalize_to_slam_coords(env.object_positions, room_size)
        vecs = obj_pos_norm - pos_norm
        all_vec_to_landmarks.append(vecs)

        if obs_dict["terminated"] or obs_dict["truncated"]:
            obs_dict = env.reset_and_observe(seed=seed + step_i)

    return {
        "path": np.array(all_positions),
        "velocities": np.array(all_velocities),
        "rgb_frames": np.array(all_frames),
        "vec_to_landmarks": np.array(all_vec_to_landmarks),
        "landmark_positions": normalize_to_slam_coords(env.object_positions, room_size),
        "landmark_labels": env.object_labels,
    }
