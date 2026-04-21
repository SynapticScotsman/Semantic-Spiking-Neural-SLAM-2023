"""sspslam.environments package."""
from .miniworld_env import SLAMRoom, collect_episode, normalize_to_slam_coords

try:
    from .habitat_env import HabitatSLAMEnv
    from .habitat_env import collect_episode as collect_habitat_episode
    from .habitat_env import normalize_to_slam_coords as normalize_habitat_coords
except ImportError:
    pass  # habitat-sim not installed
