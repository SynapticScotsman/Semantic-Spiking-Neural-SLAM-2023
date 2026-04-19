import unittest
import numpy as np
import gymnasium as gym
import miniworld

import sys, os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from sspslam.perception.miniworld_adapter import MiniWorldVisionAdapter

class TestMiniWorldVisionAdapter(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("MiniWorld-ThreeRooms-v0", render_mode="rgb_array")
        self.adapter = MiniWorldVisionAdapter(self.env, max_landmarks=10, view_radius=4.0)

    def test_collect_episode(self):
        # Force the agent to spin to guarantee seeing features in the sparse room
        def spin_policy(obs):
            return 0  # 0 is turn left

        vel, vec, feat, path = self.adapter.collect_episode(max_steps=50, policy=spin_policy)
        
        # Check shapes
        # vel should be (20, 2)
        # vec should be (20, 10, 2)
        # feat should be (20, 10, 32)
        
        self.assertEqual(vel.shape, (50, 2))
        self.assertEqual(vec.shape, (50, 10, 2))
        self.assertEqual(feat.shape, (50, 10, 32))
        self.assertEqual(path.shape, (51, 2))
        
        # Ensure that some feature representations are non-zero (landmarks tracked)
        # Not all may be tracked, but some sum should be > 0
        feat_sum = np.sum(np.abs(feat))
        self.assertTrue(feat_sum > 0, "No features were extracted")

    def tearDown(self):
        self.env.close()

if __name__ == '__main__':
    unittest.main()
