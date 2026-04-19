import unittest
import numpy as np
import sys, os
sys.path.insert(1, os.path.dirname(os.getcwd()))

from sspslam.perception.event_dataset_parser import accumulate_events_to_frames
from sspslam.perception.event_vision_adapter import EventVisionAdapter

class TestEventVisionPipeline(unittest.TestCase):
    def test_accumulate_events(self):
        # Create dummy events across 100ms
        np.random.seed(42)
        n_events = 5000
        t = np.sort(np.random.uniform(0.0, 0.1, n_events))
        x = np.random.randint(0, 346, n_events)
        y = np.random.randint(0, 260, n_events)
        p = np.random.randint(0, 2, n_events)
        
        frames, times = accumulate_events_to_frames(t, x, y, p, dt_window=0.02, width=346, height=260)
        
        # 100ms / 20ms window = 5 frames
        self.assertEqual(frames.shape[0], 5)
        self.assertEqual(frames.shape[1], 260)
        self.assertEqual(frames.shape[2], 346)
        self.assertEqual(times.shape[0], 5)
        
        # Ensure we have bright spots
        self.assertTrue(np.max(frames) == 255)

    def test_event_adapter(self):
        # Create 10 dummy frames
        frames = np.random.randint(0, 2, (10, 260, 346), dtype=np.uint8) * 255
        
        adapter = EventVisionAdapter(max_landmarks=15, view_radius=5.0)
        
        # Process dummy frames
        vel_arr, vec_arr, feat_arr = adapter.process_dataset(frames)
        
        self.assertEqual(vel_arr.shape, (10, 2))
        self.assertEqual(vec_arr.shape, (10, 15, 2))
        self.assertEqual(feat_arr.shape, (10, 15, 32))
        
        # If the frames were random static noise, some corners must have been found
        feat_sum = np.sum(np.abs(feat_arr))
        self.assertTrue(feat_sum > 0, "No features were extracted from the event surfaces")

if __name__ == '__main__':
    unittest.main()
