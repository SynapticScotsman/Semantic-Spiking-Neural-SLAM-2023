"""
Tests for ContrastThresholdEventSimulator.
No GPU or special dependencies required — pure NumPy.
"""

import numpy as np
import unittest
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sspslam.perception.event_simulator import (
    ContrastThresholdEventSimulator,
    make_event_simulator,
)


class TestContrastThresholdEventSimulator(unittest.TestCase):

    def _make_frames(self, n=10, h=32, w=32, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (n, h, w), dtype=np.uint8)

    def test_process_frame_returns_correct_shape(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        frames = self._make_frames(5)
        sim.reset(frames[0])
        events = sim.process_frame(frames[1])
        self.assertEqual(events.ndim, 2)
        self.assertEqual(events.shape[1], 4)

    def test_polarity_is_plus_minus_one(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        frames = self._make_frames(5)
        events = sim.frames_to_events(frames)
        if len(events) > 0:
            unique_pol = set(events[:, 3])
            self.assertTrue(unique_pol.issubset({-1.0, 1.0}))

    def test_no_events_on_constant_frame(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.2, C_neg=0.2, dt=1 / 30)
        frame = np.full((32, 32), 128, dtype=np.uint8)
        frames = np.stack([frame] * 5)
        events = sim.frames_to_events(frames)
        self.assertEqual(len(events), 0)

    def test_events_on_sudden_brightness_change(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        dark = np.zeros((32, 32), dtype=np.uint8)
        bright = np.full((32, 32), 255, dtype=np.uint8)
        frames = np.stack([dark, bright])
        events = sim.frames_to_events(frames)
        self.assertGreater(len(events), 0)
        # All should be positive polarity (brightness increased)
        self.assertTrue(np.all(events[:, 3] == 1.0))

    def test_timestamps_monotonically_increase(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        frames = self._make_frames(8)
        events = sim.frames_to_events(frames)
        if len(events) > 1:
            self.assertTrue(np.all(np.diff(events[:, 0]) >= 0))

    def test_simulate_pipeline(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        frames = self._make_frames(10, h=16, w=16)
        event_frames, times, raw_events = sim.simulate(frames, dt_window=0.1)
        self.assertEqual(event_frames.ndim, 3)
        self.assertEqual(event_frames.shape[1], 16)
        self.assertEqual(event_frames.shape[2], 16)
        self.assertEqual(len(times), event_frames.shape[0])

    def test_noise_adds_events(self):
        sim_clean = ContrastThresholdEventSimulator(C_pos=0.5, noise_rate=0.0, dt=1 / 30)
        sim_noisy = ContrastThresholdEventSimulator(C_pos=0.5, noise_rate=10.0, dt=1 / 30, seed=1)
        frame = np.full((32, 32), 128, dtype=np.uint8)
        frames = np.stack([frame] * 5)
        clean_events = sim_clean.frames_to_events(frames)
        noisy_events = sim_noisy.frames_to_events(frames)
        self.assertGreater(len(noisy_events), len(clean_events))

    def test_make_event_simulator_factory(self):
        sim = make_event_simulator(backend="numpy", C_pos=0.2)
        self.assertIsInstance(sim, ContrastThresholdEventSimulator)

    def test_rgb_input(self):
        sim = ContrastThresholdEventSimulator(C_pos=0.1, C_neg=0.1, dt=1 / 30)
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        events = sim.frames_to_events(frames)
        self.assertEqual(events.shape[1], 4)

    def test_refractory_period_limits_events(self):
        # Very long refractory: after first event, same pixel should not fire again
        sim_long = ContrastThresholdEventSimulator(
            C_pos=0.1, C_neg=0.1, refractory_period=1000.0, dt=1 / 30
        )
        sim_short = ContrastThresholdEventSimulator(
            C_pos=0.1, C_neg=0.1, refractory_period=0.0, dt=1 / 30
        )
        frames = self._make_frames(20)
        long_events = sim_long.frames_to_events(frames)
        short_events = sim_short.frames_to_events(frames)
        self.assertLessEqual(len(long_events), len(short_events))


if __name__ == "__main__":
    unittest.main()
