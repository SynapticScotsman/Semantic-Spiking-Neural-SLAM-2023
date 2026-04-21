"""
event_simulator.py
==================
Converts a sequence of rendered (or real) grayscale frames into
synthetic DVS-style event streams using a contrast-threshold model.

Pure-NumPy implementation of the core ESIM/v2e pixel model — runs
without GPU or special dependencies. If v2e is installed it can be
used as a higher-fidelity backend via ``V2EEventSimulator``.

The model
---------
Each pixel maintains a log-brightness reference. When the cumulative
log-intensity change exceeds +C_pos (positive threshold) or drops below
-C_neg (negative threshold) an event is fired and the reference is
reset. This matches the Inivation DAVIS / DVXplorer behaviour.
"""

import numpy as np


class ContrastThresholdEventSimulator:
    """Convert frame sequences to synthetic DVS events.

    Parameters
    ----------
    C_pos : float
        Positive contrast threshold. Typical real DVS value: 0.15–0.3.
    C_neg : float
        Negative contrast threshold.
    refractory_period : float
        Minimum seconds between events for the same pixel.
    dt : float
        Time between consecutive frames in seconds.
    noise_rate : float
        Background noise events per pixel per second (0 = no noise).
    seed : int
        RNG seed for noise generation.
    """

    def __init__(self, C_pos=0.2, C_neg=0.2, refractory_period=0.001,
                 dt=1 / 30, noise_rate=0.0, seed=0):
        self.C_pos = C_pos
        self.C_neg = C_neg
        self.refractory_period = refractory_period
        self.dt = dt
        self.noise_rate = noise_rate
        self.rng = np.random.default_rng(seed)

        self._log_ref = None        # (H, W) log-brightness reference
        self._t_last = None         # (H, W) time of last event per pixel
        self._t = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, first_frame):
        """Initialise internal state from the first frame of a sequence."""
        gray = self._to_gray(first_frame).astype(np.float32)
        self._log_ref = np.log(gray + 1.0)
        self._t_last = np.full_like(self._log_ref, -np.inf)
        self._t = 0.0

    def process_frame(self, frame):
        """Generate events for one new frame.

        Parameters
        ----------
        frame : np.ndarray
            Shape ``(H, W)`` or ``(H, W, 3)``, dtype uint8.

        Returns
        -------
        np.ndarray
            Shape ``(N, 4)`` — columns ``[t, x, y, polarity]``
            where polarity is +1 (brighter) or -1 (darker).
            Returns empty ``(0, 4)`` array if no events fire.
        """
        gray = self._to_gray(frame).astype(np.float32)
        log_new = np.log(gray + 1.0)

        if self._log_ref is None:
            self.reset(frame)
            return np.zeros((0, 4), dtype=np.float32)

        self._t += self.dt
        delta = log_new - self._log_ref

        # Refractory gate
        ready = (self._t - self._t_last) >= self.refractory_period

        pos_mask = (delta >= self.C_pos) & ready
        neg_mask = (delta <= -self.C_neg) & ready

        events = self._masks_to_events(pos_mask, neg_mask)

        # Background noise
        if self.noise_rate > 0.0:
            events = self._add_noise(events, gray.shape)

        # Update reference and last-event time for fired pixels
        fired = pos_mask | neg_mask
        self._log_ref[fired] = log_new[fired]
        self._t_last[fired] = self._t

        return events

    def frames_to_events(self, frames, dt=None):
        """Convert a full frame sequence to an event array.

        Parameters
        ----------
        frames : np.ndarray
            Shape ``(T, H, W)`` or ``(T, H, W, 3)``, dtype uint8.
        dt : float, optional
            Override the per-frame timestep.

        Returns
        -------
        np.ndarray
            Shape ``(N, 4)`` — ``[t, x, y, polarity]`` sorted by time.
        """
        if dt is not None:
            self.dt = dt

        self.reset(frames[0])
        chunks = [self.process_frame(f) for f in frames[1:]]
        valid = [c for c in chunks if len(c) > 0]
        return np.concatenate(valid, axis=0) if valid else np.zeros((0, 4), dtype=np.float32)

    def events_to_frames(self, events, height, width, dt_window):
        """Accumulate events into binary temporal frames.

        Convenience wrapper around
        ``event_dataset_parser.accumulate_events_to_frames``.

        Parameters
        ----------
        events : np.ndarray (N, 4)
            Output of ``frames_to_events``.
        height, width : int
        dt_window : float
            Duration of each output frame in seconds.

        Returns
        -------
        frames : np.ndarray (T, H, W) uint8
        times : np.ndarray (T,) float
        """
        from .event_dataset_parser import accumulate_events_to_frames
        if len(events) == 0:
            return np.zeros((1, height, width), dtype=np.uint8), np.array([0.0])
        t = events[:, 0]
        x = events[:, 1].astype(np.int32)
        y = events[:, 2].astype(np.int32)
        p = (events[:, 3] > 0).astype(np.uint8)
        return accumulate_events_to_frames(t, x, y, p, dt_window, width, height)

    def simulate(self, frames, dt_window=0.05, dt=None):
        """Full pipeline: frames → events → accumulated event frames.

        Parameters
        ----------
        frames : np.ndarray (T, H, W[, 3])
        dt_window : float
            Accumulation window in seconds for output frames.
        dt : float, optional
            Override per-input-frame timestep.

        Returns
        -------
        event_frames : np.ndarray (T_out, H, W) uint8
        times : np.ndarray (T_out,) float
        raw_events : np.ndarray (N, 4)
        """
        H = frames.shape[1]
        W = frames.shape[2]
        raw_events = self.frames_to_events(frames, dt=dt)
        event_frames, times = self.events_to_frames(raw_events, H, W, dt_window)
        return event_frames, times, raw_events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return (0.299 * frame[:, :, 0] +
                    0.587 * frame[:, :, 1] +
                    0.114 * frame[:, :, 2])
        return frame.astype(np.float32)

    def _masks_to_events(self, pos_mask, neg_mask):
        """Vectorised conversion of boolean masks to event rows."""
        rows = []
        ys_p, xs_p = np.where(pos_mask)
        if len(xs_p):
            t_col = np.full(len(xs_p), self._t, dtype=np.float32)
            rows.append(np.stack([t_col, xs_p, ys_p,
                                  np.ones(len(xs_p), dtype=np.float32)], axis=1))
        ys_n, xs_n = np.where(neg_mask)
        if len(xs_n):
            t_col = np.full(len(xs_n), self._t, dtype=np.float32)
            rows.append(np.stack([t_col, xs_n, ys_n,
                                  -np.ones(len(xs_n), dtype=np.float32)], axis=1))
        if rows:
            return np.concatenate(rows, axis=0).astype(np.float32)
        return np.zeros((0, 4), dtype=np.float32)

    def _add_noise(self, events, shape):
        H, W = shape
        n = int(self.noise_rate * H * W * self.dt)
        if n == 0:
            return events
        xs = self.rng.integers(0, W, n).astype(np.float32)
        ys = self.rng.integers(0, H, n).astype(np.float32)
        ts = np.full(n, self._t, dtype=np.float32)
        ps = self.rng.choice([-1.0, 1.0], n).astype(np.float32)
        noise = np.stack([ts, xs, ys, ps], axis=1)
        if len(events):
            return np.concatenate([events, noise], axis=0)
        return noise


class V2EEventSimulator:
    """Higher-fidelity backend using the v2e library (requires GPU).

    Wraps ``v2e.EventEmulator`` with the same API as
    ``ContrastThresholdEventSimulator`` so the two are drop-in
    replacements.

    Install: ``pip install v2e`` (CUDA GPU strongly recommended).
    """

    def __init__(self, C_pos=0.2, C_neg=0.2, refractory_period=0.001,
                 dt=1 / 30, **kwargs):
        try:
            from v2e import EventEmulator
        except ImportError:
            raise ImportError(
                "v2e is required for V2EEventSimulator. "
                "Install with: pip install v2e\n"
                "Note: GPU (CUDA) strongly recommended for real-time speed."
            )
        self.dt = dt
        self._emulator = EventEmulator(
            pos_thres=C_pos,
            neg_thres=C_neg,
            refractory_period_s=refractory_period,
            **kwargs,
        )
        self._t = 0.0

    def reset(self, first_frame):
        self._emulator.reset()
        self._t = 0.0

    def process_frame(self, frame):
        import torch
        self._t += self.dt
        if frame.ndim == 3:
            gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] +
                    0.114 * frame[:, :, 2]).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        t_tensor = torch.tensor(self._t)
        frame_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        events = self._emulator.generate_events(frame_tensor, t_tensor)
        if events is None or len(events) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return events.numpy().astype(np.float32)

    def frames_to_events(self, frames, dt=None):
        if dt is not None:
            self.dt = dt
        self.reset(frames[0])
        chunks = [self.process_frame(f) for f in frames[1:]]
        valid = [c for c in chunks if len(c) > 0]
        return np.concatenate(valid, axis=0) if valid else np.zeros((0, 4), dtype=np.float32)

    def events_to_frames(self, events, height, width, dt_window):
        return ContrastThresholdEventSimulator().events_to_frames(
            events, height, width, dt_window)

    def simulate(self, frames, dt_window=0.05, dt=None):
        H, W = frames.shape[1], frames.shape[2]
        raw = self.frames_to_events(frames, dt=dt)
        ef, times = self.events_to_frames(raw, H, W, dt_window)
        return ef, times, raw


def make_event_simulator(backend="numpy", **kwargs):
    """Factory: return a simulator by name.

    Parameters
    ----------
    backend : str
        ``"numpy"`` (default, no GPU needed) or ``"v2e"`` (GPU, higher fidelity).
    **kwargs
        Passed to the simulator constructor.
    """
    if backend == "v2e":
        return V2EEventSimulator(**kwargs)
    return ContrastThresholdEventSimulator(**kwargs)
