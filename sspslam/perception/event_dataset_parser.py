import numpy as np
import scipy.interpolate

def load_events_txt(file_path):
    """
    Load an event sequence from a text file standard to UZH-RPG and others.
    Expected format: [timestamp, x, y, polarity]
    """
    data = np.loadtxt(file_path)
    # Ensure standard order (some datasets reverse x, y or add commas)
    # Assume: t, x, y, p
    t = data[:, 0]
    x = data[:, 1].astype(int)
    y = data[:, 2].astype(int)
    p = data[:, 3].astype(int)
    return t, x, y, p

def load_groundtruth_txt(file_path):
    """
    Load a ground truth pose sequence from a text file.
    Expected format: [timestamp, x, y, z, qx, qy, qz, qw]
    """
    data = np.loadtxt(file_path)
    t = data[:, 0]
    # Pos: x, y, z
    pos = data[:, 1:4] 
    # Quat: qx, qy, qz, qw
    quat = data[:, 4:8]
    return t, pos, quat

def interpolate_groundtruth(gt_t, gt_pos, query_t):
    """
    Interpolates ground truth positions at query timestamps.
    """
    interp_func = scipy.interpolate.interp1d(gt_t, gt_pos, axis=0, kind='linear', fill_value='extrapolate')
    return interp_func(query_t)

def accumulate_events_to_frames(t, x, y, p, dt_window, width, height):
    """
    Bin sequential events into "event frames" representing a time window `dt_window`.
    Returns:
       frames: boolean numpy arrays shape (N_frames, height, width) indicating if an event occurred
       times: numpy array shape (N_frames) representing the end timestamp of the frame window
    """
    if len(t) == 0:
        return np.array([]), np.array([])
        
    t_start = t[0]
    t_end = t[-1]
    
    n_frames = max(1, int(np.ceil((t_end - t_start) / dt_window)))
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    times = t_start + np.arange(1, n_frames + 1) * dt_window
    
    # Calculate the frame index for each event
    frame_indices = np.floor((t - t_start) / dt_window).astype(int)
    
    # Clamp to avoid off-by-one at the very end
    frame_indices = np.clip(frame_indices, 0, n_frames - 1)
    
    # Efficient numpy accumulation using flat indices
    # We assign 255 to pixels that saw an event so it acts as an 8-bit image for OpenCV detectors
    flat_indices = frame_indices * (height * width) + y * width + x
    np.put(frames.ravel(), flat_indices, 255)
    
    return frames, times
