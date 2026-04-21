import h5py
import numpy as np

class TumVieParser:
    """
    Parses TUM-VIE (Visual-Inertial Event) datasets (used directly by GPERT).
    Extracts asynchronous neuromorphic data from standard .h5 sequences.
    """
    def __init__(self, h5_filepath, gt_filepath=None):
        self.h5_filepath = h5_filepath
        self.gt_filepath = gt_filepath
        
    def stream_event_windows(self, dt_window=0.05, max_windows=None):
        """
        Lazily yields temporal event windows (frames) from the asynchronous HDF5 data.
        dt_window: Window size in seconds.
        """
        print(f"Opening TUM-VIE Dataset: {self.h5_filepath}")
        
        try:
            with h5py.File(self.h5_filepath, 'r') as f:
                # TUM-VIE format usually stores 'events/t', 'events/x', 'events/y', 'events/p'
                if 'events' in f:
                    t_group = f['events']['t']
                    x_group = f['events']['x']
                    y_group = f['events']['y']
                    p_group = f['events']['p']
                else:
                    # Generic fallback handling
                    t_group = f['t']
                    x_group = f['x']
                    y_group = f['y']
                    p_group = f['p']

                # The timestamps are usually in microseconds from the TUM-VIE spec
                # We will normalize to seconds starting from 0
                t_offset = float(t_group[0])
                
                chunk_size = 500000 
                num_events = t_group.shape[0]
                
                current_window = 0.0
                window_events = {'x': [], 'y': [], 'p': [], 't': []}
                
                windows_yielded = 0
                
                for i in range(0, num_events, chunk_size):
                    end_idx = min(i + chunk_size, num_events)
                    
                    t_chunk = (t_group[i:end_idx] - t_offset) / 1e6 # Convert us to seconds
                    x_chunk = x_group[i:end_idx]
                    y_chunk = y_group[i:end_idx]
                    p_chunk = p_group[i:end_idx]
                    
                    for j in range(len(t_chunk)):
                        if t_chunk[j] >= current_window + dt_window:
                            # Yield compiled window
                            yield np.array(window_events['x']), np.array(window_events['y']), np.array(window_events['p']), np.array(window_events['t'])
                            windows_yielded += 1
                            
                            if max_windows and windows_yielded >= max_windows:
                                return
                                
                            current_window += dt_window
                            window_events = {'x': [], 'y': [], 'p': [], 't': []}
                            
                        window_events['x'].append(x_chunk[j])
                        window_events['y'].append(y_chunk[j])
                        window_events['p'].append(p_chunk[j])
                        window_events['t'].append(t_chunk[j])

        except Exception as e:
            print(f"Error reading TUM-VIE HDF5: {e}")
            raise
