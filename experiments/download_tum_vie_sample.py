import h5py
import numpy as np
import os

def create_synthetic_tum_vie_subset(output_path, num_events=50000, duration=2.0):
    """
    Since the real TUM-VIE sequence 'mocap-1d-trans' is multiple Gigabytes,
    this script dynamically creates a perfectly structured synthetic subset (HDF5)
    to validate the ingestion layer natively offline.
    """
    print(f"Synthesizing TUM-VIE local test package: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        grp = f.create_group('events')
        
        # TUM-VIE stores 't' in microseconds
        t = np.sort(np.random.uniform(0.0, duration * 1e6, num_events)).astype(np.int64)
        time_sec = t / 1e6
        
        # To make it visualizable and trackable by Shi-Tomasi, generate a rigid moving grid!
        grid_x, grid_y = np.meshgrid(np.arange(25, 300, 40), np.arange(25, 200, 40))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        point_idx = np.random.randint(0, len(grid_points), num_events)
        
        # Simulate lateral camera pan (objects move up and right diagonally across field of view)
        vX, vY = 80.0, -30.0
        
        x = grid_points[point_idx, 0] + time_sec * vX + np.random.normal(0, 1.0, num_events)
        y = grid_points[point_idx, 1] + time_sec * vY + np.random.normal(0, 1.0, num_events)
        
        x = np.clip(x, 0, 345).astype(np.int16)
        y = np.clip(y, 0, 259).astype(np.int16)
        p = np.random.randint(0, 2, num_events).astype(np.int8)
        
        grp.create_dataset('t', data=t)
        grp.create_dataset('x', data=x)
        grp.create_dataset('y', data=y)
        grp.create_dataset('p', data=p)
        
    print(f"Successfully generated {num_events} events.")
    print("For full benchamrks, download genuine sequence 'mocap-1d-trans-events_left.h5' via the TUM-VIE portal.")

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tum_vie_synthetic.h5")
    create_synthetic_tum_vie_subset(target)
