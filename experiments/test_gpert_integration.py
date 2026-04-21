import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sspslam.perception.event_vision_adapter import EventVisionAdapter

def test_gpert_backend():
    print("Initializing EventVisionAdapter with backend='gpert'...")
    adapter = EventVisionAdapter(max_landmarks=10, backend='gpert')
    
    # Create fake event frames
    print("Generating synthetic event frames and depth maps...")
    frames = []
    depths = []
    
    # Small square moving left and getting "closer" (depth decreases)
    for t in range(5):
        frame = np.zeros((100, 100), dtype=np.uint8)
        # Shift a small bright square
        x = 50 + t * 5
        frame[40:60, x:x+20] = 255
        
        # Add some noise
        frame += np.random.randint(0, 20, (100, 100), dtype=np.uint8)
        
        frames.append(frame)
        
        # Depth maps mock: A real simulator would render 100x100 metric depth.
        # Let's mock a depth map where the center is closer (true depth = 3.0 - t/2)
        depth_map = np.ones((100, 100), dtype=np.float32) * 10.0 # background far
        depth_map[40:60, x:x+20] = 3.0 - (t * 0.5)
        depths.append(depth_map)
        
    print("Processing dataset...")
    vel_arr, vec_arr, feat_arr = adapter.process_dataset(frames, landmark_depths=depths)
    
    print(f"Results:")
    print(f"Velocity array shape: {np.array(vel_arr).shape}")
    print(f"Vector array shape: {np.array(vec_arr).shape}")
    print(f"Features array shape: {np.array(feat_arr).shape}")
    
    # Check if depths were updated to reflect the mock GPERT depths instead of flat 1.0 dummy depth
    print(f"Sample Vector tracked in Frame 0 (should have magnitude ~3.0): {np.linalg.norm(vec_arr[0, 0])}")
    print(f"Sample Vector tracked in Frame 4 (should have magnitude ~1.0): {np.linalg.norm(vec_arr[4, 0])}")
    
    assert np.array(vel_arr).shape == (5, 2), "Velocity array shape incorrect."
    assert np.array(vec_arr).shape == (5, 10, 2), "Vector array shape incorrect."
    
    print("Integration test passed successfully!")

if __name__ == "__main__":
    test_gpert_backend()
