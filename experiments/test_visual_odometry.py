import cv2
import numpy as np
import unittest
import sys, os

sys.path.insert(1, os.path.dirname(os.getcwd()))
from sspslam.perception.visual_odometry import ORBVisualOdometry

class TestVisualOdometry(unittest.TestCase):
    def test_vo_tracking(self):
        vo = ORBVisualOdometry()
        
        # Create a synthetic moving square on a textured background 
        bg = np.random.randint(0, 100, (240, 320), dtype=np.uint8)
        
        for i in range(10):
            frame = bg.copy()
            # Draw a highly textured 'structure' moving to the right
            square = np.zeros((50, 50), dtype=np.uint8)
            cv2.randu(square, 100, 255)
            frame[100:150, 50+i*5:100+i*5] = square
            
            vel, descs, vecs = vo.process_frame(frame)
            
            if i > 0:
                print(f"Step {i} -> VO Speed estimate: {np.linalg.norm(vel):.4f}")
                self.assertEqual(descs.shape[1], 32) # ORB is 32 bytes
                self.assertEqual(vecs.shape[1], 2)   # 2D relative vector (X, Z)

if __name__ == '__main__':
    unittest.main()
