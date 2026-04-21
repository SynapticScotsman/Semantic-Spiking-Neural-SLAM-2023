import numpy as np
import cv2
import sys
import os

# Ensure sspslam is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sspslam.perception.event_odometry import EventOdometry

def test_event_odometry_synthetic():
    # Create simple synthetic test frames simulating moving edges
    evo = EventOdometry(focal_length=500.0, pp=(250.0, 250.0))
    
    # Base synthetic image 500x500
    base_img = np.zeros((500, 500), dtype=np.uint8)
    
    # Paint some corners
    cv2.rectangle(base_img, (100, 100), (200, 200), (255,), cv2.FILLED)
    cv2.rectangle(base_img, (300, 300), (450, 450), (200,), cv2.FILLED)
    cv2.rectangle(base_img, (100, 400), (150, 450), (150,), cv2.FILLED)
    cv2.rectangle(base_img, (400, 100), (450, 150), (255,), cv2.FILLED)
    
    # Add noise for texture
    noise = np.random.randint(0, 50, (500, 500), dtype=np.uint8)
    base_img = cv2.add(base_img, noise)
    
    print("Testing initialization frame...")
    vel, rot = evo.process_frame(base_img)
    assert np.all(vel == 0), "Initialization velocity should be 0"
    
    # Frame 2: simulate panning the camera to the left (content moves right)
    M1 = np.float32([[1, 0, 5], [0, 1, 0]]) # shift right by 5 pixels
    img2 = cv2.warpAffine(base_img, M1, (500, 500))
    img2 = cv2.add(img2, np.random.randint(0, 50, (500, 500), dtype=np.uint8))
    
    print("Testing Frame 2 (Shift right by 5px)...")
    vel, rot = evo.process_frame(img2)
    print(f"Velocity: {vel}, Rot: {rot}")
    
    # Frame 3: simulate panning further right 
    M2 = np.float32([[1, 0, 10], [0, 1, 0]])
    img3 = cv2.warpAffine(base_img, M2, (500, 500))
    img3 = cv2.add(img3, np.random.randint(0, 50, (500, 500), dtype=np.uint8))
    
    print("Testing Frame 3 (Shift right by 10px)...")
    vel, rot = evo.process_frame(img3)
    print(f"Velocity: {vel}, Rot: {rot}")
    
    assert vel is not None, "Velocity should not be None"
    
    print("Event Odometry tests passed.")

if __name__ == "__main__":
    test_event_odometry_synthetic()
