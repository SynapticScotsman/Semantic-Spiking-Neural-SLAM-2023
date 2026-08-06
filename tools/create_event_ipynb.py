import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

title_cell = nbf.v4.new_markdown_cell("""\
# Event-Driven ORB Cognitive SLAM: Real Data Evaluation 🚀

This notebook downloads a tiny subset of the UZH-RPG Event Camera Dataset (the `shapes_6dof` sequence) and pushes it entirely through the native pure Python Event->ORB->SSP-SLAM pipeline.

This acts as a visual validation script to verify every individual step of the cognitive track mapping.
""")

imports_cell = nbf.v4.new_code_cell("""\
import os, sys, urllib.request, zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nengo

# Add root folder to sys path
sys.path.insert(1, os.path.dirname(os.getcwd()))
from sspslam.perception.event_dataset_parser import load_events_txt, accumulate_events_to_frames
from sspslam.perception.visual_odometry import ORBVisualOdometry
from sspslam.perception.event_vision_adapter import EventVisionAdapter
""")

download_cell = nbf.v4.new_code_cell("""\
DATASET_URL = "http://rpg.ifi.uzh.ch/datasets/davis/shapes_6dof.zip"
ZIP_PATH = "shapes_6dof.zip"
EXTRACT_DIR = "shapes_6dof"

if not os.path.exists(EXTRACT_DIR):
    print("Downloading UZH-RPG shapes sequence (approx. 158MB)...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Done!")
else:
    print("Dataset already downloaded.")
    
EVENTS_FILE = os.path.join(EXTRACT_DIR, "events.txt")
""")

parse_cell = nbf.v4.new_code_cell("""\
print("Loading raw events (timestamps, x, y, polarity)...")
# We only load a small subset (first 500,000 events which is ~0.5 seconds of data)
with open(EVENTS_FILE, 'r') as f:
    lines = f.readlines()[:500000] 
    
data = []
for line in lines:
    parts = line.strip().split()
    if len(parts) == 4:
        data.append([float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
data = np.array(data)

t = data[:, 0]
x = data[:, 1].astype(int)
y = data[:, 2].astype(int)
p = data[:, 3].astype(int)

print(f"Loaded {len(t)} events spanning {t[-1] - t[0]:.2f} seconds.")
""")

frame_cell = nbf.v4.new_code_cell("""\
# Condense asynchronous spike streams into sharp 50ms 2D frames
dt_window = 0.05 
width, height = 240, 180  # DAVIS240 resolution

# Function modified slightly to safely ignore boundary errors
frames, frame_times = accumulate_events_to_frames(t, x, y, p, dt_window=dt_window, width=width, height=height)

print(f"Constructed {frames.shape[0]} discrete Event Frames.")

# Visualize the first 3 frames
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axs[i].imshow(frames[i], cmap='gray')
    axs[i].set_title(f"Event Time-Surface {i+1}")
plt.show()
""")

vo_cell = nbf.v4.new_code_cell("""\
print("Running Native Python ORB Visual Odometry & Landmark Extraction...")

adapter = EventVisionAdapter(max_landmarks=20, view_radius=4.0)

# Pass the frames through ORB odometry and matching! Agent velocity = None to trigger VO solver
vel_data, vec_to_landmarks_data, feat_data = adapter.process_dataset(frames, agent_velocities=None)

print(f"Odometry Output Shape: {vel_data.shape}")
print(f"Extracted Semantic Feature Trajectories: {feat_data.shape}")

# Plot the absolute ego-motion velocities estimated
plt.figure(figsize=(8, 4))
plt.plot(vel_data[:, 0], label='Vx (lateral)')
plt.plot(vel_data[:, 1], label='Vz (forward)')
plt.title("Estimated Native Visual Odometry Velocities")
plt.xlabel("Frame")
plt.ylabel("Velocity Component")
plt.legend()
plt.show()
""")

nb['cells'] = [title_cell, imports_cell, download_cell, parse_cell, frame_cell, vo_cell]

with open('experiments/test_event_orb_slam.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 'experiments/test_event_orb_slam.ipynb' created successfully!")
