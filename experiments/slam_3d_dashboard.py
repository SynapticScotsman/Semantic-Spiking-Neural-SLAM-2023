# %% [markdown]
# # Semantic 3D SLAM Dashboard
# This notebook visualizes the data collected from the 3D MiniWorld Environment
# and lets us query the learned Cognitive Map using text.

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# %% [markdown]
# ## 1. Load the 3D Trajectory and Visual Perception Data

# %%
DATA_DIR = "data/3d"

path = np.load(os.path.join(DATA_DIR, "path.npy"))
rgb_frames = np.load(os.path.join(DATA_DIR, "rgb_frames_subset.npy"))
landmark_positions = np.load(os.path.join(DATA_DIR, "landmark_positions.npy"))

print(f"Loaded {len(path)} path steps.")
print(f"Loaded {len(rgb_frames)} visual frames.")

# Plot the path and the objects
plt.figure(figsize=(6, 6))
plt.plot(path[:, 0], path[:, 1], color='gray', alpha=0.5, label='Agent Path')
plt.plot(path[0, 0], path[0, 1], 'go', label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'ro', label='End')

for i, pos in enumerate(landmark_positions):
    plt.scatter(pos[0], pos[1], s=150, marker='*', label=f'Object {i}')

plt.legend()
plt.title("True 2D Map")
plt.axis("equal")
plt.show()

# %% [markdown]
# ## 2. View the Agent's Perspective Over Time

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
num_frames = len(rgb_frames)
for i in range(4):
    idx = i * (num_frames // 4)
    axes[i].imshow(rgb_frames[idx])
    axes[i].axis('off')
    axes[i].set_title(f"Step {idx * 100}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Semantic Querying (Where is the red box?)
# Using CLIP and the Associative Memory to recall locations based on human text.
