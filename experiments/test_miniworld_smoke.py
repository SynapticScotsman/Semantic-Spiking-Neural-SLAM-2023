"""
Smoke test: verify MiniWorld renders and basic agent movement works.
Runs headless (no window) and saves a sample frame as PNG.
"""
import numpy as np
import gymnasium as gym
import miniworld

# Create a simple one-room environment
env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)

print(f"Environment created: MiniWorld-OneRoom-v0")
print(f"Observation shape: {obs.shape}")  # Expected: (60, 80, 3)
print(f"Action space: {env.action_space}")

# Take a few steps and collect frames
frames = [obs]
positions = []
for i in range(20):
    # Action 0 = turn left, 1 = turn right, 2 = move forward
    action = 2 if i % 3 != 0 else 0  # mostly move forward, sometimes turn
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(obs)
    
    # Get agent position from the unwrapped env
    agent_pos = env.unwrapped.agent.pos
    positions.append([agent_pos[0], agent_pos[2]])  # x, z (2D ground plane)
    
    if terminated or truncated:
        obs, info = env.reset()

positions = np.array(positions)
print(f"\nCollected {len(frames)} frames")
print(f"Agent moved from {positions[0]} to {positions[-1]}")
print(f"Frame dtype: {frames[0].dtype}, range: [{frames[0].min()}, {frames[0].max()}]")

# Save a sample frame
try:
    from PIL import Image
    img = Image.fromarray(frames[-1])
    img.save("experiments/miniworld_test_frame.png")
    print(f"Saved sample frame to experiments/miniworld_test_frame.png")
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave("experiments/miniworld_test_frame.png", frames[-1])
    print(f"Saved sample frame to experiments/miniworld_test_frame.png")

env.close()
print("\n[OK] MiniWorld smoke test PASSED!")
