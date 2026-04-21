# Semantic Spiking Neural SLAM — Project Overview

---

## What Is This Project?

Imagine a robot navigating a building it has never seen before. As it moves around, it needs to answer two questions at the same time:

- *"Where am I?"*
- *"Where are the things I've seen?"*

This is called **SLAM** — Simultaneous Localisation and Mapping. This project builds a SLAM system that works the way the brain is thought to work, using **spiking neural networks** and a mathematical framework called **Vector Symbolic Architecture (VSA)**.

Instead of storing a traditional grid-based map, the system stores knowledge as **high-dimensional vectors** — long lists of numbers — that can be combined, compared, and queried using simple maths. The result is a system that can:

- Track where it is as it moves, using only its own movement speed as input
- Recognise landmarks it has seen before (objects, walls, colours)
- Build a memory of where each landmark is
- Answer questions like *"where is the red box?"* — even using plain English (via an AI vision model called CLIP)

---

## The Big Picture — How It Works

The system has four main stages that run together in real time:

```
1. MOVEMENT TRACKING       2. LANDMARK RECOGNITION
   "I moved this way"  →      "I can see that object"
         │                            │
         └──────────┬─────────────────┘
                    ▼
           3. MAP BUILDING
              "That object is over there — remember it"
                    │
                    ▼
           4. SELF-CORRECTION
              "My memory says I should be here — adjust"
```

**Stage 1 — Path Integration:** The agent tracks its own position by continuously integrating its speed and direction, like counting steps in the dark. This drifts over time, which is why the next stages are needed.

**Stage 2 — Landmark Perception:** When the agent sees a known object, it extracts a visual "fingerprint" from the image (using techniques like HOG, SIFT, or CLIP) and converts it into a vector that represents that object's identity.

**Stage 3 — Map Building:** The system stores pairs of *"who"* and *"where"* in an associative memory — a neural network that learns to recall locations when given an identity. Over time, it builds up a complete map.

**Stage 4 — Self-Correction:** When the agent recognises a landmark whose location it already knows, it uses that knowledge to correct its position estimate. This keeps drift from accumulating.

---

## What Makes This Biologically Inspired?

The system is modelled on structures found in the mammalian brain:

| Brain Structure | What It Does | This Project's Equivalent |
|----------------|--------------|--------------------------|
| **Grid cells** | Fire in repeating hexagonal patterns as you move; encode position | `HexagonalSSPSpace` — the maths behind position vectors |
| **Place cells** | Fire when you're in a specific location | The decoded output of the path integrator |
| **Object vector cells** | Fire when a specific object is at a specific angle and distance | The landmark vector encoding in `SLAMNetwork` |
| **Hippocampus** | Stores associations between places and things | `AssociativeMemory` — the neural map |

All of this runs inside **Nengo**, a Python framework for simulating spiking neural networks.

---

## Project Structure — What's Where

```
Semantic-Spiking-Neural-SLAM-2023/
│
├── sspslam/              ← The core library (the actual SLAM system)
│   ├── sspspace.py       ← The maths for encoding positions and symbols as vectors
│   ├── networks/         ← The neural network models (path integrator, memory, etc.)
│   ├── perception/       ← Vision tools (image features, CLIP, event cameras)
│   ├── environments/     ← A 3D test room the agent can move around in
│   └── utils/            ← Plotting and helper tools
│
├── experiments/          ← Scripts and notebooks you can run
│   ├── run_slam_features.py        ← Best starting point for running SLAM
│   ├── run_semantic_slam.py        ← SLAM with object class labels (no GPU needed)
│   ├── slam_map_new.py             ← Full 2D demo with walls and objects
│   ├── collect_3d_data.py          ← Collect new data from the 3D environment
│   ├── run_slam_3d.py              ← Run the full 3D pipeline end-to-end
│   ├── example_slam_walkthrough.ipynb  ← Step-by-step interactive tutorial ← START HERE
│   ├── slam_3d_dashboard.ipynb         ← Visualise and query a saved SLAM run
│   └── test_*.py                       ← Automated checks to verify things work
│
├── data/3d/              ← Pre-collected data (ready to use, no setup needed)
│   ├── path.npy                    ← Where the agent went (x, y positions)
│   ├── velocities.npy              ← How fast the agent moved at each step
│   ├── feature_vectors.npy         ← Visual fingerprints of each object at each step
│   ├── landmark_positions.npy      ← True positions of the objects in the room
│   └── slam_features_*.npz         ← Saved results from a previous SLAM run
│
├── PROJECT_OVERVIEW.md   ← This file
└── README.md             ← Technical theory and installation instructions
```

---

## Key Ideas Explained Simply

### Vectors as Memory

At the heart of this system, everything — positions, object identities, memories — is stored as a **vector**: a long list of numbers (typically 97 to 151 numbers long). This might seem strange, but it has powerful properties:

- Two similar things (nearby positions, similar-looking objects) produce vectors with a **high dot product** (they "point in the same direction")
- Two unrelated things produce vectors with a **low dot product** (they are nearly perpendicular)
- You can **combine** two vectors by binding them together, and later **pull them apart** to retrieve what was stored

### Binding — The Key Operation

Binding is how the system links two pieces of information:

```
bind("red box identity", "position near the door")
→ a single vector that encodes both facts together
```

Later, you can unbind:

```
unbind("red box identity", memory)
→ recovers "position near the door"
```

This is what allows the system to answer *"where is the red box?"* purely through vector arithmetic — no lookup tables, no databases.

### The Semantic Query

Because each object's identity is stored as a vector built from its visual appearance *and* its class label, you can also ask:

```
"Where are all the boxes?"
→ sum up the bound vectors for everything labelled 'box'
→ the result points toward the average location of all boxes
```

With CLIP (an AI model that understands images and text together), you can go further and ask using plain English — *"a red box"* — and get back a spatial heatmap showing where the system thinks it is.

---

## How To Get Started

### Option 1 — No installation beyond Python (easiest)

You can run the basic tests and a simplified SLAM experiment with nothing more than the core packages:

```bash
pip install nengo nengo_spa numpy scipy matplotlib

# Check everything is working
python experiments/test_semantic_encoding.py
python experiments/test_feature_extraction.py

# Run a SLAM experiment (CPU only, generates its own synthetic data)
python experiments/run_semantic_slam.py --backend cpu
```

### Option 2 — Use the pre-collected data (recommended next step)

The `data/3d/` folder contains a ready-to-use dataset collected from the 3D test room. You can run SLAM on it without setting up the 3D environment:

```bash
python experiments/run_slam_features.py \
    --feature-data data/3d/feature_vectors.npy \
    --pos-data     data/3d/path.npy \
    --vel-data     data/3d/velocities.npy \
    --vec-data     data/3d/vec_to_landmarks.npy \
    --save-dir     data/3d
```

### Option 3 — Interactive notebook tutorial (best for learning)

Open `experiments/example_slam_walkthrough.ipynb` in Jupyter. It walks through every concept step by step with visualisations — no prior knowledge required.

### Option 4 — Full 3D pipeline (needs extra packages)

If you have `gymnasium` and `miniworld` installed, you can collect fresh data and run the complete pipeline:

```bash
python experiments/run_slam_3d.py --policy explore --n-steps 2000
```

---

## What Each Experiment Does

| Script / Notebook | What It Does | What You Need |
|-------------------|-------------|---------------|
| `example_slam_walkthrough.ipynb` | Step-by-step tutorial — SSPs, SPs, encoding, querying | Just numpy + nengo |
| `test_semantic_encoding.py` | Checks the semantic encoder works correctly | Just numpy |
| `test_feature_extraction.py` | Checks image feature extraction | Just numpy |
| `run_semantic_slam.py` | Full SLAM with object class labels | numpy + nengo |
| `run_slam_features.py` | Full SLAM with visual features | numpy + nengo |
| `slam_3d_dashboard.ipynb` | Interactive map viewer for saved results | numpy + matplotlib |
| `slam_map_new.py` | Detailed 2D demo with walls and query plots | nengo + nengo_ocl (GPU) |
| `run_slam_3d.py` | Collect 3D data then run SLAM | + miniworld |
| `run_event_slam.py` | SLAM using neuromorphic event-camera data | numpy + nengo |
| `run_event_orb_slam.py` | Event SLAM with visual odometry | + opencv |
| `run_miniworld_slam.py` | SLAM in a 3D room with ORB visual tracking | + miniworld + opencv |

**Traffic light summary:**

| Colour | Meaning |
|--------|---------|
| 🟢 | Works right now with the core install |
| 🟡 | Works once you install one extra package |
| 🔴 | Needs multiple optional packages |

| Script | Status |
|--------|--------|
| `test_semantic_encoding.py` | 🟢 |
| `test_feature_extraction.py` | 🟢 |
| `test_event_pipeline.py` | 🟢 |
| `run_semantic_slam.py` | 🟢 |
| `run_slam_features.py` | 🟢 |
| `run_event_slam.py` | 🟢 |
| `slam_map_new.py` | 🟡 needs `nengo_ocl` |
| `test_visual_odometry.py` | 🟡 needs `opencv-python` |
| `run_event_orb_slam.py` | 🟡 needs `opencv-python` |
| `collect_3d_data.py` | 🔴 needs `miniworld` + `gymnasium` |
| `run_miniworld_slam.py` | 🔴 needs `miniworld` + `opencv` |

---

## Optional Add-Ons

The core system works without these, but they unlock additional capabilities:

| Package | What It Unlocks | Install Command |
|---------|----------------|-----------------|
| `nengo_ocl` | GPU acceleration (much faster simulations) | `pip install nengo_ocl` |
| `opencv-python` | SIFT & ORB image features, visual odometry | `pip install opencv-python` |
| `scikit-image` | HOG image features | `pip install scikit-image` |
| `torch` + `transformers` | CLIP — natural language map queries | `pip install torch transformers` |
| `gymnasium` + `miniworld` | 3D test environment for data collection | `pip install gymnasium miniworld` |
| `tensorflow` | Neural network-based SSP decoder (advanced) | `pip install tensorflow` |
| `nengo_loihi` | Deploy on Intel Loihi neuromorphic chip | See Nengo docs |

---

## How The Neural Network Architecture Fits Together

```
┌───────────────────────────────────────────���─────────────┐
│                     SLAM Network                        │
│                                                         │
│  Agent speed ──► Path Integrator ──► Position estimate  │
│      (velocity)    (VCO neurons)         (SSP vector)   │
│                          │                    ▲          │
│                          │                    │ correct  │
│                          ▼                    │          │
│  Seen object ──► Bind position + ──► Associative Memory │
│   (identity SP)   object vector     (learns ID→location)│
│                                               │          │
│                                               ▼          │
│                                        Recalled location │
│                                        ──► Unbind ──► Δ │
└─────────────────────────────────────────────────────────┘
```

In plain English:
1. The agent's velocity feeds into a **path integrator** that tracks position as a vector
2. When an object is spotted, the system creates a vector representing *"this object is X metres in that direction"*
3. These are stored in an **associative memory** that learns over time
4. When a known object is seen again, the memory recalls where it should be — which is used to **correct any drift** in the position estimate

---

## Known Limitations

- **No per-object camera crops:** The 3D environment gives one wide-angle image per step, not separate images per object. The system works around this by using object labels and small random offsets to give each object a distinct identity vector.

- **Single object per step:** The main experiment only processes the closest visible object at any moment. Code exists to handle multiple objects simultaneously but hasn't been wired into the experiments yet.

- **Monocular depth:** The visual odometry module estimates motion from a single camera, which means depth is ambiguous. A fixed scale factor is used as a workaround.

- **Event camera pipeline:** The neuromorphic event-camera experiments are partially implemented — the data loading and frame accumulation work, but the full end-to-end pipeline is not yet complete.

---

## Further Reading

- `README.md` — the original technical documentation with mathematical background
- `experiments/example_slam_walkthrough.ipynb` — hands-on interactive tutorial
- `experiments/slam_3d_dashboard.ipynb` — visual explorer for saved results
